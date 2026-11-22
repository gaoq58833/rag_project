"""Wiki service 编排测试（mock DB/Qdrant/LLM，不依赖真实服务与 torch）。

覆盖：全量重建落库+建索引、任务去重、检索融合排序、关系图边过滤、单页重写、溯源解析。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wiki import service  # noqa: E402
from src.wiki.generator import WikiPageDraft  # noqa: E402


class FakeRepo:
    """收集调用记录的假仓库。"""
    def __init__(self):
        self.calls = []
    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return method


class FakeVectorStore:
    def __init__(self):
        self.deleted_user = []
        self.deleted_ids = []
        self.upserts = []
    def delete_by_user(self, user_id):
        self.deleted_user.append(user_id)
    def delete_by_ids(self, ids):
        self.deleted_ids.extend(ids)
    def upsert(self, ids, vecs, payloads):
        self.upserts.append((list(ids), list(vecs), list(payloads)))
    def search(self, query_vec, top_k=None, filter_=None):
        return getattr(self, "_hits", [])


class FakeLLM:
    def __init__(self):
        self.embed_calls = []
        self.embed_query_calls = []
    def embed(self, texts, batch_size=20):
        self.embed_calls.append(list(texts))
        return [[0.1] * 4 for _ in texts]
    def embed_query(self, text):
        self.embed_query_calls.append(text)
        return [0.2] * 4


def install(fake_wiki_repo=None, fake_job_repo=None, fake_chunk_repo=None,
            fake_vs=None, fake_llm=None, fake_generate=None, fake_single=None,
            fake_bm25=None, fake_tokenize=None):
    service.WikiPageRepository = fake_wiki_repo if fake_wiki_repo is not None else FakeRepo()
    service.WikiJobRepository = fake_job_repo if fake_job_repo is not None else FakeRepo()
    service.ChunkRepository = fake_chunk_repo if fake_chunk_repo is not None else FakeRepo()
    service.wiki_vector_store = fake_vs if fake_vs is not None else FakeVectorStore()
    service.llm_client = fake_llm if fake_llm is not None else FakeLLM()
    service.generate_wiki = fake_generate or (lambda user_id, on_progress=None: [])
    service.generate_single_page = fake_single or (lambda *a, **k: None)
    service.wiki_bm25_search = fake_bm25 or (lambda *a, **k: [])
    service.tokenize = fake_tokenize or (lambda text: "tok")
    service._clear_chat_cache_safe = lambda: None  # 避免测试中惰性导入 src.rag 拉起 torch


def case(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return False


# ---------- 测试用例 ----------

def test_process_full_rebuild():
    drafts = [
        WikiPageDraft(slug="rnn", title="循环神经网络", summary="seq", content_md="RNN内容",
                      paragraphs=[{"text": "p", "chunk_ids": ["c1"]}], outgoing_links=["cnn"],
                      source_chunk_ids=["c1"], status="ready"),
        WikiPageDraft(slug="cnn", title="卷积神经网络", summary="img", content_md="CNN内容",
                      paragraphs=[{"text": "p2", "chunk_ids": ["c2"]}], outgoing_links=[],
                      source_chunk_ids=["c2"], status="ready"),
    ]
    wiki_repo = FakeRepo()
    job_repo = FakeRepo()
    vs = FakeVectorStore()
    llm = FakeLLM()

    def fake_gen(user_id, on_progress=None):
        on_progress(2, 2)
        return drafts
    install(wiki_repo, job_repo, fake_vs=vs, fake_llm=llm, fake_generate=fake_gen)

    service.process_wiki_generation("job1", "u1")

    # 全量重建：清空旧页 + 旧向量
    assert any(c[0] == "delete_by_user" for c in wiki_repo.calls), "应清空旧页面"
    assert "u1" in vs.deleted_user, "应清空旧 Wiki 向量"
    # 每页都 upsert
    upserts = [c for c in wiki_repo.calls if c[0] == "upsert"]
    assert len(upserts) == 2, f"应写 2 页，实际 {len(upserts)}"
    slugs_written = {c[2].get("slug") for c in upserts}
    assert slugs_written == {"rnn", "cnn"}, slugs_written
    # 向量批量写入（2 个 ready 页）
    assert len(vs.upserts) == 1, f"应批量 upsert 一次向量，实际 {len(vs.upserts)}"
    assert len(vs.upserts[0][0]) == 2, vs.upserts[0][0]
    assert llm.embed_calls and len(llm.embed_calls[0]) == 2
    # 任务标记 done，进度已上报
    assert any(c[0] == "mark_done" for c in job_repo.calls), "应标记任务 done"
    assert any(c[0] == "update_progress" for c in job_repo.calls), "应上报进度"


def test_submit_dedup():
    job_repo = FakeRepo()
    # 模拟已有进行中任务
    job_repo.get_active = lambda user_id: {"id": "existing-job", "status": "generating"}
    create_calls = []
    job_repo.create = lambda *a, **k: create_calls.append(a) or "new-job"
    install(fake_job_repo=job_repo)

    info = service.submit_wiki_generation("u1")
    assert info["reused"] is True, info
    assert info["job_id"] == "existing-job"
    assert not create_calls, "已有进行中任务时不应创建新任务"

    # 无 active 任务时创建
    job_repo.get_active = lambda user_id: None
    info2 = service.submit_wiki_generation("u1")
    assert info2["reused"] is False
    assert info2["job_id"] == "new-job"
    assert len(create_calls) == 1


def test_retrieve_fusion():
    llm = FakeLLM()
    vs = FakeVectorStore()
    # 向量路命中 rnn, cnn；BM25 路命中 cnn（cnn 两路都命中 → RRF 分更高）
    vs._hits = [
        {"payload": {"slug": "rnn"}, "score": 0.9},
        {"payload": {"slug": "cnn"}, "score": 0.8},
    ]
    def fake_bm25(query, user_id, top_k=10):
        return [{"slug": "cnn", "rank": 1.0}]

    wiki_repo = FakeRepo()
    pages_db = {
        "rnn": {"slug": "rnn", "title": "RNN", "summary": "s1", "content_md": "c1"},
        "cnn": {"slug": "cnn", "title": "CNN", "summary": "s2", "content_md": "c2"},
    }
    wiki_repo.get_by_slug = lambda user_id, slug: pages_db.get(slug)
    install(fake_wiki_repo=wiki_repo, fake_vs=vs, fake_llm=llm, fake_bm25=fake_bm25)

    results = service.retrieve_wiki("神经网络", "u1", top_k=2)
    slugs = [r["slug"] for r in results]
    assert slugs[0] == "cnn", f"cnn 两路命中应排第一，实际顺序 {slugs}"
    assert set(slugs) == {"cnn", "rnn"}


def test_build_graph_filters_missing():
    wiki_repo = FakeRepo()
    wiki_repo.get_all_by_user = lambda user_id: [
        {"slug": "a", "title": "A", "status": "ready", "outgoing_links": ["b", "ghost"]},
        {"slug": "b", "title": "B", "status": "ready", "outgoing_links": ["a"]},
    ]
    install(fake_wiki_repo=wiki_repo)
    g = service.build_graph("u1")
    slugs = {n["slug"] for n in g["nodes"]}
    assert slugs == {"a", "b"}
    edge_pairs = {(e["from"], e["to"]) for e in g["edges"]}
    assert ("a", "b") in edge_pairs
    assert ("a", "ghost") not in edge_pairs, "指向不存在页面的边应被过滤"


def test_regenerate_page():
    wiki_repo = FakeRepo()
    wiki_repo.get_by_slug = lambda user_id, slug: {
        "slug": "rnn", "title": "RNN", "summary": "s",
        "source_chunk_ids": ["c1", "c2"], "outgoing_links": [],
    }
    wiki_repo.list_by_user = lambda user_id: [
        {"title": "RNN"}, {"title": "CNN"},
    ]
    chunk_repo = FakeRepo()
    chunk_repo.get_sources_by_chunk_ids = lambda user_id, ids: [
        {"point_id": "c1", "content": "chunk1", "section": "1.1", "filename": "a.pdf", "document_id": "d1"},
    ]
    vs = FakeVectorStore()
    llm = FakeLLM()

    def fake_single(title, summary, source_chunks, all_page_titles):
        assert title == "RNN"
        assert "CNN" in all_page_titles
        return WikiPageDraft(slug="rnn", title="RNN", summary="s", content_md="新内容",
                             paragraphs=[{"text": "np", "chunk_ids": ["c1"]}],
                             outgoing_links=[], source_chunk_ids=["c1"], status="ready")
    install(fake_wiki_repo=wiki_repo, fake_chunk_repo=chunk_repo, fake_vs=vs,
            fake_llm=llm, fake_single=fake_single)

    res = service.regenerate_page("u1", "rnn")
    assert res["status"] == "ready"
    # 先删后插该页向量
    assert any("u1::rnn" in ids for ids in vs.deleted_ids), vs.deleted_ids
    assert vs.upserts and vs.upserts[0][0] == ["u1::rnn"]
    # 写了库
    assert any(c[0] == "upsert" for c in wiki_repo.calls)


def test_get_page_detail_sources():
    wiki_repo = FakeRepo()
    wiki_repo.get_by_slug = lambda user_id, slug: {
        "slug": "rnn", "title": "RNN", "summary": "s", "content_md": "body",
        "paragraphs": [], "outgoing_links": [], "status": "ready",
        "source_chunk_ids": ["c1"], "error_msg": None, "updated_at": None,
    }
    chunk_repo = FakeRepo()
    chunk_repo.get_sources_by_chunk_ids = lambda user_id, ids: [
        {"qdrant_point_id": "c1", "content": "原始内容", "section": "1.1",
         "filename": "a.pdf", "chunk_index": 0},
    ]
    install(fake_wiki_repo=wiki_repo, fake_chunk_repo=chunk_repo)

    detail = service.get_page_detail("u1", "rnn")
    assert detail is not None
    assert detail["slug"] == "rnn"
    assert len(detail["sources"]) == 1
    assert detail["sources"][0]["filename"] == "a.pdf"
    assert detail["sources"][0]["content"] == "原始内容"


if __name__ == "__main__":
    print("== Wiki service 编排测试 ==")
    results = [
        case("process 全量重建：清旧+落库+建向量索引+任务done", test_process_full_rebuild),
        case("submit 任务去重", test_submit_dedup),
        case("retrieve 向量+BM25 融合排序", test_retrieve_fusion),
        case("build_graph 过滤不存在的链接", test_build_graph_filters_missing),
        case("regenerate_page 单页重写+刷新向量", test_regenerate_page),
        case("get_page_detail 溯源来源解析", test_get_page_detail_sources),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)

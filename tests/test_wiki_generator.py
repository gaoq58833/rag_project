"""Wiki 生成内核测试（mock 数据，不依赖 PG/Qdrant/LLM 网络）。

覆盖：聚类分组、批量命名、单页生成、溯源校验（非法引用丢弃 / 全非法则 failed）、
互链白名单（非白名单链接丢弃）、无数据返回空、进度回调。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wiki import generator  # noqa: E402

# ---- 构造 mock 数据：2 个语义簇（RNN / CNN）----
FAKE_ROWS = [
    {"qdrant_point_id": "rnn1", "content": "循环神经网络RNN用于处理序列数据", "section": "1.1", "document_id": "d1", "filename": "dl.pdf"},
    {"qdrant_point_id": "rnn2", "content": "RNN通过隐藏状态在时间步间传递信息", "section": "1.2", "document_id": "d1", "filename": "dl.pdf"},
    {"qdrant_point_id": "cnn1", "content": "卷积神经网络CNN用于图像识别任务", "section": "2.1", "document_id": "d2", "filename": "cv.pdf"},
    {"qdrant_point_id": "cnn2", "content": "CNN通过卷积核提取图像局部特征", "section": "2.2", "document_id": "d2", "filename": "cv.pdf"},
]
FAKE_VECTORS = {
    "rnn1": [1.0, 0.0, 0.0],
    "rnn2": [0.99, 0.01, 0.0],
    "cnn1": [0.0, 1.0, 0.0],
    "cnn2": [0.01, 0.99, 0.0],
}


class FakeLLM:
    """按标题路由的 chat_json；可配置每页返回内容。"""
    def __init__(self, page_responses, naming_response):
        self.page_responses = page_responses  # {title: dict}
        self.naming_response = naming_response
        self.calls = []

    def chat_json(self, messages, schema_hint="", **kwargs):
        self.calls.append(schema_hint)
        if "pages" in schema_hint:
            return self.naming_response
        # page 调用：从 user 消息里提取拟定标题
        user_text = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        title = None
        for line in user_text.splitlines():
            if line.startswith("拟定标题："):
                title = line.replace("拟定标题：", "").strip()
                break
        return self.page_responses.get(title, {"paragraphs": [], "outgoing_links": []})


def install(fake_llm, rows=FAKE_ROWS, vectors=FAKE_VECTORS):
    generator.ChunkRepository.get_wiki_eligible_chunks = lambda user_id: list(rows)
    generator.vector_store.fetch_vectors = lambda ids: {k: v for k, v in vectors.items() if k in ids}
    generator.llm_client.chat_json = fake_llm.chat_json


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

def test_basic_generation():
    naming = {"pages": [
        {"index": 0, "title": "循环神经网络", "summary": "处理序列数据的网络"},
        {"index": 1, "title": "卷积神经网络", "summary": "图像识别的网络"},
    ]}
    pages = {
        "循环神经网络": {
            "paragraphs": [{"text": "RNN 用于序列建模。", "chunk_ids": ["rnn1", "rnn2"]}],
            "outgoing_links": ["卷积神经网络", "不存在的页面"],
        },
        "卷积神经网络": {
            "paragraphs": [{"text": "CNN 用于图像识别。", "chunk_ids": ["cnn1"]}],
            "outgoing_links": ["循环神经网络"],
        },
    }
    fake = FakeLLM(pages, naming)
    install(fake)
    progress = []
    drafts = generator.generate_wiki("u1", on_progress=lambda d, t: progress.append((d, t)))

    assert len(drafts) == 2, f"期望 2 页，得到 {len(drafts)}"
    titles = sorted(d.title for d in drafts)
    assert titles == ["卷积神经网络", "循环神经网络"], titles

    rnn = next(d for d in drafts if d.title == "循环神经网络")
    cnn = next(d for d in drafts if d.title == "卷积神经网络")
    assert rnn.status == "ready" and cnn.status == "ready"
    # 溯源：只保留合法 chunk_id
    assert rnn.source_chunk_ids == ["rnn1", "rnn2"], rnn.source_chunk_ids
    assert cnn.source_chunk_ids == ["cnn1"], cnn.source_chunk_ids
    # 互链白名单：非白名单的“不存在的页面”被丢弃，白名单内的解析为 slug
    assert "卷积神经网络" in rnn.outgoing_links or any("卷积" in x for x in rnn.outgoing_links), rnn.outgoing_links
    assert not any("不存在" in x for x in rnn.outgoing_links), rnn.outgoing_links
    assert rnn.content_md.strip() != ""
    # slug 唯一性 + 非空
    assert rnn.slug and cnn.slug and rnn.slug != cnn.slug
    # 进度回调：最终 (2,2)
    assert progress and progress[-1] == (2, 2), progress


def test_invalid_citation_dropped():
    naming = {"pages": [{"index": 0, "title": "循环神经网络", "summary": "x"}]}
    pages = {
        "循环神经网络": {
            "paragraphs": [
                {"text": "有效段", "chunk_ids": ["rnn1"]},
                {"text": "引用了不存在的id", "chunk_ids": ["evil_id"]},
                {"text": "空引用段", "chunk_ids": []},
            ],
            "outgoing_links": [],
        },
    }
    fake = FakeLLM(pages, naming)
    install(fake)
    drafts = generator.generate_wiki("u1")
    # 只有 rnn 簇的页面（cnn 簇因 pages 里没给返回会得到空 paragraphs → 但本测试只关心 rnn 校验）
    rnn = next((d for d in drafts if d.title == "循环神经网络"), None)
    assert rnn is not None
    assert rnn.status == "ready", rnn.error_msg
    assert len(rnn.paragraphs) == 1, rnn.paragraphs
    assert rnn.paragraphs[0]["text"] == "有效段"
    assert rnn.source_chunk_ids == ["rnn1"]


def test_all_invalid_citations_failed():
    naming = {"pages": [{"index": 0, "title": "坏页面", "summary": "x"}]}
    pages = {
        "坏页面": {
            "paragraphs": [{"text": "这段没有合法引用", "chunk_ids": ["nope"]}],
            "outgoing_links": [],
        },
    }
    fake = FakeLLM(pages, naming)
    install(fake)
    drafts = generator.generate_wiki("u1")
    bad = next((d for d in drafts if d.title == "坏页面"), None)
    assert bad is not None
    assert bad.status == "failed", bad.status
    assert bad.error_msg, "failed 页面应有错误信息"


def test_empty_data():
    fake = FakeLLM({}, {"pages": []})
    install(fake, rows=[], vectors={})
    drafts = generator.generate_wiki("u2")
    assert drafts == [], f"无数据应返回空列表，得到 {drafts}"


def test_naming_failure_fallback():
    """命名 LLM 抛错时退回通用标题，仍能生成页面（不阻塞）。"""
    class BadNamingLLM(FakeLLM):
        def chat_json(self, messages, schema_hint="", **kwargs):
            if "pages" in schema_hint:
                raise ValueError("naming boom")
            return super().chat_json(messages, schema_hint, **kwargs)

    pages = {
        "知识页 1": {"paragraphs": [{"text": "通用标题页内容", "chunk_ids": ["rnn1"]}], "outgoing_links": []},
        "知识页 2": {"paragraphs": [{"text": "另一页", "chunk_ids": ["cnn1"]}], "outgoing_links": []},
    }
    fake = BadNamingLLM(pages, {"pages": []})
    install(fake)
    drafts = generator.generate_wiki("u1")
    assert len(drafts) == 2, drafts
    assert all(d.status == "ready" for d in drafts), [(d.title, d.error_msg) for d in drafts]
    assert {d.title for d in drafts} == {"知识页 1", "知识页 2"}


if __name__ == "__main__":
    print("== Wiki 生成内核测试 ==")
    results = [
        case("基本生成：聚类+命名+溯源+白名单+进度", test_basic_generation),
        case("非法引用丢弃，保留有效段", test_invalid_citation_dropped),
        case("全部引用非法 → 页面 failed", test_all_invalid_citations_failed),
        case("无 eligible 数据 → 空列表", test_empty_data),
        case("命名失败 → 退回通用标题不阻塞", test_naming_failure_fallback),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    sys.exit(0 if passed == len(results) else 1)

"""LLM Wiki 服务编排：异步生成任务、落库、建索引、检索、关系图、单页重写、防抖自动重生成。

依赖说明：本模块在 import 时不触发 torch（clear_chat_cache 采用惰性导入），
因此可在无 torch 的环境下被单元测试 mock 验证。
"""
import logging
import threading
from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import settings
from src.db.fts import tokenize, wiki_bm25_search
from src.db.repository import ChunkRepository, WikiJobRepository, WikiPageRepository
from src.llm import llm_client
from src.vectorstore import wiki_vector_store
from src.wiki.generator import WikiPageDraft, generate_single_page, generate_wiki

logger = logging.getLogger(__name__)

_REGEN_DEBOUNCE_S = 30  # 入库后自动重生成的防抖秒数
_regen_timers: Dict[str, threading.Timer] = {}


def _clear_chat_cache_safe() -> None:
    """清空问答缓存（best-effort）。src.rag 导入会拉起 torch，故惰性导入并吞异常，
    避免 Wiki 流程在无 torch 环境下整体失败。"""
    try:
        from src.rag import clear_chat_cache  # noqa: WPS433
        clear_chat_cache()
    except Exception as e:
        logger.warning(f"清空问答缓存失败（不影响 Wiki 生成）: {e}")


def _friendly_error(e: Exception) -> str:
    return (str(e) or e.__class__.__name__)[:500]


# ---------- 生成任务 ----------

def submit_wiki_generation(user_id: str) -> Dict[str, Any]:
    """提交生成任务。同用户已有进行中任务则复用（去重，避免重复消耗 token）。"""
    active = WikiJobRepository.get_active(user_id)
    if active:
        return {"job_id": str(active["id"]), "status": "generating", "reused": True}
    job_id = WikiJobRepository.create(user_id, page_total=0)
    return {"job_id": job_id, "status": "generating", "reused": False}


def get_latest_job(user_id: str) -> Optional[Dict[str, Any]]:
    return WikiJobRepository.get_latest(user_id)


def _page_point_id(user_id: str, slug: str) -> str:
    return f"{user_id}::{slug}"


def process_wiki_generation(job_id: str, user_id: str) -> None:
    """后台执行：生成 → 全量重建（清旧页/旧向量）→ 写 PG + 建向量索引 → 更新任务状态。"""
    try:
        def on_progress(done: int, total: int) -> None:
            WikiJobRepository.update_progress(job_id, done, page_total=total)

        drafts: List[WikiPageDraft] = generate_wiki(user_id, on_progress=on_progress)

        # 全量重建：先清旧页面与旧向量，保证 slug/向量不残留
        WikiPageRepository.delete_by_user(user_id)
        try:
            wiki_vector_store.delete_by_user(user_id)
        except Exception as e:
            logger.warning(f"清空旧 Wiki 向量失败（继续）: {e}")

        embed_texts: List[str] = []
        embed_slugs: List[str] = []
        for d in drafts:
            tokens = tokenize(f"{d.title} {d.summary} {d.content_md}")
            WikiPageRepository.upsert(
                user_id=user_id,
                slug=d.slug,
                title=d.title,
                summary=d.summary,
                content_md=d.content_md,
                paragraphs=d.paragraphs,
                outgoing_links=d.outgoing_links,
                source_chunk_ids=d.source_chunk_ids,
                tokens=tokens,
                status=d.status,
            )
            if d.status == "ready":
                embed_texts.append(f"{d.title}。{d.summary}")
                embed_slugs.append(d.slug)

        if embed_texts:
            try:
                vecs = llm_client.embed(embed_texts)
                ids = [_page_point_id(user_id, s) for s in embed_slugs]
                payloads = [{"user_id": user_id, "slug": s} for s in embed_slugs]
                wiki_vector_store.upsert(ids, vecs, payloads)
            except Exception as e:
                logger.warning(f"Wiki 页面向量索引失败（页面仍可浏览，仅影响问答召回）: {e}")

        ready_count = sum(1 for d in drafts if d.status == "ready")
        WikiJobRepository.mark_done(job_id, page_total=len(drafts))
        _clear_chat_cache_safe()
        logger.info(f"Wiki 生成完成：用户 {user_id}，共 {len(drafts)} 页（ready {ready_count}）")
    except Exception as e:
        logger.exception(f"Wiki 生成任务失败 job={job_id}")
        WikiJobRepository.mark_failed(job_id, _friendly_error(e))
        _clear_chat_cache_safe()


# ---------- 读取 / 检索 ----------

def list_pages(user_id: str) -> List[Dict[str, Any]]:
    return WikiPageRepository.list_by_user(user_id)


def get_page_detail(user_id: str, slug: str) -> Optional[Dict[str, Any]]:
    """单页详情 + 溯源来源（解析 source_chunk_ids 回查 chunk 内容）。"""
    page = WikiPageRepository.get_by_slug(user_id, slug)
    if not page:
        return None
    source_ids = page.get("source_chunk_ids") or []
    sources = ChunkRepository.get_sources_by_chunk_ids(user_id, [str(x) for x in source_ids])
    # 反向链接：其他页面的 outgoing_links 指向本页，用于实现双向跳转
    backlinks = [
        {"slug": r["slug"], "title": r["title"], "summary": r.get("summary")}
        for r in (WikiPageRepository.get_all_by_user(user_id) or [])
        if r["slug"] != slug and slug in (r.get("outgoing_links") or [])
    ]
    return {
        "slug": page["slug"],
        "title": page["title"],
        "summary": page.get("summary"),
        "content_md": page.get("content_md"),
        "paragraphs": page.get("paragraphs") or [],
        "outgoing_links": page.get("outgoing_links") or [],
        "backlinks": backlinks,
        "status": page.get("status", "ready"),
        "error_msg": page.get("error_msg"),
        "updated_at": page["updated_at"].isoformat() if page.get("updated_at") else None,
        "sources": [
            {
                "chunk_id": str(s["qdrant_point_id"]),
                "filename": s.get("filename"),
                "section": s.get("section"),
                "content": s.get("content"),
                "chunk_index": s.get("chunk_index"),
            }
            for s in sources
        ],
    }


def _rrf_slugs(vec_slugs: List[str], bm25_slugs: List[str], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, s in enumerate(vec_slugs, start=1):
        scores[s] = scores.get(s, 0.0) + 1.0 / (k + rank)
    for rank, s in enumerate(bm25_slugs, start=1):
        scores[s] = scores.get(s, 0.0) + 1.0 / (k + rank)
    return scores


def retrieve_wiki(query: str, user_id: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """召回相关 Wiki 页面（向量 + BM25 融合），供问答拼接上下文。返回 [{slug,title,summary,content_md,score}]。"""
    if not query or not query.strip():
        return []
    try:
        query_vec = llm_client.embed_query(query)
        filter_ = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        hits = wiki_vector_store.search(query_vec, top_k=max(top_k * 3, 5), filter_=filter_)
        vec_slugs = [c["payload"]["slug"] for c in hits if c.get("payload", {}).get("slug")]
    except Exception as e:
        logger.warning(f"Wiki 向量检索失败: {e}")
        vec_slugs = []

    try:
        bm25_rows = wiki_bm25_search(query, user_id, top_k=max(top_k * 3, 5))
        bm25_slugs = [r["slug"] for r in bm25_rows]
    except Exception as e:
        logger.warning(f"Wiki BM25 检索失败: {e}")
        bm25_slugs = []

    score_map = _rrf_slugs(vec_slugs, bm25_slugs)
    ranked = sorted(score_map.keys(), key=lambda s: score_map[s], reverse=True)[:top_k]
    if not ranked:
        return []
    results = []
    for slug in ranked:
        page = WikiPageRepository.get_by_slug(user_id, slug)
        if not page:
            continue
        results.append({
            "slug": page["slug"],
            "title": page["title"],
            "summary": page.get("summary"),
            "content_md": page.get("content_md"),
            "score": score_map.get(slug, 0.0),
        })
    return results


def search_wiki(query: str, user_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Wiki 站内搜索（给前端 /wiki/search 用）。"""
    pages = retrieve_wiki(query, user_id, top_k=top_k)
    return [{"slug": p["slug"], "title": p["title"], "summary": p["summary"], "score": p["score"]} for p in pages]


def build_graph(user_id: str) -> Dict[str, Any]:
    """关系图：节点 = 页面，边 = 页面间引用（自动补齐反向边，实现双向）。"""
    rows = WikiPageRepository.get_all_by_user(user_id)
    slug_set = {r["slug"] for r in rows}
    nodes = [{"slug": r["slug"], "title": r["title"], "status": r.get("status", "ready")} for r in rows]
    edges: List[Dict[str, str]] = []
    seen = set()
    for r in rows:
        for target in r.get("outgoing_links") or []:
            if target not in slug_set or target == r["slug"]:
                continue
            # 单向存储 → 补一条反向边，关系图双向可见
            for a, b in ((r["slug"], target), (target, r["slug"])):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                edges.append({"from": a, "to": b})
    return {"nodes": nodes, "edges": edges}


# ---------- 单页重生成 ----------

def regenerate_page(user_id: str, slug: str) -> Dict[str, Any]:
    """基于该页面现有来源 chunk 重写单页（不重新聚类）。成功则更新 PG + 向量。"""
    page = WikiPageRepository.get_by_slug(user_id, slug)
    if not page:
        raise ValueError("页面不存在")
    source_ids = [str(x) for x in (page.get("source_chunk_ids") or [])]
    chunks = ChunkRepository.get_sources_by_chunk_ids(user_id, source_ids)
    if not chunks:
        raise ValueError("该页面没有可用于重写的来源 chunk")

    all_titles = [r["title"] for r in WikiPageRepository.list_by_user(user_id)]
    draft = generate_single_page(
        title=page["title"],
        summary=page.get("summary") or "",
        source_chunks=chunks,
        all_page_titles=all_titles,
    )
    tokens = tokenize(f"{draft.title} {draft.summary} {draft.content_md}")
    WikiPageRepository.upsert(
        user_id=user_id,
        slug=draft.slug,
        title=draft.title,
        summary=draft.summary,
        content_md=draft.content_md,
        paragraphs=draft.paragraphs,
        outgoing_links=draft.outgoing_links,
        source_chunk_ids=draft.source_chunk_ids,
        tokens=tokens,
        status=draft.status,
    )
    # 刷新该页向量（先删后插，point_id 稳定）
    pid = _page_point_id(user_id, draft.slug)
    try:
        wiki_vector_store.delete_by_ids([pid])
        if draft.status == "ready":
            vec = llm_client.embed_query(f"{draft.title}。{draft.summary}")
            wiki_vector_store.upsert([pid], [vec], [{"user_id": user_id, "slug": draft.slug}])
    except Exception as e:
        logger.warning(f"重写页面《{draft.title}》向量刷新失败: {e}")
    _clear_chat_cache_safe()
    return {"slug": draft.slug, "status": draft.status, "error_msg": draft.error_msg}


# ---------- 入库后自动重生成（默认关闭，防抖） ----------

def schedule_wiki_regen(user_id: str) -> None:
    """入库完成后调用：若开启自动重生成，防抖延迟触发一次全量 Wiki 生成。"""
    if not settings.wiki_enabled or not settings.wiki_auto_regen:
        return
    old = _regen_timers.get(user_id)
    if old is not None:
        old.cancel()
    t = threading.Timer(_REGEN_DEBOUNCE_S, _do_auto_regen, args=(user_id,))
    t.daemon = True
    _regen_timers[user_id] = t
    t.start()


def _do_auto_regen(user_id: str) -> None:
    _regen_timers.pop(user_id, None)
    try:
        info = submit_wiki_generation(user_id)
        process_wiki_generation(info["job_id"], user_id)
    except Exception:
        logger.exception(f"自动 Wiki 重生成失败 user={user_id}")

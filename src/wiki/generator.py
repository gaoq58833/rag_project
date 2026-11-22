"""LLM Wiki 生成内核：聚类 → 批量命名 → 逐页生成 → 溯源校验 → 互链白名单。

只负责“读数据 + 调 LLM + 校验”，产出 WikiPageDraft 列表；写库/建索引/异步任务由 service 层处理，
因此本模块可被单元测试 mock（替换 ChunkRepository / vector_store / llm_client）独立验证。
"""
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# OpenBLAS 在内存/页面文件紧张的机器上多线程会分配失败（Observed: WinError 1455 类问题）。
# 在导入 numpy 前设默认单线程，仅当用户未显式设置时生效，避免聚类时崩溃。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from src.config import settings
from src.db.repository import ChunkRepository
from src.llm import llm_client
from src.vectorstore import vector_store
from src.wiki import prompts

logger = logging.getLogger(__name__)


@dataclass
class _Chunk:
    point_id: str
    content: str
    section: str
    filename: str
    document_id: str
    vector: List[float]


@dataclass
class WikiPageDraft:
    """生成器产出的 Wiki 页面草稿（未写库）。status: ready / failed。"""
    slug: str
    title: str
    summary: str
    content_md: str
    paragraphs: List[Dict] = field(default_factory=list)
    outgoing_links: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    status: str = "ready"
    error_msg: Optional[str] = None


# ---------- 数据加载 ----------

def _load_eligible_chunks(user_id: str) -> List[_Chunk]:
    """加载该用户 wiki_eligible 文档的 chunk，并取回向量。无向量的 chunk（如纯 BM25 JSON）跳过。"""
    rows = ChunkRepository.get_wiki_eligible_chunks(user_id)
    if not rows:
        return []
    point_ids = [str(r["qdrant_point_id"]) for r in rows]
    vec_map = vector_store.fetch_vectors(point_ids)
    chunks: List[_Chunk] = []
    for r in rows:
        pid = str(r["qdrant_point_id"])
        vec = vec_map.get(pid)
        if not vec:
            continue
        chunks.append(_Chunk(
            point_id=pid,
            content=r.get("content") or "",
            section=r.get("section") or "",
            filename=r.get("filename") or "",
            document_id=str(r["document_id"]),
            vector=list(vec),
        ))
    return chunks


# ---------- 聚类 ----------

def _cluster_indices(chunks: List[_Chunk], threshold: float, max_pages: int) -> List[List[int]]:
    """基于余弦相似度的连通分量聚类；超过 max_pages 时合并最相近的聚类。"""
    n = len(chunks)
    if n == 0:
        return []
    X = np.asarray([c.vector for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [g for g in groups.values()]

    # 超过页面上限：按质心相似度反复合并最相近的两个聚类
    while len(clusters) > max_pages and len(clusters) > 1:
        cents = []
        for cl in clusters:
            cv = Xn[cl].mean(axis=0)
            cv = cv / (np.linalg.norm(cv) + 1e-9)
            cents.append(cv)
        C = np.asarray(cents, dtype=np.float32)
        csim = C @ C.T
        np.fill_diagonal(csim, -1.0)
        a, b = np.unravel_index(int(np.argmax(csim)), csim.shape)
        clusters[a].extend(clusters[b])
        clusters.pop(b)

    return clusters


# ---------- 命名 ----------

def _build_previews(chunks: List[_Chunk], clusters: List[List[int]]) -> List[Dict]:
    previews = []
    for idx, cl in enumerate(clusters):
        members = [chunks[i] for i in cl]
        samples = []
        for c in members[:3]:
            samples.append({"section": c.section, "preview": (c.content or "")[:160]})
        previews.append({"index": idx, "size": len(members), "samples": samples})
    return previews


def _name_clusters(previews: List[Dict]) -> Dict[int, Dict]:
    """返回 {cluster_index: {title, summary}}。LLM 失败时退回通用标题，不阻塞整体生成。"""
    generic = {i: {"title": f"知识页 {i + 1}", "summary": ""} for i in range(len(previews))}
    if not previews:
        return generic
    try:
        data = llm_client.chat_json(
            prompts.naming_messages(previews),
            schema_hint=prompts.naming_schema_hint(),
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning(f"Wiki 命名 LLM 调用失败，退回通用标题: {e}")
        return generic

    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        return generic
    result: Dict[int, Dict] = {}
    for item in pages:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        title = (item.get("title") or "").strip()
        if not isinstance(idx, int) or not title:
            continue
        result[idx] = {"title": title, "summary": (item.get("summary") or "").strip()}
    # 补齐缺失的 index
    for i in range(len(previews)):
        result.setdefault(i, generic[i])
    return result


# ---------- 标题 / slug 去重 ----------

def _unique_titles(naming: Dict[int, Dict], count: int) -> List[str]:
    """按聚类顺序产出唯一标题列表；重名追加 (2)。"""
    titles: List[str] = []
    seen: Dict[str, int] = {}
    for i in range(count):
        base = (naming.get(i, {}).get("title") or f"知识页 {i + 1}").strip()
        t = base
        k = seen.get(base, 0)
        while t in seen:
            k += 1
            t = f"{base} ({k})"
        seen[base] = k
        seen[t] = 1
        titles.append(t)
    return titles


def _slugify(title: str, used: set) -> str:
    """中文标题直接做 slug（URL 层编码）；去重追加 -2、-3。"""
    slug = re.sub(r"\s+", "-", title.strip()) or "page"
    candidate = slug
    i = 2
    while candidate in used:
        candidate = f"{slug}-{i}"
        i += 1
    used.add(candidate)
    return candidate


# ---------- 单页生成 + 溯源校验 ----------

def _validate_paragraphs(paragraphs, allowed_ids: set) -> List[Dict]:
    """过滤：段落 text 非空，且 chunk_ids 只保留在 allowed_ids 内的；无有效引用的段落丢弃。"""
    cleaned: List[Dict] = []
    for p in paragraphs or []:
        if not isinstance(p, dict):
            continue
        text = (p.get("text") or "").strip()
        if not text:
            continue
        refs = [str(x) for x in (p.get("chunk_ids") or []) if str(x) in allowed_ids]
        if not refs:
            continue
        cleaned.append({"text": text, "chunk_ids": refs})
    return cleaned


def _generate_one(
    title: str,
    summary: str,
    members: List[_Chunk],
    title_to_slug: Dict[str, str],
    all_titles: List[str],
) -> WikiPageDraft:
    """生成单页（含 1 次校验重试）。单页失败返回 status=failed 的草稿，不抛异常。"""
    # 取聚类内的 chunk 作为上下文（聚类成员即最相关片段）
    sources = [
        {"id": c.point_id, "filename": c.filename, "section": c.section, "content": c.content}
        for c in members
    ][: settings.wiki_chunks_per_page]
    allowed_ids = {s["id"] for s in sources}
    link_whitelist = [t for t in all_titles if t != title]

    cleaned: List[Dict] = []
    outgoing: List[str] = []
    last_err: Optional[str] = None
    for attempt in range(2):
        try:
            data = llm_client.chat_json(
                prompts.page_messages(title, summary, sources, link_whitelist),
                schema_hint=prompts.page_schema_hint(),
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            last_err = f"LLM 调用失败: {e}"
            logger.warning(f"Wiki 页面《{title}》生成失败（第 {attempt + 1} 次）: {last_err}")
            continue
        cleaned = _validate_paragraphs(data.get("paragraphs") if isinstance(data, dict) else None, allowed_ids)
        if cleaned:
            raw_links = data.get("outgoing_links") if isinstance(data, dict) else None
            outgoing = [
                title_to_slug[t]
                for t in (raw_links or [])
                if t in title_to_slug and t != title
            ]
            break
        last_err = "所有段落均无有效来源引用"
        logger.warning(f"Wiki 页面《{title}》校验未通过（第 {attempt + 1} 次），重试")

    if not cleaned:
        return WikiPageDraft(
            slug=title_to_slug.get(title, _slugify(title, set())),
            title=title,
            summary=summary,
            content_md="",
            paragraphs=[],
            outgoing_links=[],
            source_chunk_ids=[],
            status="failed",
            error_msg=last_err or "生成失败",
        )

    content_md = "\n\n".join(p["text"] for p in cleaned)
    source_ids: List[str] = []
    for p in cleaned:
        for cid in p["chunk_ids"]:
            if cid not in source_ids:
                source_ids.append(cid)
    # 去重 outgoing_links 保序
    seen = set()
    outgoing_unique = [x for x in outgoing if not (x in seen or seen.add(x))]
    return WikiPageDraft(
        slug=title_to_slug[title],
        title=title,
        summary=summary,
        content_md=content_md,
        paragraphs=cleaned,
        outgoing_links=outgoing_unique,
        source_chunk_ids=source_ids,
        status="ready",
    )


# ---------- 单页重写（无需聚类/向量） ----------

def generate_single_page(
    title: str,
    summary: str,
    source_chunks: List[Dict],
    all_page_titles: List[str],
) -> WikiPageDraft:
    """基于已知来源 chunk 重写单页（单页重生成用）。
    source_chunks: [{point_id, content, section, filename}]；all_page_titles 为互链白名单。"""
    members = [
        _Chunk(
            point_id=str(c["point_id"]),
            content=c.get("content") or "",
            section=c.get("section") or "",
            filename=c.get("filename") or "",
            document_id=str(c.get("document_id") or ""),
            vector=[0.0],  # 单页重写不需要向量
        )
        for c in source_chunks
    ]
    used: set = set()
    title_to_slug: Dict[str, str] = {t: _slugify(t, used) for t in all_page_titles}
    return _generate_one(title, summary, members, title_to_slug, all_page_titles)


# ---------- 总入口 ----------

def generate_wiki(
    user_id: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[WikiPageDraft]:
    """为该用户生成 Wiki 页面草稿列表。不写库，由调用方（service/测试）决定如何落库。

    on_progress(done, total) 每完成一页回调一次，用于进度上报。
    """
    chunks = _load_eligible_chunks(user_id)
    if not chunks:
        logger.info(f"Wiki 生成：用户 {user_id} 无符合条件的 chunk")
        if on_progress:
            on_progress(0, 0)
        return []

    if len(chunks) > settings.wiki_max_chunks:
        chunks = chunks[: settings.wiki_max_chunks]
        logger.info(f"Wiki 生成：chunk 数超过上限，截取前 {settings.wiki_max_chunks} 个参与聚类")

    clusters = _cluster_indices(chunks, settings.wiki_cluster_sim_threshold, settings.wiki_max_pages)
    previews = _build_previews(chunks, clusters)
    naming = _name_clusters(previews)
    titles = _unique_titles(naming, len(clusters))

    used_slugs: set = set()
    title_to_slug: Dict[str, str] = {t: _slugify(t, used_slugs) for t in titles}

    drafts: List[WikiPageDraft] = []
    total = len(clusters)
    if total == 0:
        if on_progress:
            on_progress(0, 0)
        return []

    done = 0
    with ThreadPoolExecutor(max_workers=settings.wiki_concurrency) as ex:
        future_map = {}
        for idx, cl in enumerate(clusters):
            members = [chunks[i] for i in cl]
            fut = ex.submit(_generate_one, titles[idx], naming.get(idx, {}).get("summary", ""), members, title_to_slug, titles)
            future_map[fut] = idx
        for fut in as_completed(future_map):
            try:
                drafts.append(fut.result())
            except Exception as e:  # 理论上 _generate_one 已内部兜底，这里再保险一层
                idx = future_map[fut]
                logger.exception(f"Wiki 页面 {idx} 生成异常")
                drafts.append(WikiPageDraft(
                    slug=title_to_slug.get(titles[idx], f"page-{idx}"),
                    title=titles[idx],
                    summary=naming.get(idx, {}).get("summary", ""),
                    content_md="",
                    status="failed",
                    error_msg=f"生成异常: {e}",
                ))
            done += 1
            if on_progress:
                on_progress(done, total)

    # 按聚类顺序返回，便于前端稳定展示
    order = {titles[idx]: idx for idx in range(len(titles))}
    drafts.sort(key=lambda d: order.get(d.title, 9999))
    return drafts

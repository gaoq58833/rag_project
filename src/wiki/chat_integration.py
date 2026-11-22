"""Wiki → 问答上下文融合（torch-free，可独立单元测试）。

pipeline._build_chat_messages 委托本模块组装 messages。把「Wiki 融入问答」的逻辑
与 RAG 入库/检索主链路解耦，使该逻辑在无 torch 的环境下也能被单测覆盖。

设计要点（参考历史经验）：
- 资料与用户问题分离注入，Wiki 块在前、文档片段在后；
- 加硬约束（在 SYSTEM_PROMPT 侧）：仅依据给定资料作答；
- Wiki 检索失败时降级为仅用 chunk，不阻断问答，但 error 显式暴露给上层（不静默空上下文）；
- 响应带 wiki 可观测信号（used / count / error）。
"""
import logging
import time
from typing import Any, Callable, Dict, List, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

WIKI_TOP_K = 2  # 单次问答召回的 Wiki 页面数


def assemble_context(
    sources: List[dict], wiki_pages: List[dict], query: str
) -> Tuple[str, List[dict]]:
    """把 chunk 来源 + Wiki 页面拼成 LLM 上下文与前端来源列表（纯函数，无副作用）。

    返回 (context_msg, all_sources)：
      - context_msg：Wiki 块在前、文档片段块在后，与「用户问题」分离
      - all_sources：Wiki 页面项 type="wiki" 排在最前，其后是 chunk 来源
    """
    context_blocks: List[str] = []

    if wiki_pages:
        wiki_parts = []
        for i, p in enumerate(wiki_pages, 1):
            title = p.get("title") or p.get("slug")
            summary = p.get("summary") or ""
            body = p.get("content") or p.get("content_md") or ""
            wiki_parts.append(f"[W{i}] 《{title}》 摘要：{summary}\n内容：\n{body}")
        context_blocks.append("## 知识库结构化页面（Wiki）\n" + "\n\n".join(wiki_parts))

    if sources:
        chunk_parts = []
        for i, s in enumerate(sources, 1):
            section = f"[{s['section']}] " if s.get("section") else ""
            chunk_parts.append(f"[{i}] {section}{s['content']}")
        context_blocks.append("## 文档片段\n" + "\n\n".join(chunk_parts))

    context_msg = (
        "\n\n".join(context_blocks) + f"\n\n用户问题：{query}"
    ) if context_blocks else query

    all_sources: List[dict] = []
    for p in wiki_pages:
        all_sources.append({
            "type": "wiki",
            "slug": p.get("slug"),
            "title": p.get("title"),
            "summary": p.get("summary"),
            "content": p.get("content") or p.get("content_md") or "",
            "score": p.get("score"),
        })
    all_sources.extend(sources)

    return context_msg, all_sources


def build_chat_context(
    query: str,
    user_id: str,
    history: List[dict],
    use_rerank: bool,
    timings: Dict[str, Any],
    retrieve_chunks: Callable[..., List[dict]],
    system_prompt: str,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    """检索（chunk + 可选 Wiki）并组装发给 LLM 的 messages。

    参数 retrieve_chunks 由 pipeline 注入其 retrieve（避免本模块反向依赖 pipeline / torch）。
    返回 (messages, all_sources, wiki_info)：
      - messages: system + history + user(上下文 + 问题)
      - all_sources: chunk 来源 + Wiki 页面来源（Wiki 项带 type="wiki"）
      - wiki_info: {"used": bool, "count": int, "error": Optional[str]}
    """
    sources = retrieve_chunks(query, user_id=user_id, use_rerank=use_rerank, timings=timings)

    wiki_pages: List[dict] = []
    wiki_info: Dict[str, Any] = {"used": False, "count": 0, "error": None}
    if settings.wiki_in_chat:
        t_wiki = time.perf_counter()
        try:
            # 惰性导入：service 不依赖本模块/pipeline，避免循环；也避免启动期拉起 torch
            from src.wiki.service import retrieve_wiki
            wiki_pages = retrieve_wiki(query, user_id, top_k=WIKI_TOP_K)
            wiki_info["used"] = len(wiki_pages) > 0
            wiki_info["count"] = len(wiki_pages)
        except Exception as e:
            # 失败显式化：降级为仅用 chunk，但 error 暴露给上层决定如何提示用户
            wiki_info["error"] = f"{type(e).__name__}: {e}"
            logger.warning(f"Wiki 检索失败，降级为仅使用文档片段: {e}")
        finally:
            timings["wiki_s"] = round(time.perf_counter() - t_wiki, 3)

    context_msg, all_sources = assemble_context(sources, wiki_pages, query)

    messages: List[dict] = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": context_msg})

    return messages, all_sources, wiki_info

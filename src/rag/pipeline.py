"""RAG 管线：入库（解析→分块→向量化→存 PG+Qdrant）、检索（向量召回→重排→回查 PG）、问答"""
import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from src.config import settings
from src.db.fts import bm25_search, rrf_fuse, tokenize
from src.db.repository import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
)
from src.llm import llm_client
from src.parsers.chunker import ChunkStrategy, TextChunker
from src.parsers.factory import get_parser
from src.vectorstore import vector_store
from qdrant_client.models import FieldCondition, Filter, MatchValue


DEFAULT_CHUNKER = TextChunker(
    strategy=ChunkStrategy.PARAGRAPH,
    max_chunk_size=settings.chunk_max_size,
    min_chunk_size=settings.chunk_min_size,
    chunk_overlap=settings.chunk_overlap,
)

# 系统提示词
SYSTEM_PROMPT = """你是一个基于知识库的问答助手，负责根据检索到的文档片段回答用户问题。

## 核心原则
1. 严格基于上下文回答，不得编造、推测或引入上下文之外的信息。
2. 若上下文不足以回答问题，明确回复："根据现有知识库无法回答该问题。"
3. 回答语言与用户提问语言保持一致。

## 上下文使用
- 下方会提供带编号的文档片段，格式为 [编号] [章节] 内容。
- 回答时引用对应片段，用 [1]、[2] 标注来源。
- 多个片段相关时，综合整理后回答，不要逐片段罗列。

## 回答风格
- 先给结论，再展开说明，避免空话套话。
- 技术性问题可使用分点、代码块、表格等结构化表达。
- 不输出"根据上下文"、"检索到的内容显示"等元话语，直接回答。

## 多轮对话
- 结合对话历史理解用户的指代（如"它"、"上面那个"）。
- 若用户追问，基于历史上下文和新检索结果继续回答。

## 回答示例

### 示例 1：上下文足够
用户：RNN 为什么能处理序列数据？
上下文：
[1] [3.1 为什么不能把文本直接丢进全连接层] 要保留顺序，就得让网络一个词一个词地读，并且把"读到第几个词"这个状态带到下一步——这就是循环神经网络。
[2] [3.1] 循环神经网络的隐藏状态会传递到下一个时间步，从而保留序列的顺序信息。

✅ 正确回答：
RNN 通过隐藏状态在时间步之间传递来处理序列。它逐个读取输入词，并将当前步的状态传递到下一步，从而保留序列的顺序信息 [1][2]。

❌ 错误回答：
根据检索到的内容，RNN 是一种神经网络。它有隐藏层，可以处理序列。（没有引用来源，且混入了上下文外的说法）

### 示例 2：上下文不足
用户：什么是 Transformer 的多头注意力机制？
上下文：
[1] [3.2 词向量] 词向量将词映射为稠密向量，语义相近的词向量也相近。

✅ 正确回答：
根据现有知识库无法回答该问题。

❌ 错误回答：
多头注意力机制是 Transformer 的核心，它通过多个注意力头并行关注不同子空间...（编造了上下文外的内容）
"""


def submit_ingest(file_bytes: bytes, filename: str, user_id: Optional[str] = None) -> dict:
    """入库快速通道（同步，毫秒级返回）：

    1. 内容哈希查重：同用户传过相同文件 → 直接复用向量，同步完成（不调 embedding，很快）
    2. 否则创建 status=processing 的文档记录，立即返回 document_id，
       解析/分块/向量化交给 process_ingest 后台执行

    返回：{document_id, filename, status, reused, chunk_count?, stored_count?}
    """
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = DocumentRepository.get_by_content_hash(content_hash, user_id=user_id)
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if existing is not None:
        # 仅同一用户的相同内容才复用向量（跨用户不复用，保证 Qdrant user_id 过滤隔离）
        existing_id = existing["id"]
        existing_chunks = ChunkRepository.get_by_document_id(existing_id)

        # 已有文档 chunks 为空（入库失败残留或正在处理中），不复用，走完整入库
        if not existing_chunks:
            logger.warning(f"Document {existing_id} has 0 chunks (processing or failed), skipping reuse")
        else:
            document_id = DocumentRepository.create(
                filename=filename,
                file_ext=file_ext,
                total_pages=existing["total_pages"],
                table_count=existing["table_count"],
                chunk_count=len(existing_chunks),
                raw_text=existing.get("raw_text"),
                content_hash=content_hash,
                user_id=user_id,
                status="completed",
            )

            reused_chunks = []
            for c in existing_chunks:
                reused_chunks.append({
                    "document_id": document_id,
                    "qdrant_point_id": str(c["qdrant_point_id"]),
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "length": c["length"],
                    "chunk_type": c.get("chunk_type", "text"),
                    "section": c.get("section", ""),
                    "images": c.get("images", []),
                    "tokens": tokenize(c["content"]),
                })
            ChunkRepository.bulk_insert(reused_chunks)
            clear_chat_cache()  # 新文档立即可检索，清空问答缓存

            return {
                "document_id": document_id,
                "filename": filename,
                "status": "completed",
                "chunk_count": len(existing_chunks),
                "stored_count": len(existing_chunks),
                "reused": True,
            }

    # 先落一条 processing 记录，立即返回；解析/向量化在后台进行
    document_id = DocumentRepository.create(
        filename=filename,
        file_ext=file_ext,
        total_pages=0,
        table_count=0,
        chunk_count=0,
        raw_text=None,
        content_hash=content_hash,
        user_id=user_id,
        status="processing",
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "status": "processing",
        "reused": False,
    }


def process_ingest(document_id: str, file_bytes: bytes, filename: str, user_id: Optional[str] = None) -> None:
    """异步入库（后台执行）：解析 → 分块 → 向量化 → 写 PG chunks + Qdrant → 更新状态

    向量只在全部处理完后才写入，所以检索天然不会搜到 processing 的半成品文档。
    失败时把文档置为 failed 并记录 error_msg，文档记录保留供用户看到失败原因。

    数据分工：
      - PostgreSQL：完整 chunk 数据（content/section/type/images 等）
      - Qdrant：向量 + 最小 payload（用于检索后回查 PG）
    """
    try:
        # 文档可能在处理期间被用户删除：开始前检查一次
        if DocumentRepository.get_by_id(document_id) is None:
            logger.warning(f"Document {document_id} deleted before processing, abort")
            return

        # 1. 解析
        parser = get_parser(filename)
        result = parser.parse_from_bytes(file_bytes, filename)

        # 2. 分块
        chunks = DEFAULT_CHUNKER.split(result.raw_text)

        # 3. 向量化 content（只向量化正文）
        contents = [c["content"] for c in chunks]
        embeddings = llm_client.embed(contents)

        # 4. 构造 PG chunks 记录 + Qdrant points
        pg_chunks = []
        qdrant_ids = []
        qdrant_payloads = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            qdrant_ids.append(point_id)
            pg_chunks.append({
                "document_id": document_id,
                "qdrant_point_id": point_id,
                "chunk_index": chunk["index"],
                "content": chunk["content"],
                "length": chunk["length"],
                "chunk_type": chunk.get("type", "text"),
                "section": chunk.get("section", ""),
                "images": chunk.get("images", []),
                "tokens": tokenize(chunk["content"]),
            })
            # Qdrant payload 只存最小信息，完整数据回 PG 查
            qdrant_payloads.append({
                "qdrant_point_id": point_id,
                "filename": filename,
                "section": chunk.get("section", ""),
                "user_id": user_id,
            })

        # 5. 写 PG chunks + Qdrant 向量（先 PG 后 Qdrant：PG 失败则 Qdrant 不会留孤儿向量）
        ChunkRepository.bulk_insert(pg_chunks)
        vector_store.upsert(qdrant_ids, embeddings, qdrant_payloads)

        # 6. 回填解析结果，置为 completed
        DocumentRepository.mark_completed(
            document_id,
            total_pages=result.total_pages,
            table_count=result.table_count,
            chunk_count=len(chunks),
            raw_text=result.raw_text,
        )
        clear_chat_cache()  # 新文档可检索，清空问答缓存
        logger.info(f"Document {document_id} ({filename}) ingested: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Ingest failed for document {document_id} ({filename}): {e}")
        DocumentRepository.mark_failed(document_id, str(e))


def retrieve(query: str, user_id: str, top_k: Optional[int] = None, use_rerank: bool = True,
             use_bm25: bool = True) -> List[dict]:
    """混合检索：向量召回 + BM25 关键词召回 → RRF 融合 → 重排 → 回 PostgreSQL 取完整 chunk

    - 向量路：语义相近（Qdrant，按 user_id 过滤）
    - BM25 路：字面精确（PG tsvector，JOIN documents 按 user_id 过滤）
      两路互补：编号/专名/错误码靠 BM25，同义改写靠向量
    - use_bm25=False 时走纯向量模式（跳过 BM25 与 RRF），用于检索策略 A/B 对比评测
    返回：[{content, section, type, images, score, filename, index}, ...]
    """
    top_k = top_k or settings.retrieval_top_k

    # 1. 向量召回（按 user_id 过滤，只召回当前用户的文档）
    query_vec = llm_client.embed_query(query)
    filter_ = Filter(must=[
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ])
    vec_candidates = vector_store.search(query_vec, top_k=top_k, filter_=filter_)
    vec_ids = [c["payload"]["qdrant_point_id"] for c in vec_candidates]
    # 向量路 payload 里的 filename，供结果组装
    filename_map = {c["payload"]["qdrant_point_id"]: c["payload"].get("filename", "") for c in vec_candidates}

    # 2. 召回结果融合：use_bm25=True 走 BM25+RRF 混合，False 走纯向量（A/B 对比用）
    if use_bm25:
        # BM25 关键词召回（同 user_id 隔离）
        bm25_rows = bm25_search(query, user_id, top_k=settings.bm25_top_k)
        for row in bm25_rows:
            filename_map.setdefault(str(row["qdrant_point_id"]), row.get("filename", ""))
        # RRF 融合两路排名，得到统一候选顺序
        score_map = rrf_fuse(vec_ids, bm25_rows)
        merged_ids = sorted(score_map.keys(), key=lambda p: score_map[p], reverse=True)
    else:
        # 纯向量模式：直接按向量相似度排序，分数为余弦相似度
        merged_ids = list(vec_ids)
        score_map = {c["payload"]["qdrant_point_id"]: c.get("score", 0.0) for c in vec_candidates}
    if not merged_ids:
        return []

    # 4. 回 PG 取完整数据
    pg_rows = ChunkRepository.get_by_qdrant_ids(merged_ids)
    pg_map = {str(r["qdrant_point_id"]): r for r in pg_rows}
    merged_ids = [pid for pid in merged_ids if pid in pg_map]
    if not merged_ids:
        return []

    # 5. 重排（用 content 文本重排，从融合候选中取 top_n）
    if use_rerank and len(merged_ids) > 1:
        docs = [pg_map[pid]["content"] for pid in merged_ids]
        rerank_indices = llm_client.rerank(query, docs, top_n=settings.rerank_top_n)
        merged_ids = [merged_ids[i] for i in rerank_indices]
    else:
        merged_ids = merged_ids[:settings.rerank_top_n]

    results = []
    for pid in merged_ids:
        row = pg_map[pid]
        results.append({
            "content": row["content"],
            "section": row.get("section", ""),
            "type": row.get("chunk_type", "text"),
            "images": row.get("images", []),
            "score": score_map.get(pid, 0.0),  # 混合模式为 RRF 融合分，纯向量模式为余弦相似度
            "filename": filename_map.get(pid, ""),
            "index": row.get("chunk_index", 0),
        })
    return results


# 问答缓存：同用户同问题（含相同历史上下文）10 分钟内直接返回，跳过检索+LLM
class _TTLCache:
    """简易 TTL 缓存（纯标准库实现，线程安全，LRU 淘汰）"""

    def __init__(self, maxsize: int = 500, ttl: int = 600):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            value, expire_at = item
            if time.time() > expire_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (value, time.time() + self.ttl)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


_chat_cache = _TTLCache(maxsize=settings.chat_cache_maxsize, ttl=settings.chat_cache_ttl)


def clear_chat_cache():
    """文档增删后清空问答缓存，保证答案新鲜度"""
    _chat_cache.clear()


def _chat_cache_key(query: str, user_id: str, history: List[dict], use_rerank: bool = True) -> str:
    """缓存键：用户 + 问题 + 最近历史摘要（多轮上下文不同则不命中）"""
    history_sig = "|".join(f"{m['role']}:{m['content'][:settings.chat_cache_history_sig_len]}" for m in history)
    return hashlib.sha256(f"{user_id}|{query}|{use_rerank}|{history_sig}".encode()).hexdigest()


def _chat_generate(query: str, user_id: str, history: List[dict], use_rerank: bool = True) -> Dict[str, Any]:
    """检索 + LLM 生成（带缓存），不做落库"""
    cache_key = _chat_cache_key(query, user_id, history, use_rerank)
    cached = _chat_cache.get(cache_key)
    if cached is not None:
        return cached

    sources = retrieve(query, user_id=user_id, use_rerank=use_rerank)

    if not sources:
        answer = "知识库中没有相关内容，请尝试其他问题。"
        context_msg = query
    else:
        context_parts = []
        for i, s in enumerate(sources, 1):
            section = f"[{s['section']}] " if s.get("section") else ""
            context_parts.append(f"[{i}] {section}{s['content']}")
        context = "\n\n".join(context_parts)
        context_msg = f"上下文：\n{context}\n\n用户问题：{query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": context_msg})

    answer = llm_client.chat(messages)
    result = {"answer": answer, "sources": sources}
    _chat_cache.set(cache_key, result)
    return result


def chat(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    use_rerank: bool = True,
) -> dict:
    """问答：检索 → LLM 生成（可缓存）→ 落库

    - 不传 conversation_id 时新建对话（标题取 query 前 30 字）
    - 取最近 chat_history_rounds 轮历史作为多轮上下文
    - 同用户同问题+同历史 10 分钟内命中缓存，跳过检索+LLM，但仍记录消息
    - 返回：{answer, sources, conversation_id}
    """
    # 1. 已有对话：取历史
    history = []
    if conversation_id:
        conv = ConversationRepository.get_by_id(conversation_id)
        if conv is None or str(conv["user_id"]) != user_id:
            raise ValueError("对话不存在或无权访问")
        history = MessageRepository.get_recent(conversation_id, settings.chat_history_rounds)

    # 2. 生成回答（命中缓存则跳过检索+LLM）
    result = _chat_generate(query, user_id, history, use_rerank=use_rerank)
    answer = result["answer"]
    sources = result["sources"]

    # 3. 新对话此时才创建
    if not conversation_id:
        conversation_id = ConversationRepository.create(user_id, title=query[:30])

    # 4. 落库：user 消息 + assistant 消息（缓存命中也记录）
    MessageRepository.create(conversation_id, role="user", content=query)
    MessageRepository.create(
        conversation_id,
        role="assistant",
        content=answer,
        sources=sources,
    )
    ConversationRepository.touch(conversation_id)

    return {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
    }


def chat_stream(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    use_rerank: bool = True,
):
    """流式问答：逐 token yield 内容，结束后落库

    生成器 yield 的内容：
      - {"type": "start", "conversation_id": str, "sources": list}
      - {"type": "token", "content": str}  (多次)
      - {"type": "done"}
    """
    # 1. 已有对话：取历史
    history = []
    if conversation_id:
        conv = ConversationRepository.get_by_id(conversation_id)
        if conv is None or str(conv["user_id"]) != user_id:
            raise ValueError("对话不存在或无权访问")
        history = MessageRepository.get_recent(conversation_id, settings.chat_history_rounds)

    # 2. 新对话先创建（流式需要提前拿到 conversation_id）
    if not conversation_id:
        conversation_id = ConversationRepository.create(user_id, title=query[:30])

    # 3. 查缓存
    cache_key = _chat_cache_key(query, user_id, history, use_rerank)
    cached = _chat_cache.get(cache_key)

    if cached is not None:
        # 缓存命中：把答案拆成小块"流式"吐出，保证 UX 一致
        answer = cached["answer"]
        sources = cached["sources"]
        yield {"type": "start", "conversation_id": conversation_id, "sources": sources}
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            yield {"type": "token", "content": answer[i:i + chunk_size]}
    else:
        # 缓存未命中：检索 + 流式生成
        sources = retrieve(query, user_id=user_id, use_rerank=use_rerank)
        if not sources:
            answer = "知识库中没有相关内容，请尝试其他问题。"
            context_msg = query
        else:
            context_parts = []
            for i, s in enumerate(sources, 1):
                section = f"[{s['section']}] " if s.get("section") else ""
                context_parts.append(f"[{i}] {section}{s['content']}")
            context = "\n\n".join(context_parts)
            context_msg = f"上下文：\n{context}\n\n用户问题：{query}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": context_msg})

        yield {"type": "start", "conversation_id": conversation_id, "sources": sources}

        answer_parts = []
        for token in llm_client.chat_stream(messages):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
        answer = "".join(answer_parts)
        _chat_cache.set(cache_key, {"answer": answer, "sources": sources})

    # 4. 落库
    MessageRepository.create(conversation_id, role="user", content=query)
    MessageRepository.create(
        conversation_id,
        role="assistant",
        content=answer,
        sources=sources,
    )
    ConversationRepository.touch(conversation_id)

    yield {"type": "done"}

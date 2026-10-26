"""RAG 管线：入库（解析→分块→向量化→存 PG+Qdrant）、检索（向量召回→重排→回查 PG）、问答"""
import uuid
from typing import List, Optional

from src.config import settings
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


DEFAULT_CHUNKER = TextChunker(
    strategy=ChunkStrategy.PARAGRAPH,
    max_chunk_size=1000,
    min_chunk_size=100,
    chunk_overlap=200,
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


def ingest_file(file_bytes: bytes, filename: str) -> dict:
    """入库：解析 → 分块 → 向量化 → 同时写入 PostgreSQL 和 Qdrant

    数据分工：
      - PostgreSQL：完整 chunk 数据（content/section/type/images 等）
      - Qdrant：向量 + 最小 payload（用于检索后回查 PG）

    返回：{filename, chunk_count, stored_count}
    """
    # 1. 解析
    parser = get_parser(filename)
    result = parser.parse_from_bytes(file_bytes, filename)

    # 2. 分块
    chunks = DEFAULT_CHUNKER.split(result.raw_text, tables=result.tables)

    # 3. 先删同名旧文档（PG 级联删 chunks，Qdrant 按 filename 删），避免外键冲突
    DocumentRepository.delete_by_filename(filename)
    vector_store.delete_by_filename(filename)

    # 4. 写 PG documents 表，拿到 document_id
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    document_id = DocumentRepository.create(
        filename=filename,
        file_ext=file_ext,
        total_pages=result.total_pages,
        table_count=result.table_count,
        chunk_count=len(chunks),
        raw_text=result.raw_text,
    )

    # 5. 向量化 content（只向量化正文）
    contents = [c["content"] for c in chunks]
    embeddings = llm_client.embed(contents)

    # 6. 构造 PG chunks 记录 + Qdrant points
    pg_chunks = []
    qdrant_ids = []
    qdrant_payloads = []
    for i, chunk in enumerate(chunks):
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
        })
        # Qdrant payload 只存最小信息，完整数据回 PG 查
        qdrant_payloads.append({
            "qdrant_point_id": point_id,
            "filename": filename,
            "section": chunk.get("section", ""),
        })

    # 7. 写 PG chunks + Qdrant 向量
    ChunkRepository.bulk_insert(pg_chunks)
    vector_store.upsert(qdrant_ids, embeddings, qdrant_payloads)

    return {
        "filename": filename,
        "chunk_count": len(chunks),
        "stored_count": len(qdrant_ids),
    }


def retrieve(query: str, top_k: Optional[int] = None, use_rerank: bool = True) -> List[dict]:
    """检索：Qdrant 向量召回 → 重排 → 回 PostgreSQL 取完整 chunk

    返回：[{content, section, type, images, score, filename, index}, ...]
    """
    top_k = top_k or settings.retrieval_top_k

    # 1. 向量召回（拿到 qdrant_point_id + score）
    query_vec = llm_client.embed_query(query)
    candidates = vector_store.search(query_vec, top_k=top_k)

    if not candidates:
        return []

    # 2. 重排（用 content 文本重排）
    if use_rerank and len(candidates) > 1:
        # 先回 PG 拿到完整 content 用于重排
        qdrant_ids = [c["payload"]["qdrant_point_id"] for c in candidates]
        pg_rows = ChunkRepository.get_by_qdrant_ids(qdrant_ids)
        pg_map = {r["qdrant_point_id"]: r for r in pg_rows}

        docs = [pg_map.get(cid, {}).get("content", "") for cid in qdrant_ids]
        rerank_indices = llm_client.rerank(query, docs, top_n=settings.rerank_top_n)
        candidates = [candidates[i] for i in rerank_indices]

    # 3. 回 PG 取完整数据，保持 Qdrant 召回顺序
    qdrant_ids = [c["payload"]["qdrant_point_id"] for c in candidates]
    pg_rows = ChunkRepository.get_by_qdrant_ids(qdrant_ids)
    pg_map = {str(r["qdrant_point_id"]): r for r in pg_rows}

    results = []
    for c in candidates:
        pid = c["payload"]["qdrant_point_id"]
        row = pg_map.get(pid)
        if not row:
            continue
        results.append({
            "content": row["content"],
            "section": row.get("section", ""),
            "type": row.get("chunk_type", "text"),
            "images": row.get("images", []),
            "score": c["score"],
            "filename": c["payload"].get("filename", ""),
            "index": row.get("chunk_index", 0),
        })
    return results


def chat(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    use_rerank: bool = True,
) -> dict:
    """问答：检索 → 构造上下文（含多轮历史）→ LLM 生成 → 落库

    - 不传 conversation_id 时新建对话（标题取 query 前 30 字）
    - 取最近 chat_history_rounds 轮历史作为多轮上下文
    - user / assistant 消息都会写入 messages 表

    返回：{answer, sources, conversation_id}
    """
    # 1. 已有对话：取历史；新对话：延迟到检索成功后再创建（避免留下空对话）
    history = []
    if conversation_id:
        conv = ConversationRepository.get_by_id(conversation_id)
        if conv is None or str(conv["user_id"]) != user_id:
            raise ValueError("对话不存在或无权访问")
        history = MessageRepository.get_recent(conversation_id, settings.chat_history_rounds)

    # 2. 检索相关片段
    sources = retrieve(query, use_rerank=use_rerank)

    if not sources:
        answer = "知识库中没有相关内容，请尝试其他问题。"
        context_msg = query  # 无来源时直接用原问题
    else:
        # 构造当前问题的上下文
        context_parts = []
        for i, s in enumerate(sources, 1):
            section = f"[{s['section']}] " if s.get("section") else ""
            context_parts.append(f"[{i}] {section}{s['content']}")
        context = "\n\n".join(context_parts)
        context_msg = f"上下文：\n{context}\n\n用户问题：{query}"

    # 3. 拼接多轮历史 + 当前问题
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": context_msg})

    answer = llm_client.chat(messages)

    # 4. 新对话此时才创建
    if not conversation_id:
        conversation_id = ConversationRepository.create(user_id, title=query[:30])

    # 5. 落库：user 消息 + assistant 消息
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

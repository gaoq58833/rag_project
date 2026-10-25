"""RAG 管线：入库（解析→分块→向量化→存 PG+Qdrant）、检索（向量召回→重排→回查 PG）、问答"""
import uuid
from typing import List, Optional

from src.config import settings
from src.db.repository import ChunkRepository, DocumentRepository
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
SYSTEM_PROMPT = """你是一个基于知识库的问答助手。请根据下方检索到的上下文内容回答用户问题。

要求：
1. 只基于上下文回答，不要编造上下文之外的信息
2. 如果上下文不足以回答问题，明确说"根据现有文档无法回答该问题"
3. 回答简洁准确，引用相关原文"""


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

    # 3. 先写 PG documents 表，拿到 document_id
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    document_id = DocumentRepository.create(
        filename=filename,
        file_ext=file_ext,
        total_pages=result.total_pages,
        table_count=result.table_count,
        chunk_count=len(chunks),
        raw_text=result.raw_text,
    )

    # 4. 向量化 content（只向量化正文）
    contents = [c["content"] for c in chunks]
    embeddings = llm_client.embed(contents)

    # 5. 构造 PG chunks 记录 + Qdrant points
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

    # 6. 同名文件先删旧的（PG 级联删 chunks，Qdrant 按 filename 删）
    DocumentRepository.delete_by_filename(filename)
    vector_store.delete_by_filename(filename)

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


def chat(query: str, use_rerank: bool = True) -> dict:
    """问答：检索 → 构造上下文 → LLM 生成

    返回：{answer, sources}
    """
    sources = retrieve(query, use_rerank=use_rerank)

    if not sources:
        return {
            "answer": "知识库中没有相关内容，请尝试其他问题。",
            "sources": [],
        }

    # 构造上下文
    context_parts = []
    for i, s in enumerate(sources, 1):
        section = f"[{s['section']}] " if s.get("section") else ""
        context_parts.append(f"[{i}] {section}{s['content']}")
    context = "\n\n".join(context_parts)

    user_msg = f"上下文：\n{context}\n\n用户问题：{query}"

    answer = llm_client.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])

    return {
        "answer": answer,
        "sources": sources,
    }

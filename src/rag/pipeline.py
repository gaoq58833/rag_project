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
from src.parsers.factory import get_parser, index_type_for
from src.parsers.image_parser import ImageParser
from src.rag.table_expand import merge_table_chunks
from src.vectorstore import image_vector_store, vector_store
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

## 知识库结构化页面（Wiki，若提供）
- 有时会额外提供由知识库综合生成的结构化知识页，格式为 [W编号] 《页面标题》 摘要 / 内容。
- 若提供了 Wiki 页面，优先综合其内容作答；引用时用 [[页面标题]] 标注（如 [[循环神经网络]]）。
- 文档片段仍可补充细节，用 [1]、[2] 标注。
- Wiki 页面与文档片段同属"给定资料"，一律不得引入资料之外的信息。

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
    file_size = len(file_bytes)
    # 默认不参与 Wiki 生成；用户在前端文档列表手动开启后才纳入（与文件大小无关）
    wiki_eligible = False

    if existing is not None:
        # 仅同一用户的相同内容才复用（跨用户不复用，保证 Qdrant user_id 过滤隔离）
        # 索引类型不同也不复用：JSON 只建倒排索引，普通文档有向量，不能共享同一批 chunk
        existing_id = existing["id"]
        same_index = index_type_for(existing["filename"]) == index_type_for(filename)
        existing_chunks = ChunkRepository.get_by_document_id(existing_id) if same_index else []

        if not same_index:
            logger.info(f"Document {existing_id} uses different index type, skipping reuse")
        # 已有文档 chunks 为空（入库失败残留或正在处理中），不复用，走完整入库
        elif not existing_chunks:
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
                file_size=file_size,
                wiki_eligible=wiki_eligible,
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
                    "table_index": c.get("table_index"),
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
        file_size=file_size,
        wiki_eligible=wiki_eligible,
    )
    return {
        "document_id": document_id,
        "filename": filename,
        "status": "processing",
        "reused": False,
    }


def _friendly_ingest_error(exc: Exception) -> str:
    """把技术异常翻译为用户能看懂的提示（原始异常仍会打印在后台日志中）"""
    msg = str(exc)
    low = msg.lower()
    # 服务连接/超时类：稍后重试可能成功
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        k in low for k in ("timeout", "timed out", "connection", "connect")
    ) or "连接" in msg:
        return "模型服务连接失败，请稍后点击「更新」重试"
    # 文件格式不支持
    if "不支持的文件格式" in msg or "不支持的图片格式" in msg:
        return "您的文档格式不符合要求，请重新上传"
    # 文件过大
    if "过大" in msg:
        return "文件过大，请压缩后重新上传"
    # 内容为空
    if "全部为空" in msg or "empty" in low:
        return "文档内容为空，请重新上传"
    # 解析失败（文件损坏、编码异常、JSON 语法错误等）
    if "解析失败" in msg or "编码识别失败" in msg or "无法识别图片" in msg or "嵌套层级过深" in msg:
        return "文档无法解析（可能已损坏或编码异常），请重新上传"
    # 兜底
    return "文档处理失败，请点击「更新」重新上传"


def process_ingest(document_id: str, file_bytes: bytes, filename: str, user_id: Optional[str] = None) -> None:
    """异步入库（后台执行）：解析 → 分块 → 向量化/倒排索引 → 写 PG + Qdrant → 更新状态

    JSON（INDEX_TYPE=bm25）只建 PG 倒排索引，跳过 embedding 与 Qdrant；
    其他格式仍走原有的向量 + 倒排混合索引。
    向量只在全部处理完后才写入，所以检索天然不会搜到 processing 的半成品文档。
    失败时把文档置为 failed 并记录 error_msg，文档记录保留供用户看到失败原因。

    数据分工：
      - PostgreSQL：完整 chunk 数据（content/section/type/images 等）+ tsv 倒排索引
      - Qdrant：向量 + 最小 payload（用于检索后回查 PG）；仅 INDEX_TYPE=vector 的格式写入
    """
    try:
        # 文档可能在处理期间被用户删除：开始前检查一次
        if DocumentRepository.get_by_id(document_id) is None:
            logger.warning(f"Document {document_id} deleted before processing, abort")
            return

        # 1. 解析
        parser = get_parser(filename)
        result = parser.parse_from_bytes(file_bytes, filename)

        # 2. 分块：解析器自带 chunks（JSON 按记录预分块）优先，否则走通用 chunker
        chunks = result.chunks or DEFAULT_CHUNKER.split(result.raw_text)

        # 3. 向量化：JSON 只建倒排索引（PG tsv），跳过 embedding 和 Qdrant
        use_vector = parser.INDEX_TYPE != "bm25"
        embeddings = llm_client.embed([c["content"] for c in chunks]) if use_vector else None

        # 4. 构造 PG chunks 记录 + Qdrant points
        pg_chunks = []
        qdrant_ids = []
        qdrant_payloads = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            pg_chunks.append({
                "document_id": document_id,
                "qdrant_point_id": point_id,
                "chunk_index": chunk["index"],
                "content": chunk["content"],
                "length": chunk["length"],
                "chunk_type": chunk.get("type", "text"),
                "section": chunk.get("section", ""),
                "table_index": chunk.get("table_index"),
                "images": chunk.get("images", []),
                "tokens": tokenize(chunk["content"]),
            })
            if use_vector:
                # Qdrant payload 只存最小信息，完整数据回 PG 查
                qdrant_ids.append(point_id)
                qdrant_payloads.append({
                    "qdrant_point_id": point_id,
                    "filename": filename,
                    "section": chunk.get("section", ""),
                    "user_id": user_id,
                })

        # 5. 写 PG chunks（tsv 倒排索引在这里建立）；只有向量索引的格式才写 Qdrant
        ChunkRepository.bulk_insert(pg_chunks)
        if use_vector:
            vector_store.upsert(qdrant_ids, embeddings, qdrant_payloads)

        # 5b. 图片文档额外写入图片向量集合（以图搜图），与文本集合隔离
        #     point_id 与 rag_chunks 中的 chunk 共用，检索到后可直接回查 PG 取描述/原图
        if isinstance(parser, ImageParser) and pg_chunks:
            try:
                prepared, _, _ = ImageParser._prepare(file_bytes)
                image_vec = llm_client.embed_image(prepared)
                point_id = pg_chunks[0]["qdrant_point_id"]
                image_vector_store.upsert(
                    [point_id],
                    [image_vec],
                    [{"qdrant_point_id": point_id, "user_id": user_id, "filename": filename}],
                )
            except Exception as e:
                # 图片向量写入失败不阻塞主流程，仅影响以图搜图能力
                logger.error(f"Image vector ingest failed for document {document_id} ({filename}): {e}")

        # 6. 回填解析结果，置为 completed
        DocumentRepository.mark_completed(
            document_id,
            total_pages=result.total_pages,
            table_count=result.table_count,
            chunk_count=len(chunks),
            raw_text=result.raw_text,
        )
        clear_chat_cache()  # 新文档可检索，清空问答缓存
        # 若开启自动重生成（WIKI_AUTO_REGEN=true），防抖触发 Wiki 重建；默认关闭，失败不影响入库
        try:
            from src.wiki.service import schedule_wiki_regen
            schedule_wiki_regen(user_id)
        except Exception as _e:
            logger.warning(f"触发 Wiki 自动重生成失败（不影响入库）: {_e}")
        logger.info(f"Document {document_id} ({filename}) ingested: {len(chunks)} chunks, index={parser.INDEX_TYPE}")
    except Exception as e:
        logger.error(f"Ingest failed for document {document_id} ({filename}): {e}")
        DocumentRepository.mark_failed(document_id, _friendly_ingest_error(e))


def _sec(dt: float) -> float:
    """耗时统一上报为秒，保留 3 位小数（即毫秒精度）"""
    return round(dt, 3)


def _table_key(row: Dict[str, Any]) -> Optional[tuple]:
    """表格归属键：(document_id, table_index)；非表格或旧数据（无 table_index）返回 None"""
    table_index = row.get("table_index")
    if table_index is None:
        return None
    return (str(row.get("document_id")), int(table_index))


def _expand_tables(pg_map: Dict[str, Dict[str, Any]], merged_ids: List[str]) -> None:
    """把命中的表格子块原地替换为整表内容。

    大表被按行窗口切开后，命中任一子块都能看到完整的表。同一张表只查一次兄弟块；
    整表超过上限时由 merge_table_chunks 截断并注明。
    """
    cache: Dict[tuple, str] = {}
    for pid in merged_ids:
        row = pg_map[pid]
        key = _table_key(row)
        if key is None:
            continue
        if key not in cache:
            siblings = ChunkRepository.get_table_siblings(key[0], key[1])
            # 单块表无需回填；多于一块才合并
            cache[key] = merge_table_chunks([s["content"] for s in siblings]) if len(siblings) > 1 else ""
        if cache[key]:
            row["content"] = cache[key]


def retrieve(query: str, user_id: str, top_k: Optional[int] = None, use_rerank: bool = True,
             use_bm25: bool = True, timings: Optional[Dict[str, Any]] = None) -> List[dict]:
    """混合检索：向量召回 + BM25 关键词召回 → RRF 融合 → 重排 → 回 PostgreSQL 取完整 chunk

    - 向量路：语义相近（Qdrant，按 user_id 过滤）
    - BM25 路：字面精确（PG tsvector，JOIN documents 按 user_id 过滤）
      两路互补：编号/专名/错误码靠 BM25，同义改写靠向量
      JSON 文档只建了倒排索引（不写 Qdrant），因此只可能从 BM25 路召回
    - use_bm25=False 时走纯向量模式（跳过 BM25 与 RRF），用于检索策略 A/B 对比评测
       此时 JSON 文档不可见
    - timings：传入 dict 时原地写入各阶段耗时（毫秒），供前端展示链路耗时
    返回：[{content, section, type, images, score, filename, index}, ...]
    """
    top_k = top_k or settings.retrieval_top_k
    t_start = time.perf_counter()

    # 1. 向量召回（按 user_id 过滤，只召回当前用户的文档）
    query_vec = llm_client.embed_query(query)
    t_embed = time.perf_counter()
    filter_ = Filter(must=[
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ])
    vec_candidates = vector_store.search(query_vec, top_k=top_k, filter_=filter_)
    t_vector = time.perf_counter()
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
    t_bm25 = time.perf_counter()
    if timings is not None:
        timings.update({
            "embed_query_s": _sec(t_embed - t_start),
            "vector_search_s": _sec(t_vector - t_embed),
            "bm25_s": _sec(t_bm25 - t_vector),
        })
    if not merged_ids:
        if timings is not None:
            timings["retrieve_total_s"] = _sec(t_bm25 - t_start)
        return []

    # 4. 回 PG 取完整数据
    pg_rows = ChunkRepository.get_by_qdrant_ids(merged_ids)
    t_pg = time.perf_counter()
    pg_map = {str(r["qdrant_point_id"]): r for r in pg_rows}
    merged_ids = [pid for pid in merged_ids if pid in pg_map]
    if timings is not None:
        timings["pg_fetch_s"] = _sec(t_pg - t_bm25)
    if not merged_ids:
        if timings is not None:
            timings["retrieve_total_s"] = _sec(t_pg - t_start)
        return []

    # 4b. 表格回填：命中表格子块时还原整表（必须在重排之前，否则重排仍只看行块）
    #     同一张表的多个子块回填后内容相同，只保留排名最靠前的一块
    t_expand = time.perf_counter()
    _expand_tables(pg_map, merged_ids)
    deduped: List[str] = []
    seen_tables = set()
    for pid in merged_ids:
        key = _table_key(pg_map[pid])
        if key is not None:
            if key in seen_tables:
                continue
            seen_tables.add(key)
        deduped.append(pid)
    merged_ids = deduped
    t_dedup = time.perf_counter()
    if timings is not None:
        timings["table_expand_s"] = _sec(t_dedup - t_expand)

    # 5. 重排（用 content 文本重排，从融合候选中取 top_n）
    if use_rerank and len(merged_ids) > 1:
        docs = [pg_map[pid]["content"] for pid in merged_ids]
        rerank_indices = llm_client.rerank(query, docs, top_n=settings.rerank_top_n)
        merged_ids = [merged_ids[i] for i in rerank_indices]
    else:
        merged_ids = merged_ids[:settings.rerank_top_n]
    t_rerank = time.perf_counter()
    if timings is not None:
        timings["rerank_s"] = _sec(t_rerank - t_pg)
        timings["retrieve_total_s"] = _sec(t_rerank - t_start)

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


def retrieve_images(image_bytes: bytes, user_id: str, top_k: Optional[int] = None) -> List[dict]:
    """以图搜图：图片预处理 → 多模态 embedding → 搜 rag_images 集合 → 回 PG 取描述/原图

    检索的是图片向量集合（与文本向量同 point_id），按 user_id 隔离。
    返回：[{content, images, score, filename}, ...]，score 为余弦相似度
    """
    top_k = top_k or settings.retrieval_top_k

    # 与入库时同样的预处理（EXIF 转正 + 缩放到长边 1568 + JPEG），保证向量可比
    prepared, _, _ = ImageParser._prepare(image_bytes)
    query_vec = llm_client.embed_image(prepared)

    filter_ = Filter(must=[
        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    ])
    candidates = image_vector_store.search(query_vec, top_k=top_k, filter_=filter_)
    if not candidates:
        return []

    point_ids = [c["payload"]["qdrant_point_id"] for c in candidates]
    score_map = {c["payload"]["qdrant_point_id"]: c.get("score", 0.0) for c in candidates}
    filename_map = {c["payload"]["qdrant_point_id"]: c["payload"].get("filename", "") for c in candidates}

    # 回 PG 取完整 chunk（VLM 描述 + 原图路径）
    pg_rows = ChunkRepository.get_by_qdrant_ids(point_ids)
    pg_map = {str(r["qdrant_point_id"]): r for r in pg_rows}

    results = []
    for pid in point_ids:
        row = pg_map.get(pid)
        if row is None:
            continue
        results.append({
            "content": row["content"],
            "images": row.get("images", []),
            "score": score_map.get(pid, 0.0),
            "filename": filename_map.get(pid, ""),
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


def _build_chat_messages(
    query: str,
    user_id: str,
    history: List[dict],
    use_rerank: bool,
    timings: Dict[str, Any],
):
    """检索（传统 chunk + 可选 Wiki 页面）并组装发给 LLM 的 messages。

    实际逻辑委托给 src.wiki.chat_integration.build_chat_context（torch-free，可独立单测），
    本函数仅注入 pipeline.retrieve 与 SYSTEM_PROMPT。
    """
    from src.wiki.chat_integration import build_chat_context
    return build_chat_context(query, user_id, history, use_rerank, timings, retrieve, SYSTEM_PROMPT)


def _chat_generate(query: str, user_id: str, history: List[dict], use_rerank: bool = True) -> Dict[str, Any]:
    """检索 + LLM 生成（带缓存），不做落库

    返回值额外带 timings（各阶段耗时，秒，3 位小数）：
      - cached=True 时只有 total_s（缓存查询本身的开销）
      - 未命中时含 embed_query_s / vector_search_s / bm25_s / pg_fetch_s /
        rerank_s / retrieve_total_s / wiki_s / llm_s / total_s
    """
    t_start = time.perf_counter()
    cache_key = _chat_cache_key(query, user_id, history, use_rerank)
    cached = _chat_cache.get(cache_key)
    if cached is not None:
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "wiki": cached.get("wiki", {"used": False, "count": 0, "error": None}),
            "timings": {"cached": True, "total_s": _sec(time.perf_counter() - t_start)},
        }

    timings: Dict[str, Any] = {}
    messages, sources, wiki_info = _build_chat_messages(
        query, user_id, history, use_rerank, timings
    )

    if not sources:
        # 无任何资料（chunk 与 Wiki 均空）：不调 LLM，直接兜底回答
        answer = "知识库中没有相关内容，请尝试其他问题。"
    else:
        t_llm = time.perf_counter()
        answer = llm_client.chat(messages)
        timings["llm_s"] = _sec(time.perf_counter() - t_llm)

    timings["cached"] = False
    timings["total_s"] = _sec(time.perf_counter() - t_start)

    # 缓存答案、来源与 Wiki 状态；耗时每次重新统计
    _chat_cache.set(cache_key, {"answer": answer, "sources": sources, "wiki": wiki_info})
    return {"answer": answer, "sources": sources, "wiki": wiki_info, "timings": timings}


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
    - 返回：{answer, sources, wiki, conversation_id, timings}
      wiki = {used, count, error} 标识本次是否命中 Wiki 页面（可观测信号）
    """
    t_start = time.perf_counter()

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
    wiki_info = result.get("wiki", {"used": False, "count": 0, "error": None})
    timings = result.get("timings", {})

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

    # 5. 服务端总耗时 = 请求进入 → 检索/生成 + 落库全部完成
    timings["total_s"] = _sec(time.perf_counter() - t_start)

    return {
        "answer": answer,
        "sources": sources,
        "wiki": wiki_info,
        "conversation_id": conversation_id,
        "timings": timings,
    }


def chat_stream(
    query: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    use_rerank: bool = True,
):
    """流式问答：逐 token yield 内容，结束后落库

    生成器 yield 的内容：
      - {"type": "start", "conversation_id": str, "sources": list, "wiki": dict, "timings": dict}
      - {"type": "token", "content": str}  (多次)
      - {"type": "done", "wiki": dict, "timings": dict}
    wiki = {used, count, error} 标识本次是否命中 Wiki 页面（可观测信号）。

    timings（秒，3 位小数）：embed_query_s / vector_search_s / bm25_s / pg_fetch_s /
    rerank_s / retrieve_total_s / wiki_s / llm_ttft_s（模型首字延迟）/ llm_s（生成总时长）/
    total_s（服务端从接收到全部完成的总耗时）；缓存命中时 cached=True。
    """
    t_start = time.perf_counter()

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
    wiki_info: Dict[str, Any] = {"used": False, "count": 0, "error": None}

    if cached is not None:
        # 缓存命中：把答案拆成小块"流式"吐出，保证 UX 一致
        answer = cached["answer"]
        sources = cached["sources"]
        wiki_info = cached.get("wiki", wiki_info)
        timings: Dict[str, Any] = {"cached": True}
        yield {"type": "start", "conversation_id": conversation_id,
               "sources": sources, "wiki": dict(wiki_info), "timings": dict(timings)}
        chunk_size = 3
        for i in range(0, len(answer), chunk_size):
            yield {"type": "token", "content": answer[i:i + chunk_size]}
    else:
        # 缓存未命中：检索（含 Wiki）+ 流式生成
        timings = {}
        messages, sources, wiki_info = _build_chat_messages(
            query, user_id, history, use_rerank, timings
        )

        # start 事件带检索阶段耗时快照（dict 拷贝，避免后续生成耗时覆盖）
        yield {"type": "start", "conversation_id": conversation_id,
               "sources": sources, "wiki": dict(wiki_info), "timings": dict(timings)}

        if not sources:
            # 无任何资料：不调 LLM，直接兜底（逐字吐出）
            answer = "知识库中没有相关内容，请尝试其他问题。"
            for i in range(0, len(answer), 3):
                yield {"type": "token", "content": answer[i:i + 3]}
        else:
            answer_parts = []
            t_llm = time.perf_counter()
            for token in llm_client.chat_stream(messages):
                if "llm_ttft_s" not in timings:
                    timings["llm_ttft_s"] = _sec(time.perf_counter() - t_llm)
                answer_parts.append(token)
                yield {"type": "token", "content": token}
            answer = "".join(answer_parts)
            timings["llm_s"] = _sec(time.perf_counter() - t_llm)
        timings["cached"] = False
        _chat_cache.set(cache_key, {"answer": answer, "sources": sources, "wiki": wiki_info})

    # 4. 落库
    MessageRepository.create(conversation_id, role="user", content=query)
    MessageRepository.create(
        conversation_id,
        role="assistant",
        content=answer,
        sources=sources,
    )
    ConversationRepository.touch(conversation_id)

    # 5. 总耗时 = 从接收请求到落库完成
    timings["total_s"] = _sec(time.perf_counter() - t_start)

    yield {"type": "done", "wiki": dict(wiki_info), "timings": timings}

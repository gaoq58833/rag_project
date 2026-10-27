"""全文检索（BM25 风格）：jieba 分词 + PostgreSQL tsvector/ts_rank

为什么需要它：向量检索对"语义相近"强，但对编号、错误码、专有名词等字面精确匹配弱
（"编号38"在向量空间没有强语义）。BM25 走关键词字面匹配，与向量检索互补。

实现说明：
- PostgreSQL 自带的 ts_rank 是 TF-IDF 风格的排序（词频 + 逆文档频率 + 长度归一化），
  与 BM25 同族但不完全等同，对本项目体量足够，无需引入 Elasticsearch。
- PG 自带的 to_tsvector 不会给中文分词，所以用 jieba 在应用侧分词，
  把空格分隔的词串交给 to_tsvector('simple', ...) 建立索引。
"""
import re
from typing import Any, Dict, List, Optional

import jieba

from src.db import db

# 只保留纯字母数字词和纯中文词，标点/符号不进索引（也保证 tsquery 拼接安全）
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[一-鿿]+")

# jieba 首次调用会加载词典，模块加载时预热，避免首次检索卡顿
jieba.initialize()


def tokenize(text: str) -> str:
    """文本 → 空格分隔的小写词串，供 to_tsvector('simple', ...) 建索引"""
    words = jieba.cut(text or "")
    return " ".join(w.lower() for w in words if _TOKEN_RE.fullmatch(w))


def bm25_search(query: str, user_id: Optional[str], top_k: int = 10) -> List[Dict[str, Any]]:
    """关键词召回：在 chunks.tsv 上做全文检索，按 ts_rank 排序

    通过 JOIN documents 过滤 user_id，与向量检索同样保证用户隔离。
    查询词之间用 OR 连接：关键词路求召回率，精确排序交给后面的 rerank。
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    # _TOKEN_RE 已过滤 tsquery 特殊字符（& | ! ( ) 等），可安全拼接
    tsquery = " | ".join(tokens.split())
    sql = """
        SELECT c.qdrant_point_id, c.content, d.filename,
               ts_rank(c.tsv, to_tsquery('simple', %s)) AS rank
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.user_id = %s AND c.tsv @@ to_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    return db.query_all(sql, (tsquery, user_id, tsquery, top_k))


def rrf_fuse(vector_ids: List[str], bm25_rows: List[Dict[str, Any]], k: int = 60) -> Dict[str, float]:
    """RRF（Reciprocal Rank Fusion）融合两路召回结果

    score = Σ 1/(k + rank)，rank 是各路内部的排名（从 1 开始）。
    不依赖两路分数的量纲（向量是余弦相似度、BM25 是 ts_rank），只看排名，融合稳定。

    返回：{qdrant_point_id: 融合分数}（未排序，调用方按需排序）
    """
    scores: Dict[str, float] = {}

    for rank, pid in enumerate(vector_ids, start=1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)

    for rank, row in enumerate(bm25_rows, start=1):
        pid = str(row["qdrant_point_id"])
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)

    return scores

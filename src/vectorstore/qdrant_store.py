"""Qdrant 向量存储：HNSW 索引 + payload 过滤（懒加载，连接失败不阻塞导入）"""
from typing import List, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import settings


class VectorStore:
    """封装 Qdrant 操作：建库、入库、检索"""

    def __init__(self):
        self.collection = settings.qdrant_collection
        self._client: Optional[QdrantClient] = None
        self._collection_ready = False

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if settings.qdrant_path:
                # 本地嵌入式模式：无需启动 Qdrant 服务
                self._client = QdrantClient(path=settings.qdrant_path)
            else:
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                )
        return self._client

    def _ensure_collection(self):
        """集合不存在则创建，使用 HNSW 索引"""
        if self._collection_ready:
            return
        client = self._get_client()
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if self.collection not in names:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,  # 从 .env 读取（EMBEDDING_DIM）
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
                # HNSW 索引参数
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
            )
        self._collection_ready = True

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[dict]):
        """批量写入向量 + payload"""
        self._ensure_collection()
        points = [
            PointStruct(id=id_, vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]
        self._get_client().upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_: Optional[dict] = None,
    ) -> List[dict]:
        """向量检索，返回 payload + score"""
        self._ensure_collection()
        top_k = top_k or settings.retrieval_top_k
        results = self._get_client().query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=models.Filter(**filter_) if filter_ else None,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"payload": r.payload, "score": r.score}
            for r in results.points
        ]

    def delete_by_filename(self, filename: str):
        """按文件名删除（覆盖上传时先删旧的）"""
        self._ensure_collection()
        self._get_client().delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename),
                        )
                    ]
                )
            ),
        )

    def count(self) -> int:
        """返回集合中的向量数"""
        self._ensure_collection()
        return self._get_client().count(collection_name=self.collection).count


# 全局单例（懒加载，导入时不连 Qdrant）
vector_store = VectorStore()

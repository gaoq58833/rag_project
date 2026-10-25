"""Repository 层：documents 和 chunks 表的增删查"""
import json
import uuid
from typing import Any, Dict, List, Optional

from src.db import db


class DocumentRepository:
    """文档表操作"""

    @staticmethod
    def create(
        filename: str,
        file_ext: str,
        total_pages: int,
        table_count: int,
        chunk_count: int,
        raw_text: Optional[str] = None,
    ) -> str:
        """插入文档记录，返回 document_id (UUID 字符串)"""
        doc_id = str(uuid.uuid4())
        db.execute(
            """
            INSERT INTO documents (id, filename, file_ext, total_pages, table_count, chunk_count, raw_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, filename, file_ext, total_pages, table_count, chunk_count, raw_text),
        )
        return doc_id

    @staticmethod
    def get_by_filename(filename: str) -> Optional[Dict[str, Any]]:
        return db.query_one(
            "SELECT * FROM documents WHERE filename = %s ORDER BY uploaded_at DESC LIMIT 1",
            (filename,),
        )

    @staticmethod
    def delete_by_filename(filename: str) -> int:
        """按文件名删除文档及其所有 chunks（ON DELETE CASCADE 自动删 chunks）"""
        return db.execute("DELETE FROM documents WHERE filename = %s", (filename,))

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        return db.query_all("SELECT * FROM documents ORDER BY uploaded_at DESC")


class ChunkRepository:
    """分块表操作"""

    @staticmethod
    def bulk_insert(chunks: List[Dict[str, Any]]) -> int:
        """批量插入 chunks，返回插入行数

        每个 chunk dict 需包含：document_id, qdrant_point_id, chunk_index,
        content, length, chunk_type, section, images(list)
        """
        if not chunks:
            return 0
        sql = """
            INSERT INTO chunks (document_id, qdrant_point_id, chunk_index, content, length, chunk_type, section, images)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        params = [
            (
                c["document_id"],
                c["qdrant_point_id"],
                c["chunk_index"],
                c["content"],
                c["length"],
                c.get("chunk_type", "text"),
                c.get("section"),
                json.dumps(c.get("images", []), ensure_ascii=False),
            )
            for c in chunks
        ]
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
                return cur.rowcount

    @staticmethod
    def get_by_qdrant_ids(qdrant_ids: List[str]) -> List[Dict[str, Any]]:
        """根据 Qdrant point id 列表批量回查 chunk 完整数据"""
        if not qdrant_ids:
            return []
        # 用 ANY 传入 UUID 数组
        sql = "SELECT * FROM chunks WHERE qdrant_point_id = ANY(%s::uuid[])"
        return db.query_all(sql, (qdrant_ids,))

    @staticmethod
    def get_by_document_id(document_id: str) -> List[Dict[str, Any]]:
        return db.query_all(
            "SELECT * FROM chunks WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )

    @staticmethod
    def delete_by_document_id(document_id: str) -> int:
        return db.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

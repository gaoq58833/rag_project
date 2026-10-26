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
    def get_by_id(document_id: str) -> Optional[Dict[str, Any]]:
        return db.query_one(
            "SELECT * FROM documents WHERE id = %s",
            (document_id,),
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


class UserRepository:
    """用户表操作"""

    @staticmethod
    def create(username: str, password_hash: str, role: str = "user") -> str:
        """创建用户，返回 user_id"""
        user_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO users (id, username, password_hash, role) VALUES (%s, %s, %s, %s)",
            (user_id, username, password_hash, role),
        )
        return user_id

    @staticmethod
    def get_by_username(username: str) -> Optional[Dict[str, Any]]:
        return db.query_one("SELECT * FROM users WHERE username = %s", (username,))

    @staticmethod
    def get_by_id(user_id: str) -> Optional[Dict[str, Any]]:
        return db.query_one("SELECT * FROM users WHERE id = %s", (user_id,))


class ConversationRepository:
    """对话表操作"""

    @staticmethod
    def create(user_id: str, title: str = "新对话") -> str:
        conv_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO conversations (id, user_id, title) VALUES (%s, %s, %s)",
            (conv_id, user_id, title),
        )
        return conv_id

    @staticmethod
    def get_by_id(conv_id: str) -> Optional[Dict[str, Any]]:
        return db.query_one("SELECT * FROM conversations WHERE id = %s", (conv_id,))

    @staticmethod
    def list_by_user(user_id: str) -> List[Dict[str, Any]]:
        return db.query_all(
            "SELECT * FROM conversations WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )

    @staticmethod
    def update_title(conv_id: str, title: str) -> int:
        return db.execute(
            "UPDATE conversations SET title = %s, updated_at = now() WHERE id = %s",
            (title, conv_id),
        )

    @staticmethod
    def touch(conv_id: str) -> int:
        return db.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s", (conv_id,)
        )

    @staticmethod
    def delete(conv_id: str) -> int:
        return db.execute("DELETE FROM conversations WHERE id = %s", (conv_id,))


class MessageRepository:
    """消息表操作"""

    @staticmethod
    def create(
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Any]] = None,
    ) -> str:
        msg_id = str(uuid.uuid4())
        db.execute(
            """INSERT INTO messages (id, conversation_id, role, content, sources)
               VALUES (%s, %s, %s, %s, %s::jsonb)""",
            (msg_id, conversation_id, role, content, json.dumps(sources or [], ensure_ascii=False)),
        )
        return msg_id

    @staticmethod
    def get_recent(conversation_id: str, rounds: int) -> List[Dict[str, Any]]:
        """取最近 rounds 轮（即 2*rounds 条消息），按时间升序返回"""
        limit = rounds * 2
        rows = db.query_all(
            """SELECT * FROM messages
               WHERE conversation_id = %s
               ORDER BY created_at DESC
               LIMIT %s""",
            (conversation_id, limit),
        )
        # 倒序取出来的，再翻成升序
        rows.reverse()
        return rows

    @staticmethod
    def list_by_conversation(conversation_id: str) -> List[Dict[str, Any]]:
        return db.query_all(
            "SELECT * FROM messages WHERE conversation_id = %s ORDER BY created_at ASC",
            (conversation_id,),
        )

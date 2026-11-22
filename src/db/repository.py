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
        content_hash: Optional[str] = None,
        user_id: Optional[str] = None,
        status: str = "processing",
        file_size: Optional[int] = None,
        wiki_eligible: bool = False,
    ) -> str:
        """插入文档记录，返回 document_id (UUID 字符串)。
        file_size：原始文件字节数（判断小文档）；wiki_eligible：是否参与 LLM Wiki 生成。"""
        doc_id = str(uuid.uuid4())
        db.execute(
            """
            INSERT INTO documents (id, user_id, filename, file_ext, content_hash, total_pages, table_count, chunk_count, raw_text, status, file_size, wiki_eligible)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (doc_id, user_id, filename, file_ext, content_hash, total_pages, table_count, chunk_count, raw_text, status, file_size, wiki_eligible),
        )
        return doc_id

    @staticmethod
    def set_wiki_eligible(document_id: str, user_id: str, eligible: bool) -> int:
        """用户手动切换某文档是否参与 Wiki 生成（用户隔离校验）。返回受影响行数。"""
        return db.execute(
            "UPDATE documents SET wiki_eligible = %s WHERE id = %s AND user_id = %s",
            (eligible, document_id, user_id),
        )

    @staticmethod
    def mark_completed(document_id: str, total_pages: int, table_count: int, chunk_count: int, raw_text: Optional[str]) -> int:
        """异步入库成功：回填解析结果并把状态置为 completed"""
        return db.execute(
            """
            UPDATE documents
            SET status = 'completed', total_pages = %s, table_count = %s, chunk_count = %s, raw_text = %s, error_msg = NULL
            WHERE id = %s
            """,
            (total_pages, table_count, chunk_count, raw_text, document_id),
        )

    @staticmethod
    def mark_failed(document_id: str, error_msg: str) -> int:
        """异步入库失败：置为 failed 并记录错误信息（截断防超长）"""
        return db.execute(
            "UPDATE documents SET status = 'failed', error_msg = %s WHERE id = %s",
            (error_msg[:500], document_id),
        )

    @staticmethod
    def reset_for_update(document_id: str, filename: str, file_ext: str, content_hash: str, file_size: Optional[int] = None) -> int:
        """热更新：文档行重置为 processing（ID 不变），等待后台重新入库。
        file_size 同步刷新；wiki_eligible 保持用户手动选择，不在此覆盖。"""
        return db.execute(
            """
            UPDATE documents
            SET filename = %s, file_ext = %s, content_hash = %s, status = 'processing',
                total_pages = 0, table_count = 0, chunk_count = 0, raw_text = NULL,
                error_msg = NULL, uploaded_at = now(),
                file_size = COALESCE(%s, file_size)
            WHERE id = %s
            """,
            (filename, file_ext, content_hash, file_size, document_id),
        )

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
    def get_by_content_hash(content_hash: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """按内容哈希查找同用户已有文档（用于去重复用；向量 payload 带 user_id，只能复用本人的）"""
        return db.query_one(
            "SELECT * FROM documents WHERE content_hash = %s AND user_id = %s ORDER BY uploaded_at ASC LIMIT 1",
            (content_hash, user_id),
        )

    @staticmethod
    def delete_by_filename(filename: str) -> int:
        """按文件名删除文档及其所有 chunks（ON DELETE CASCADE 自动删 chunks）"""
        return db.execute("DELETE FROM documents WHERE filename = %s", (filename,))

    @staticmethod
    def delete_by_id(document_id: str) -> int:
        """按 ID 删除文档及其所有 chunks（ON DELETE CASCADE 自动删 chunks）"""
        return db.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        return db.query_all("SELECT * FROM documents ORDER BY uploaded_at DESC")

    @staticmethod
    def list_by_user(user_id: str) -> List[Dict[str, Any]]:
        """只返回指定用户上传的文档"""
        return db.query_all(
            "SELECT * FROM documents WHERE user_id = %s ORDER BY uploaded_at DESC",
            (user_id,),
        )


class ChunkRepository:
    """分块表操作"""

    @staticmethod
    def bulk_insert(chunks: List[Dict[str, Any]]) -> int:
        """批量插入 chunks，返回插入行数

        每个 chunk dict 需包含：document_id, qdrant_point_id, chunk_index,
        content, length, chunk_type, section, images(list), tokens(jieba 分词词串)
        """
        if not chunks:
            return 0
        sql = """
            INSERT INTO chunks (document_id, qdrant_point_id, chunk_index, content, length, chunk_type, section, images, tsv)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, to_tsvector('simple', %s))
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
                c.get("tokens", ""),
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
    def get_wiki_eligible_chunks(user_id: str) -> List[Dict[str, Any]]:
        """取该用户 wiki_eligible=true 文档的全部 chunk（含 filename/content/section），供 Wiki 聚类生成。"""
        return db.query_all(
            """
            SELECT c.qdrant_point_id, c.content, c.section, c.document_id, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.user_id = %s AND d.wiki_eligible = true
            """,
            (user_id,),
        )

    @staticmethod
    def get_sources_by_chunk_ids(user_id: str, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Wiki 溯源：按 chunk id 回查内容 + 所属文件名（用户隔离），供前端展示来源。"""
        if not chunk_ids:
            return []
        sql = """
            SELECT c.qdrant_point_id, c.content, c.section, c.chunk_index, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.user_id = %s AND c.qdrant_point_id = ANY(%s::uuid[])
        """
        return db.query_all(sql, (user_id, chunk_ids))

    @staticmethod
    def delete_by_document_id(document_id: str) -> int:
        return db.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

    @staticmethod
    def count_by_qdrant_ids_excluding_doc(qdrant_ids: List[str], exclude_document_id: str) -> int:
        """统计这些 point_id 被其他文档引用的次数（用于判断删 Qdrant 是否安全）"""
        if not qdrant_ids:
            return 0
        sql = """
            SELECT COUNT(*) FROM chunks
            WHERE qdrant_point_id = ANY(%s::uuid[]) AND document_id != %s
        """
        row = db.query_one(sql, (qdrant_ids, exclude_document_id))
        return row["count"] if row else 0


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

    @staticmethod
    def set_avatar(user_id: str, avatar_url: Optional[str]) -> int:
        """设置用户头像 URL（传 None 清除头像）"""
        return db.execute(
            "UPDATE users SET avatar_url = %s WHERE id = %s",
            (avatar_url, user_id),
        )


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


class WikiPageRepository:
    """Wiki 页面表操作（LLM 自动生成的知识页面）"""

    @staticmethod
    def upsert(
        user_id: str,
        slug: str,
        title: str,
        summary: Optional[str],
        content_md: str,
        paragraphs: List[Dict[str, Any]],
        outgoing_links: List[str],
        source_chunk_ids: List[str],
        tokens: str,
        status: str = "ready",
    ) -> str:
        """插入或更新页面（按 user_id+slug 去重），返回 page_id。
        tokens 为 jieba 分词后的空格分隔词串，用于建立 tsv 全文索引。"""
        page_id = str(uuid.uuid4())
        sql = """
            INSERT INTO wiki_pages
                (id, user_id, slug, title, summary, content_md, paragraphs,
                 outgoing_links, source_chunk_ids, status, tsv)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                    to_tsvector('simple', %s))
            ON CONFLICT (user_id, slug) DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                content_md = EXCLUDED.content_md,
                paragraphs = EXCLUDED.paragraphs,
                outgoing_links = EXCLUDED.outgoing_links,
                source_chunk_ids = EXCLUDED.source_chunk_ids,
                status = EXCLUDED.status,
                error_msg = NULL,
                tsv = EXCLUDED.tsv,
                updated_at = now()
            RETURNING id
        """
        row = db.query_one(
            sql,
            (
                page_id, user_id, slug, title, summary, content_md,
                json.dumps(paragraphs or [], ensure_ascii=False),
                json.dumps(outgoing_links or [], ensure_ascii=False),
                json.dumps(source_chunk_ids or [], ensure_ascii=False),
                status, tokens,
            ),
        )
        return str(row["id"])

    @staticmethod
    def get_by_slug(user_id: str, slug: str) -> Optional[Dict[str, Any]]:
        return db.query_one(
            "SELECT * FROM wiki_pages WHERE user_id = %s AND slug = %s",
            (user_id, slug),
        )

    @staticmethod
    def list_by_user(user_id: str) -> List[Dict[str, Any]]:
        """页面列表（不含大字段 paragraphs/content_md），供侧边目录用"""
        return db.query_all(
            """SELECT slug, title, summary, status, updated_at
               FROM wiki_pages WHERE user_id = %s ORDER BY updated_at DESC""",
            (user_id,),
        )

    @staticmethod
    def get_all_by_user(user_id: str) -> List[Dict[str, Any]]:
        """取该用户全部页面完整字段（构建关系图用）"""
        return db.query_all(
            "SELECT * FROM wiki_pages WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        )

    @staticmethod
    def delete_by_user(user_id: str) -> int:
        """全量重建前清空该用户旧页面"""
        return db.execute("DELETE FROM wiki_pages WHERE user_id = %s", (user_id,))

    @staticmethod
    def delete_by_slug(user_id: str, slug: str) -> int:
        return db.execute(
            "DELETE FROM wiki_pages WHERE user_id = %s AND slug = %s",
            (user_id, slug),
        )

    @staticmethod
    def mark_failed(user_id: str, slug: str, error_msg: str) -> int:
        return db.execute(
            "UPDATE wiki_pages SET status = 'failed', error_msg = %s, updated_at = now() WHERE user_id = %s AND slug = %s",
            (error_msg[:500], user_id, slug),
        )

    @staticmethod
    def mark_stale_by_chunk_ids(user_id: str, chunk_ids: List[str]) -> int:
        """文档删除时，把引用了这些 chunk 的页面标记为 stale（不删除，提示重新生成）"""
        if not chunk_ids:
            return 0
        return db.execute(
            """UPDATE wiki_pages SET status = 'stale', updated_at = now()
               WHERE user_id = %s AND source_chunk_ids ?| %s::text[]""",
            (user_id, chunk_ids),
        )


class WikiJobRepository:
    """Wiki 生成任务表操作（异步进度跟踪）"""

    @staticmethod
    def create(user_id: str, page_total: int = 0) -> str:
        job_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO wiki_jobs (id, user_id, status, page_total, page_done) VALUES (%s, %s, 'generating', %s, 0)",
            (job_id, user_id, page_total),
        )
        return job_id

    @staticmethod
    def get_latest(user_id: str) -> Optional[Dict[str, Any]]:
        return db.query_one(
            "SELECT * FROM wiki_jobs WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )

    @staticmethod
    def get_active(user_id: str) -> Optional[Dict[str, Any]]:
        """取该用户正在进行的生成任务（用于去重，避免重复消耗 token）"""
        return db.query_one(
            "SELECT * FROM wiki_jobs WHERE user_id = %s AND status = 'generating' ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )

    @staticmethod
    def update_progress(job_id: str, page_done: int, page_total: Optional[int] = None) -> int:
        if page_total is not None:
            return db.execute(
                "UPDATE wiki_jobs SET page_done = %s, page_total = %s, updated_at = now() WHERE id = %s",
                (page_done, page_total, job_id),
            )
        return db.execute(
            "UPDATE wiki_jobs SET page_done = %s, updated_at = now() WHERE id = %s",
            (page_done, job_id),
        )

    @staticmethod
    def mark_done(job_id: str, page_total: Optional[int] = None) -> int:
        if page_total is not None:
            return db.execute(
                "UPDATE wiki_jobs SET status = 'done', page_total = %s, updated_at = now() WHERE id = %s",
                (page_total, job_id),
            )
        return db.execute(
            "UPDATE wiki_jobs SET status = 'done', updated_at = now() WHERE id = %s",
            (job_id,),
        )

    @staticmethod
    def mark_failed(job_id: str, error_msg: str) -> int:
        return db.execute(
            "UPDATE wiki_jobs SET status = 'failed', error_msg = %s, updated_at = now() WHERE id = %s",
            (error_msg[:500], job_id),
        )

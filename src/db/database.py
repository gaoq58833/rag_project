"""PostgreSQL 连接池（懒加载，导入时不连接）"""
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

from src.config import settings


class Database:
    """封装 PostgreSQL 连接池与常用操作"""

    def __init__(self):
        self._pool: Optional[SimpleConnectionPool] = None

    def _get_pool(self) -> SimpleConnectionPool:
        if self._pool is None:
            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.pg_host,
                port=settings.pg_port,
                dbname=settings.pg_db,
                user=settings.pg_user,
                password=settings.pg_password,
            )
        return self._pool

    @contextmanager
    def get_conn(self):
        """获取连接（自动归还连接池）"""
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            pool.putconn(conn)

    def execute(self, sql: str, params: tuple = ()) -> int:
        """执行写操作（INSERT/UPDATE/DELETE），返回受影响行数"""
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """查询单行，返回 dict 或 None"""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return dict(row) if row else None

    def query_all(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """查询多行，返回 dict 列表"""
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]

    def init_schema(self):
        """执行 schema.sql 建表（幂等）"""
        from pathlib import Path
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text(encoding="utf-8")
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def close(self):
        if self._pool:
            self._pool.closeall()
            self._pool = None


# 全局单例（懒加载）
db = Database()

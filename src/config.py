"""全局配置：从环境变量 / .env 加载"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载项目根目录的 .env
except ImportError:
    pass


@dataclass
class Settings:
    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3.7-plus")

    # Embedding
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-flash")

    # Reranker
    reranker_api_key: str = os.getenv("RERANKER_API_KEY") or os.getenv("LLM_API_KEY", "")
    reranker_base_url: str = os.getenv("RERANKER_BASE_URL") or os.getenv("LLM_BASE_URL", "")
    reranker_model: str = os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_path: str = os.getenv("QDRANT_PATH", "")  # 设了则用本地嵌入式模式，无需启服务
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    # PostgreSQL
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "rag")
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")

    # RAG
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))


settings = Settings()

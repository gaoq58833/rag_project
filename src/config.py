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
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))  # 换 embedding 模型时同步修改

    # Reranker
    reranker_api_key: str = os.getenv("RERANKER_API_KEY") or os.getenv("LLM_API_KEY", "")
    reranker_base_url: str = os.getenv("RERANKER_BASE_URL") or os.getenv("LLM_BASE_URL", "")
    reranker_model: str = os.getenv("RERANKER_MODEL", "qwen3-rerank")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_path: str = os.getenv("QDRANT_PATH", "")  # 设了则用本地嵌入式模式，无需启服务
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    # HNSW 索引参数（仅创建集合时生效，改了需重建集合）
    qdrant_hnsw_m: int = int(os.getenv("QDRANT_HNSW_M", "16"))
    qdrant_hnsw_ef_construct: int = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "100"))
    qdrant_full_scan_threshold: int = int(os.getenv("QDRANT_FULL_SCAN_THRESHOLD", "10000"))

    # PostgreSQL
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "rag")
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")

    # 分块参数
    chunk_max_size: int = int(os.getenv("CHUNK_MAX_SIZE", "1000"))
    chunk_min_size: int = int(os.getenv("CHUNK_MIN_SIZE", "100"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # RAG 检索
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "3"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "10"))  # BM25 关键词召回数量
    chat_history_rounds: int = int(os.getenv("CHAT_HISTORY_ROUNDS", "5"))  # 多轮对话保留最近 N 轮

    # LLM 生成参数
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # 问答缓存
    chat_cache_ttl: int = int(os.getenv("CHAT_CACHE_TTL", "600"))  # 缓存过期时间（秒）
    chat_cache_maxsize: int = int(os.getenv("CHAT_CACHE_MAXSIZE", "500"))  # 最大缓存条目（LRU）
    chat_cache_history_sig_len: int = int(os.getenv("CHAT_CACHE_HISTORY_SIG_LEN", "20"))  # 历史签名取每条前 N 字

    # JWT 鉴权
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 默认 24 小时


settings = Settings()

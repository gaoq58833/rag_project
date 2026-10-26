"""API 响应模型"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class FormatsResponse(BaseModel):
    supported_extensions: List[str]


class TableInfo(BaseModel):
    table_number: int
    page: Optional[int] = None
    markdown: str


class Chunk(BaseModel):
    content: str
    index: int
    length: int
    type: str  # text / table / code
    section: Optional[str] = None
    images: Optional[List[str]] = None


class ParseResponse(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int  # 字符数（与各解析器保持一致）
    table_count: int
    raw_text: str
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ChunkResponse(BaseModel):
    filename: str
    chunk_count: int
    chunks: List[Chunk]


class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None


# ===== RAG 相关 =====

class IngestResponse(BaseModel):
    filename: str
    chunk_count: int
    stored_count: int
    reused: bool = False


class RetrievedChunk(BaseModel):
    content: str
    section: Optional[str] = None
    type: str = "text"
    images: Optional[List[str]] = None
    score: float
    filename: str
    index: int


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]


class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    conversation_id: str


class ChatRequest(BaseModel):
    query: str
    use_rerank: bool = True
    conversation_id: Optional[str] = None  # 不传则新建对话，传则追加到该对话


# ===== 鉴权相关 =====

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=15)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str


class UserInfo(BaseModel):
    id: str
    username: str
    role: str


# ===== 多轮对话相关 =====

class Conversation(BaseModel):
    id: str
    title: str
    created_at: Any
    updated_at: Any


class Message(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources: Optional[List[Any]] = None
    created_at: Any


class ConversationListResponse(BaseModel):
    conversations: List[Conversation]


class MessageListResponse(BaseModel):
    messages: List[Message]

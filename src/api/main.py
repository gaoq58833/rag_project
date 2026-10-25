"""FastAPI 应用：文档解析与分块接口

启动：
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
或：
  python -m src.api.main
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AuthResponse,
    ChatRequest,
    ChunkResponse,
    FormatsResponse,
    HealthResponse,
    IngestResponse,
    LoginRequest,
    ParseResponse,
    RegisterRequest,
    RetrieveResponse,
    UserInfo,
)
from src.auth import create_access_token, hash_password, verify_password
from src.auth.deps import get_current_user
from src.db import db
from src.db.repository import UserRepository
from src.parsers.chunker import ChunkStrategy, TextChunker
from src.parsers.factory import get_parser, supported_extensions
from src.rag import chat as rag_chat, ingest_file, retrieve as rag_retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时自动建表（PostgreSQL）
    try:
        db.init_schema()
        print("[startup] PostgreSQL 表结构已初始化")
    except Exception as e:
        print(f"[startup] PostgreSQL 建表失败（请检查数据库连接）：{e}")

    # 启动时预热 PDFParser（加载 docling 模型），避免首次请求过慢
    try:
        get_parser("warmup.pdf")
        print("[startup] PDFParser 模型已加载")
    except Exception as e:
        print(f"[startup] PDFParser 预热失败（不影响其他格式）：{e}")
    yield


app = FastAPI(
    title="RAG 文档解析 API",
    description="支持 PDF / Word / Markdown / HTML / TXT 的解析与分块",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 跨域配置（允许前端开发服务器访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 生产环境请改为具体域名，如 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],          # 允许所有 HTTP 方法
    allow_headers=["*"],          # 允许所有请求头（含 Authorization）
)

# 分块器参数（可按需调整）
DEFAULT_CHUNKER = TextChunker(
    strategy=ChunkStrategy.PARAGRAPH,
    max_chunk_size=1000,
    min_chunk_size=100,
    chunk_overlap=200,
)


@app.get("/", summary="根路由")
async def root():
    return {
        "name": "RAG 文档解析 API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": [
            "/health", "/formats", "/parse", "/chunk",
            "/auth/register", "/auth/login", "/auth/me",
            "/ingest", "/retrieve", "/chat",
        ],
    }


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health():
    return HealthResponse()


@app.get("/formats", response_model=FormatsResponse, summary="支持的文件格式")
async def formats():
    return FormatsResponse(supported_extensions=supported_extensions())


@app.post(
    "/parse",
    response_model=ParseResponse,
    summary="解析文档（返回原始文本、表格、元数据）",
)
async def parse_file(file: UploadFile = File(...)):
    """上传文件，自动识别格式并解析，不做分块。"""
    filename = file.filename or "unnamed"
    file_bytes = await file.read()

    try:
        parser = get_parser(filename)
        result = parser.parse_from_bytes(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")

    return ParseResponse(
        filename=result.filename,
        total_pages=result.total_pages,
        total_chunks=result.total_chunks,
        table_count=result.table_count,
        raw_text=result.raw_text,
        tables=result.tables,
        metadata=result.metadata,
    )


@app.post(
    "/chunk",
    response_model=ChunkResponse,
    summary="解析并分块（返回 chunks 列表）",
)
async def chunk_file(
    file: UploadFile = File(...),
    max_chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """上传文件，自动识别格式、解析并分块。

    可选查询参数覆盖默认分块大小。
    """
    filename = file.filename or "unnamed"
    file_bytes = await file.read()

    # 解析
    try:
        parser = get_parser(filename)
        result = parser.parse_from_bytes(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败：{e}")

    # 分块（支持自定义参数）
    if max_chunk_size or min_chunk_size or chunk_overlap:
        chunker = TextChunker(
            strategy=ChunkStrategy.PARAGRAPH,
            max_chunk_size=max_chunk_size or DEFAULT_CHUNKER.max_chunk_size,
            min_chunk_size=min_chunk_size or DEFAULT_CHUNKER.min_chunk_size,
            chunk_overlap=chunk_overlap or DEFAULT_CHUNKER.chunk_overlap,
        )
    else:
        chunker = DEFAULT_CHUNKER

    chunks = chunker.split(result.raw_text, tables=result.tables)

    return ChunkResponse(
        filename=result.filename,
        chunk_count=len(chunks),
        chunks=chunks,
    )


# ===== 鉴权接口 =====

@app.post(
    "/auth/register",
    response_model=AuthResponse,
    summary="用户注册",
)
async def register(request: RegisterRequest):
    """注册新用户，返回 JWT token。"""
    existing = UserRepository.get_by_username(request.username)
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user_id = UserRepository.create(
        username=request.username,
        password_hash=hash_password(request.password),
        role="user",
    )
    token = create_access_token(subject=user_id, role="user")
    return AuthResponse(
        access_token=token,
        user_id=user_id,
        username=request.username,
        role="user",
    )


@app.post(
    "/auth/login",
    response_model=AuthResponse,
    summary="用户登录",
)
async def login(request: LoginRequest):
    """用户名密码登录，返回 JWT token。"""
    user = UserRepository.get_by_username(request.username)
    if user is None or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_access_token(subject=str(user["id"]), role=user["role"])
    return AuthResponse(
        access_token=token,
        user_id=str(user["id"]),
        username=user["username"],
        role=user["role"],
    )


@app.get(
    "/auth/me",
    response_model=UserInfo,
    summary="获取当前登录用户信息",
)
async def me(current_user: dict = Depends(get_current_user)):
    """需要携带有效的 Authorization: Bearer <token>。"""
    return UserInfo(**current_user)


# ===== RAG 接口（需鉴权）=====

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="入库：上传文档 → 解析分块 → 向量化 → 存入 Qdrant",
)
async def ingest(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """上传文档，自动解析、分块、向量化并写入向量库。需登录。"""
    filename = file.filename or "unnamed"
    file_bytes = await file.read()

    try:
        result = ingest_file(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败：{e}")

    return IngestResponse(**result)


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="检索：query → 向量召回 → 重排 → 返回相关块",
)
async def retrieve(
    query: str,
    top_k: Optional[int] = None,
    use_rerank: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """根据 query 检索知识库中的相关片段。需登录。"""
    try:
        results = rag_retrieve(query, top_k=top_k, use_rerank=use_rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")

    return RetrieveResponse(query=query, results=results)


@app.post(
    "/chat",
    summary="问答：检索 → 构造上下文 → LLM 生成回答",
)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """基于知识库的问答，返回答案和引用来源。需登录。"""
    try:
        result = rag_chat(request.query, use_rerank=request.use_rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    return result


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": exc.__class__.__name__},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)

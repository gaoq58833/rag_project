"""FastAPI 应用：文档解析与分块接口

启动：
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
或：
  python -m src.api.main
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    AuthResponse,
    ChatRequest,
    ChunkResponse,
    Conversation,
    ConversationListResponse,
    FormatsResponse,
    HealthResponse,
    IngestResponse,
    LoginRequest,
    Message,
    MessageListResponse,
    ParseResponse,
    RegisterRequest,
    RetrieveResponse,
    UserInfo,
)
from src.auth import create_access_token, hash_password, verify_password
from src.auth.deps import get_current_user
from src.db import db
from src.db.repository import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
    UserRepository,
)
from src.parsers.chunker import ChunkStrategy, TextChunker
from src.parsers.factory import get_parser, supported_extensions
from src.rag import chat as rag_chat, chat_stream as rag_chat_stream, clear_chat_cache, process_ingest, retrieve as rag_retrieve, submit_ingest
from src.vectorstore import vector_store


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


# 静态文件目录（前端页面）
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """根路径返回前端 index.html"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "RAG 文档解析 API",
        "version": "0.1.0",
        "docs": "/docs",
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
    summary="入库：上传文档立即返回，解析/向量化后台异步执行",
)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """上传文档：内容未重复时立即返回 document_id（status=processing），
    解析/分块/向量化在后台执行，前端轮询 /documents 看状态；内容重复则同步复用秒回。需登录。"""
    filename = file.filename or "unnamed"
    file_bytes = await file.read()

    try:
        result = submit_ingest(file_bytes, filename, user_id=current_user["id"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"入库失败：{e}")

    # 命中复用的已同步完成；新文档交给后台任务处理
    if result["status"] == "processing":
        background_tasks.add_task(
            process_ingest, result["document_id"], file_bytes, filename, current_user["id"]
        )
    return IngestResponse(**result)


@app.get("/documents", summary="获取当前用户的已入库文档列表")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """只返回当前用户上传的文档（用户隔离）。需登录。"""
    rows = DocumentRepository.list_by_user(current_user["id"])
    return {
        "documents": [
            {
                "id": r["id"],
                "filename": r["filename"],
                "file_ext": r["file_ext"],
                "chunk_count": r["chunk_count"],
                "total_pages": r["total_pages"],
                "status": r.get("status", "completed"),
                "error_msg": r.get("error_msg"),
                "uploaded_at": r["uploaded_at"].isoformat() if r.get("uploaded_at") else None,
            }
            for r in rows
        ]
    }


@app.delete("/documents/{document_id}", summary="删除文档及其向量")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """删除文档：同时清除 Qdrant 向量和 PostgreSQL 记录（chunks 级联删除）。需登录。"""
    doc = DocumentRepository.get_by_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    # 用户隔离校验：只能删自己的文档
    if str(doc.get("user_id") or "") != current_user["id"]:
        raise HTTPException(status_code=403, detail="无权删除他人文档")

    # 取出该文档所有 chunk 的 qdrant_point_id
    chunks = ChunkRepository.get_by_document_id(document_id)
    point_ids = [str(c["qdrant_point_id"]) for c in chunks if c.get("qdrant_point_id")]

    # 仅当这些向量不被其他文档引用时，才从 Qdrant 删除
    if point_ids:
        ref_count = ChunkRepository.count_by_qdrant_ids_excluding_doc(point_ids, document_id)
        if ref_count == 0:
            vector_store.delete_by_ids(point_ids)

    # PG 删除文档（chunks 表 ON DELETE CASCADE 自动清理）
    db.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    clear_chat_cache()  # 文档变化，清空问答缓存
    return {"ok": True, "deleted_chunks": len(chunks)}


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
        results = rag_retrieve(query, user_id=current_user["id"], top_k=top_k, use_rerank=use_rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")

    return RetrieveResponse(query=query, results=results)


@app.post(
    "/chat",
    summary="问答：检索 → 构造上下文（含多轮历史）→ LLM 生成回答",
)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """基于知识库的问答，自动保存多轮对话历史。

    - 不传 `conversation_id` 则新建对话
    - 传 `conversation_id` 则追加到已有对话，并带入最近 N 轮上下文
    """
    try:
        result = rag_chat(
            query=request.query,
            user_id=current_user["id"],
            conversation_id=request.conversation_id,
            use_rerank=request.use_rerank,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    return result


@app.post(
    "/chat/stream",
    summary="流式问答：SSE 逐 token 返回",
)
async def chat_stream_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """流式问答接口，返回 SSE 事件流。

    事件格式（每行 `data: {json}\\n\\n`）：
      - {"type": "start", "conversation_id": "...", "sources": [...]}
      - {"type": "token", "content": "..."}  (多次)
      - {"type": "done"}
    """
    import json

    def event_stream():
        try:
            for event in rag_chat_stream(
                query=request.query,
                user_id=current_user["id"],
                conversation_id=request.conversation_id,
                use_rerank=request.use_rerank,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'问答失败：{e}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===== 多轮对话接口 =====

@app.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="获取当前用户的对话列表",
)
async def list_conversations(current_user: dict = Depends(get_current_user)):
    rows = ConversationRepository.list_by_user(current_user["id"])
    return ConversationListResponse(
        conversations=[Conversation(**r) for r in rows]
    )


@app.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageListResponse,
    summary="获取指定对话的全部消息",
)
async def list_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    conv = ConversationRepository.get_by_id(conversation_id)
    if conv is None or str(conv["user_id"]) != current_user["id"]:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    rows = MessageRepository.list_by_conversation(conversation_id)
    return MessageListResponse(messages=[Message(**r) for r in rows])


@app.delete(
    "/conversations/{conversation_id}",
    summary="删除对话及其全部消息",
)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):
    conv = ConversationRepository.get_by_id(conversation_id)
    if conv is None or str(conv["user_id"]) != current_user["id"]:
        raise HTTPException(status_code=404, detail="对话不存在或无权访问")
    ConversationRepository.delete(conversation_id)
    return {"ok": True}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": exc.__class__.__name__},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)

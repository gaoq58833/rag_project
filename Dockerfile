# RAG 文档解析服务 —— CPU 版
# GPU 版见 Dockerfile.gpu

# ---------- 构建阶段：安装依赖 ----------
FROM python:3.12-slim AS builder

# 安装 uv（包管理器）
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /bin/

WORKDIR /app

# 把 pyproject 里定义的 pytorch-cu126 索引重定向到 CPU 索引
# 避免下载 8GB+ 的 CUDA 版 torch
ENV UV_INDEX_PYTORCH_CU126_URL=https://download.pytorch.org/whl/cpu \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 只拷贝依赖声明，利用 Docker 缓存层
COPY pyproject.toml uv.lock ./

# 安装依赖（不安装项目本身，src 布局通过 PYTHONPATH 访问）
# 不用 --frozen：uv.lock 是 CUDA 版 torch，这里重定向到 CPU 索引后需重新解析
# 写入国内可用 DNS（8.8.8.8 在容器内不通）
RUN printf "nameserver 120.196.165.24\nnameserver 223.5.5.5\n" > /etc/resolv.conf \
    && uv sync --no-install-project

# ---------- 运行阶段 ----------
FROM python:3.12-slim AS runtime

# docling / torch / pillow 需要的系统库
RUN printf "nameserver 120.196.165.24\nnameserver 223.5.5.5\n" > /etc/resolv.conf \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段拷贝已装好的虚拟环境和源码
COPY --from=builder /app/.venv /app/.venv
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

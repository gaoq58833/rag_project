-- RAG 业务库表结构
-- 执行：psql -U postgres -d rag -f schema.sql
-- 注意：本文件在单个事务内执行，顺序必须满足依赖关系——
--       被外键引用的表先建，ALTER 补列必须在引用该列的索引之前。

-- 用户表：JWT 鉴权用（documents/conversations 外键引用，须最先建）
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user',  -- user / admin
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 迁移：为已有 users 表补头像列（幂等）
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- 文档表：每个上传文件一条记录
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,  -- 上传者，用户隔离
    filename        TEXT NOT NULL,
    file_ext        TEXT NOT NULL,
    content_hash    VARCHAR(64),    -- 文件内容 SHA-256，用于去重
    total_pages     INTEGER NOT NULL DEFAULT 0,
    table_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    raw_text        TEXT,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status          TEXT NOT NULL DEFAULT 'completed',  -- processing / completed / failed
    error_msg       TEXT            -- 入库失败原因（status=failed 时）
);

-- 迁移：为已有 documents 表补列（幂等；新库这些列已内联，ALTER 为 no-op）
ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE SET NULL;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error_msg TEXT;
-- LLM Wiki：文件原始字节数（判断小文档）+ 是否参与 Wiki 生成（小文档默认 true，大文档默认 false，用户可手动切换）
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size BIGINT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS wiki_eligible BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents (filename);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents (uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);

-- 分块表：每个 chunk 一条记录，通过 qdrant_point_id 关联向量库
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    qdrant_point_id UUID NOT NULL,          -- Qdrant 中的 point id，用于向量检索后回查
    chunk_index     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    length          INTEGER NOT NULL,
    chunk_type      TEXT NOT NULL DEFAULT 'text',  -- text / table / code
    section         TEXT,
    images          JSONB NOT NULL DEFAULT '[]'::jsonb,
    tsv             TSVECTOR,       -- jieba 分词后的全文索引（BM25 风格关键词检索）
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 迁移：为已有 chunks 表补 tsv 列（必须在 idx_chunks_tsv 索引之前）
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv TSVECTOR;

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_point_id ON chunks (qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks (section);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
-- 对 images JSONB 建 GIN 索引，方便按图片路径查询
CREATE INDEX IF NOT EXISTS idx_chunks_images_gin ON chunks USING GIN (images);

-- 对话表：每个用户的一段对话
CREATE TABLE IF NOT EXISTS conversations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title         TEXT NOT NULL DEFAULT '新对话',   -- 首条消息自动生成标题
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id, updated_at DESC);

-- 消息表：对话中的每一条消息（user / assistant）
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,                   -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    sources         JSONB NOT NULL DEFAULT '[]'::jsonb,  -- 仅 assistant 有，引用来源
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_time
    ON messages (conversation_id, created_at ASC);

-- ===== LLM Wiki =====
-- Wiki 页面表：LLM 基于 chunks 自动生成的结构化知识页面，与原始文档解耦（文档删除不强制级联，按 source_chunk_ids 标记失效）
CREATE TABLE IF NOT EXISTS wiki_pages (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,  -- 用户隔离
    slug             TEXT NOT NULL,          -- URL 友好标识，user_id 内唯一（中文标题直接做 slug）
    title            TEXT NOT NULL,
    summary          TEXT,                   -- 一句话定位
    content_md       TEXT NOT NULL,          -- 正文 Markdown，含 [[页面名]] 互链
    paragraphs       JSONB NOT NULL DEFAULT '[]'::jsonb,   -- [{text, chunk_ids[]}]，逐段溯源
    outgoing_links   JSONB NOT NULL DEFAULT '[]'::jsonb,   -- 指向其他页面的 slug 列表
    source_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,   -- 本页依据的所有 chunk id（qdrant_point_id 字符串）
    status           TEXT NOT NULL DEFAULT 'ready',  -- generating / ready / failed / stale
    error_msg        TEXT,
    tsv              TSVECTOR,               -- jieba 分词全文索引（Wiki 页面 BM25 检索）
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_wiki_user_slug ON wiki_pages (user_id, slug);
CREATE INDEX IF NOT EXISTS idx_wiki_user ON wiki_pages (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wiki_status ON wiki_pages (user_id, status);
CREATE INDEX IF NOT EXISTS idx_wiki_tsv ON wiki_pages USING GIN (tsv);
-- 按被引用的 chunk id 反查页面（文档删除时定位受影响页面）
CREATE INDEX IF NOT EXISTS idx_wiki_source_chunks ON wiki_pages USING GIN (source_chunk_ids);

-- Wiki 生成任务表：记录异步生成进度，前端轮询用（类比 documents.status）
CREATE TABLE IF NOT EXISTS wiki_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'generating',  -- generating / done / failed
    page_total  INTEGER NOT NULL DEFAULT 0,
    page_done   INTEGER NOT NULL DEFAULT 0,
    error_msg   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wiki_jobs_user ON wiki_jobs (user_id, created_at DESC);

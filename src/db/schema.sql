-- RAG 业务库表结构
-- 执行：psql -U postgres -d rag -f schema.sql

-- 文档表：每个上传文件一条记录
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT NOT NULL,
    file_ext        TEXT NOT NULL,
    total_pages     INTEGER NOT NULL DEFAULT 0,
    table_count     INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    raw_text        TEXT,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status          TEXT NOT NULL DEFAULT 'completed'  -- processing / completed / failed
);

CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents (filename);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents (uploaded_at DESC);

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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_qdrant_point_id ON chunks (qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks (section);
-- 对 images JSONB 建 GIN 索引，方便按图片路径查询
CREATE INDEX IF NOT EXISTS idx_chunks_images_gin ON chunks USING GIN (images);

-- Schema for the pgvector-backed store.
--
-- One table carries both retrieval paths: an ivfflat index over the embedding
-- for dense search, a GIN index over a generated tsvector for lexical search.
-- Keeping them in one table is what makes the two retrievers transactionally
-- consistent -- a document is visible to both or to neither, never to one.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Match the dimensionality of whatever embedder you configure.
    embedding   VECTOR(256),
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_tsv_idx
    ON documents USING GIN (content_tsv);

-- ivfflat needs rows present before it can pick sensible centroids: build it
-- after the first bulk load, not before. lists ~= sqrt(row_count) is a
-- reasonable starting point.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS documents_metadata_idx
    ON documents USING GIN (metadata jsonb_path_ops);

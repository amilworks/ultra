-- Postgres schema for shared literature cache and derived paper analysis artifacts.
--
-- This schema is intentionally separate from orchestration tables so paper retrieval
-- can be reused across users and runs without coupling to run/event retention.

BEGIN;

CREATE SCHEMA IF NOT EXISTS literature;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS literature.source_rate_limits (
    source TEXT PRIMARY KEY,
    min_interval_seconds INTEGER NOT NULL CHECK (min_interval_seconds > 0),
    next_allowed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO literature.source_rate_limits (source, min_interval_seconds)
VALUES
    ('arxiv', 3),
    ('biorxiv', 1),
    ('medrxiv', 1)
ON CONFLICT (source) DO NOTHING;

CREATE TABLE IF NOT EXISTS literature.papers (
    paper_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL CHECK (source IN ('arxiv', 'biorxiv', 'medrxiv')),
    source_key TEXT NOT NULL,
    doi TEXT,
    doi_normalized TEXT,
    title TEXT NOT NULL,
    abstract TEXT,
    authors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    categories_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    links_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    version TEXT,
    license TEXT,
    published_at TIMESTAMPTZ,
    updated_at_source TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    refresh_after TIMESTAMPTZ NOT NULL,
    stale_after TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', COALESCE(title, '')), 'A')
        || setweight(to_tsvector('simple', COALESCE(abstract, '')), 'B')
    ) STORED,
    UNIQUE (source, source_key)
);

CREATE TABLE IF NOT EXISTS literature.paper_aliases (
    alias_id BIGSERIAL PRIMARY KEY,
    paper_id UUID NOT NULL REFERENCES literature.papers(paper_id) ON DELETE CASCADE,
    alias_type TEXT NOT NULL CHECK (alias_type IN ('doi', 'arxiv_id', 'biorxiv_doi', 'medrxiv_doi', 'title_slug', 'url')),
    alias_value TEXT NOT NULL,
    alias_value_normalized TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (alias_type, alias_value_normalized)
);

CREATE TABLE IF NOT EXISTS literature.paper_assets (
    asset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID NOT NULL REFERENCES literature.papers(paper_id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('pdf', 'jats_xml', 'source_json')),
    storage_uri TEXT NOT NULL,
    sha256 TEXT,
    mime_type TEXT,
    size_bytes BIGINT,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count BIGINT NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (paper_id, asset_type)
);

CREATE TABLE IF NOT EXISTS literature.derivations (
    derivation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    derivation_type TEXT NOT NULL CHECK (derivation_type IN ('summary', 'comparison', 'innovation_map', 'method_matrix', 'qa_bundle')),
    input_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    output_json JSONB NOT NULL,
    evidence_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    refresh_after TIMESTAMPTZ NOT NULL,
    stale_after TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count BIGINT NOT NULL DEFAULT 0,
    UNIQUE (derivation_type, input_hash, prompt_version, model_name)
);

CREATE TABLE IF NOT EXISTS literature.fetch_log (
    fetch_id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL CHECK (source IN ('arxiv', 'biorxiv', 'medrxiv')),
    request_key TEXT NOT NULL,
    cache_status TEXT NOT NULL CHECK (cache_status IN ('hit_fresh', 'hit_stale', 'miss')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    latency_ms INTEGER,
    status_code INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_text TEXT,
    response_bytes BIGINT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE FUNCTION literature._touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_literature_papers_updated_at ON literature.papers;
CREATE TRIGGER trg_literature_papers_updated_at
BEFORE UPDATE ON literature.papers
FOR EACH ROW
EXECUTE FUNCTION literature._touch_updated_at();

CREATE INDEX IF NOT EXISTS idx_literature_papers_doi_normalized
    ON literature.papers (doi_normalized);
CREATE INDEX IF NOT EXISTS idx_literature_papers_refresh_after
    ON literature.papers (refresh_after);
CREATE INDEX IF NOT EXISTS idx_literature_papers_stale_after
    ON literature.papers (stale_after);
CREATE INDEX IF NOT EXISTS idx_literature_papers_last_accessed
    ON literature.papers (last_accessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_literature_papers_search_vector
    ON literature.papers USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_literature_papers_title_trgm
    ON literature.papers USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_literature_papers_abstract_trgm
    ON literature.papers USING GIN (abstract gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_literature_aliases_paper_id
    ON literature.paper_aliases (paper_id);

CREATE INDEX IF NOT EXISTS idx_literature_assets_paper_id
    ON literature.paper_assets (paper_id);
CREATE INDEX IF NOT EXISTS idx_literature_assets_last_accessed
    ON literature.paper_assets (last_accessed_at DESC);

CREATE INDEX IF NOT EXISTS idx_literature_derivations_refresh_after
    ON literature.derivations (refresh_after);
CREATE INDEX IF NOT EXISTS idx_literature_derivations_stale_after
    ON literature.derivations (stale_after);
CREATE INDEX IF NOT EXISTS idx_literature_derivations_last_accessed
    ON literature.derivations (last_accessed_at DESC);

CREATE INDEX IF NOT EXISTS idx_literature_fetch_log_source_started
    ON literature.fetch_log (source, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_literature_fetch_log_request_key
    ON literature.fetch_log (request_key, started_at DESC);

COMMIT;

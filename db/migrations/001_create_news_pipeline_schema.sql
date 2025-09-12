-- Migration: Create schema for NFL News Pipeline
-- Tables: news_urls, source_watermarks, filter_decisions, pipeline_audit_log

create extension if not exists pgcrypto;

create table if not exists news_urls (
    id uuid primary key default gen_random_uuid(),
    url text unique not null,
    title text not null,
    description text,
    publication_date timestamptz not null,
    source_name text not null,
    publisher text not null,
    relevance_score double precision not null default 0,
    filter_method text not null,
    filter_reasoning text,
    entities jsonb,
    categories text[],
    raw_metadata jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_news_urls_publication_date on news_urls(publication_date);
create index if not exists idx_news_urls_source_name on news_urls(source_name);
create index if not exists idx_news_urls_relevance_score on news_urls(relevance_score);
create index if not exists idx_news_urls_entities on news_urls using gin((coalesce(entities, '{}'::jsonb)));

create table if not exists source_watermarks (
    source_name text primary key,
    last_processed_date timestamptz not null,
    last_successful_run timestamptz not null default now(),
    items_processed integer default 0,
    errors_count integer default 0,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists pipeline_audit_log (
    id uuid primary key default gen_random_uuid(),
    pipeline_run_id uuid,
    source_name text,
    event_type text not null, -- 'fetch','filter','store','error'
    event_data jsonb,
    message text,
    created_at timestamptz default now()
);

create index if not exists idx_audit_log_pipeline_run on pipeline_audit_log(pipeline_run_id);
create index if not exists idx_audit_log_event_type on pipeline_audit_log(event_type);

-- Dedicated table for filter decisions, linked to news_urls for analytics
create table if not exists filter_decisions (
    id uuid primary key default gen_random_uuid(),
    news_url_id uuid not null references news_urls(id) on delete cascade,
    method text not null, -- 'rule_based' or 'llm'
    stage text not null,  -- 'rule' or 'llm'
    confidence double precision not null,
    reasoning text,
    model_id text,
    created_at timestamptz default now()
);

create index if not exists idx_filter_decisions_news_url on filter_decisions(news_url_id);
create index if not exists idx_filter_decisions_method on filter_decisions(method);

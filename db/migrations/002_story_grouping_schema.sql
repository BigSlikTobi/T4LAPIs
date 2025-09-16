-- Migration: Story Similarity Grouping Schema
-- Adds vector extension, embeddings, groups, group members, and context summaries
-- Depends on: 001_create_news_pipeline_schema.sql (news_urls table)

-- Enable required extensions
create extension if not exists pgcrypto; -- for gen_random_uuid()
create extension if not exists vector;   -- pgvector for similarity search

-- Table: context_summaries
-- Stores LLM-generated summaries and metadata for each news URL
create table if not exists context_summaries (
    id uuid primary key default gen_random_uuid(),
    news_url_id uuid not null references news_urls(id) on delete cascade,
    summary_text text not null,
    entities jsonb,
    key_topics text[],
    llm_model text not null,
    confidence_score double precision not null default 0,
    fallback_used boolean not null default false,
    generated_at timestamptz default now(),
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    unique(news_url_id)
);

create index if not exists idx_context_summaries_news_url on context_summaries(news_url_id);
create index if not exists idx_context_summaries_model on context_summaries(llm_model);
create index if not exists idx_context_summaries_entities on context_summaries using gin((coalesce(entities, '{}'::jsonb)));

-- Table: story_embeddings
-- Stores semantic embeddings for stories. Dimension set for OpenAI text-embedding-3-small (1536)
create table if not exists story_embeddings (
    id uuid primary key default gen_random_uuid(),
    news_url_id uuid not null references news_urls(id) on delete cascade,
    embedding_vector vector(1536),
    model_name text not null,
    model_version text not null,
    summary_text text not null,
    confidence_score double precision not null default 0,
    generated_at timestamptz default now(),
    created_at timestamptz default now()
);

create index if not exists idx_story_embeddings_news_url on story_embeddings(news_url_id);
create index if not exists idx_story_embeddings_model on story_embeddings(model_name, model_version);
-- Vector similarity index for efficient similarity search (cosine distance)
create index if not exists idx_story_embeddings_vector on story_embeddings
using ivfflat (embedding_vector vector_cosine_ops) with (lists = 100);

-- Table: story_groups
-- Represents clusters of similar stories with centroid embedding
create table if not exists story_groups (
    id uuid primary key default gen_random_uuid(),
    centroid_embedding vector(1536),
    member_count integer not null default 1,
    status text not null default 'new', -- 'new' | 'updated' | 'stable'
    tags text[] default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_story_groups_status on story_groups(status);
create index if not exists idx_story_groups_member_count on story_groups(member_count);
create index if not exists idx_story_groups_updated on story_groups(updated_at);
-- Vector index for centroid similarity search (cosine distance)
create index if not exists idx_story_groups_centroid on story_groups
using ivfflat (centroid_embedding vector_cosine_ops) with (lists = 50);

-- Table: story_group_members
-- Links news URLs to groups with their similarity scores at time of addition
create table if not exists story_group_members (
    id uuid primary key default gen_random_uuid(),
    group_id uuid not null references story_groups(id) on delete cascade,
    news_url_id uuid not null references news_urls(id) on delete cascade,
    similarity_score double precision not null,
    added_at timestamptz default now(),
    unique(group_id, news_url_id)
);

create index if not exists idx_group_members_group on story_group_members(group_id);
create index if not exists idx_group_members_news_url on story_group_members(news_url_id);
create index if not exists idx_group_members_similarity on story_group_members(similarity_score);

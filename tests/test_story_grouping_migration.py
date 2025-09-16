"""
Lightweight validation that the story grouping migration defines required tables
and indexes without executing against a real database. This checks for typos
and ensures compatibility with the initial schema naming.

Note: Full DB integration tests should run in an environment with Postgres +
pgvector and execute the SQL migrations. Here we only verify the migration file
contains key statements.
"""
from pathlib import Path
import re

MIGRATION_PATH = Path(__file__).resolve().parents[1] / "db" / "migrations" / "002_story_grouping_schema.sql"


def read_sql() -> str:
    data = MIGRATION_PATH.read_text(encoding="utf-8")
    # collapse whitespace for simpler regex checks
    return re.sub(r"\s+", " ", data).strip().lower()


def test_migration_file_exists():
    assert MIGRATION_PATH.exists(), "Migration 002_story_grouping_schema.sql should exist"


def test_required_extensions_declared():
    sql = read_sql()
    assert "create extension if not exists vector" in sql
    assert "create extension if not exists pgcrypto" in sql


def test_tables_defined():
    sql = read_sql()
    for tbl in [
        "context_summaries",
        "story_embeddings",
        "story_groups",
        "story_group_members",
    ]:
        assert f"create table if not exists {tbl}" in sql, f"missing table {tbl}"


def test_foreign_keys_reference_news_urls():
    sql = read_sql()
    assert "news_url_id uuid not null references news_urls(id) on delete cascade" in sql


def test_vector_columns_and_indexes_present():
    sql = read_sql()
    # Check vector columns
    assert "embedding_vector vector(1536)" in sql
    assert "centroid_embedding vector(1536)" in sql
    # Check IVFFlat indexes with cosine ops
    assert "using ivfflat (embedding_vector vector_cosine_ops)" in sql
    assert "using ivfflat (centroid_embedding vector_cosine_ops)" in sql


def test_uniqueness_and_indexes():
    sql = read_sql()
    assert "unique(news_url_id)" in sql  # for context_summaries
    # Common helpful indexes
    assert "idx_story_embeddings_news_url" in sql
    assert "idx_story_groups_status" in sql
    assert "idx_group_members_group" in sql

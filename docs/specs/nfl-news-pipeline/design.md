# Design Document

## Architecture Overview

The NFL News Processing Pipeline is composed of modular components that work together to ingest, filter, and store NFL-related news URLs and metadata. The system is designed for scalability, auditability, and compliance, with a strong emphasis on efficient incremental processing and clear separation of concerns.

### High-Level Architecture

- Source Fetchers: Retrieve URLs and metadata from configured sources in feeds.yaml (RSS feeds and sitemaps)
- Filtering Layer: Applies rule-based and LLM-assisted relevance checks on URL metadata
- Deduplication & Watermarking: Prevents duplicate entries and supports incremental ingestion
- Storage Layer: Persists URLs, metadata, and audit logs in Supabase
- Monitoring: Tracks performance, errors, and processing statistics

## Components

### 1. Source Fetchers

- RSSFetcher: Processes RSS feed entries to extract URLs, titles, publication dates, and descriptions
- SitemapFetcher: Constructs sitemap URLs using templates and extracts story URLs and metadata
- Configuration: Sources are defined under feeds.yaml with properties like url_template, days_back, max_articles, timeout_seconds, user_agent, and nfl_only

### 2. Filtering Layer

- RuleBasedFilter: Uses heuristic rules based on titles, URLs, and other metadata
- LLMFilter: Uses an LLM to evaluate NFL relevance based on URLs, titles, and descriptions
- Decision Logging: All filtering decisions are captured with scores, methods (rule/llm), and rationale

### 3. Deduplication & Watermarking

- Deduplicator: Ensures that URLs are unique across all sources
- WatermarkManager: Tracks last processed dates per source to enable incremental processing

### 4. Storage Layer

- Supabase storage for persistent data
- Schema includes tables for news URLs, source watermarks, and audit logs
- Foreign keys link audit decisions to specific URL entries for traceability

### 5. Monitoring

- Application logs capture start/end times, counts, errors, and decision details
- Dashboard or log aggregation system for real-time monitoring and analytics

## Data Model

### news_urls

- id (UUID)
- source (text)
- publisher (text)
- url (text, unique)
- title (text)
- published_at (timestamp)
- description (text)
- relevance_score (numeric)
- created_at (timestamp)
- updated_at (timestamp)

### source_watermarks

- id (UUID)
- source (text, unique)
- last_processed_at (timestamp)
- updated_at (timestamp)

### pipeline_audit_log

- id (UUID)
- news_url_id (UUID, fk -> news_urls.id)
- stage (enum: rule, llm)
- method (text)
- score (numeric)
- rationale (text)
- model_id (text)
- created_at (timestamp)

## Sequence Flows

### Ingestion Flow

1. Load configuration from feeds.yaml
2. For each source, fetch URLs and metadata
3. Apply rule-based filtering; if inconclusive, apply LLM filtering
4. Log decisions and store results in Supabase
5. Update watermarks on success

### Error Handling

- Retry failed operations with exponential backoff
- Preserve watermarks on failure to allow safe reruns
- Comprehensive error logging with context

## Testing Strategy

- Unit tests for each component (fetchers, filters, storage, watermark manager)
- Integration tests simulating end-to-end ingestion using sample feeds
- Performance tests for high-volume feed processing
- Monitoring verification tests for log outputs and metrics

## Performance Considerations

- Configurable max_parallel_fetches for controlled concurrency
- Efficient indexing on URLs and timestamps for fast queries
- Incremental processing with watermarks minimizes redundant work

## Security & Compliance

- Only process publicly available feeds and sitemaps
- Do not fetch or store full article content
- Respect timeouts, user agents, and rate limits
- Secure database credentials and rotate API keys

## Deployment & Operations

- Run via CLI scripts with dry-run mode for safe testing
- Docker support for containerized deployment
- GitHub Actions for scheduling automated runs
- Observability via logs and optional metrics dashboards

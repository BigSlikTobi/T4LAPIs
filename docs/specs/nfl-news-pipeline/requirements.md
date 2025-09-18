# Requirements Document

## Introduction

The NFL News Processing Pipeline is a comprehensive data ingestion system that automatically collects, filters, and stores NFL-related news URLs and metadata from multiple public sources. The system leverages RSS feeds and sitemaps defined in feeds.yaml to gather news URLs and available metadata only, applies intelligent filtering using both rule-based and LLM-assisted relevance checks, and maintains clean, deduplicated data in Supabase with incremental ingestion capabilities.

The pipeline emphasizes transparency, auditability, and respect for publisher terms of service by extracting only URLs and publicly available metadata from feeds and sitemaps - never scraping or crawling full article content.

## Requirements

### Requirement 1

**User Story:** As a data consumer, I want the system to automatically ingest NFL news URLs and metadata from multiple reliable sources, so that I have comprehensive coverage of NFL-related content references.

#### Acceptance Criteria

1. WHEN the pipeline runs THEN the system SHALL fetch news URLs and metadata from all enabled RSS feeds defined in feeds.yaml
2. WHEN the pipeline runs THEN the system SHALL fetch news URLs and metadata from all enabled sitemap sources defined in feeds.yaml
3. WHEN processing sitemap sources THEN the system SHALL construct URLs using the url_template with current UTC year and zero-padded month
4. WHEN processing sitemap sources THEN the system SHALL extract only URLs and available metadata (titles, dates, descriptions) without downloading full article content
5. WHEN processing sitemap sources THEN the system SHALL respect the max_articles and days_back constraints
6. IF a source has nfl_only=true THEN the system SHALL skip some filtering steps for performance optimization
7. WHEN fetching from any source THEN the system SHALL respect the configured timeout_seconds and user_agent settings

### Requirement 2

**User Story:** As a system administrator, I want the pipeline to filter news URLs for NFL relevance using both rules and AI, so that only relevant content references are stored and processed.

#### Acceptance Criteria

1. WHEN processing news URLs THEN the system SHALL apply rule-based filtering using titles, URLs, and available metadata
2. WHEN rule-based filtering is inconclusive THEN the system SHALL use LLM-assisted relevance checks
3. WHEN using LLM checks THEN the system SHALL analyze only URLs, titles, descriptions, and other available metadata (never full article content)
4. WHEN making filtering decisions THEN the system SHALL log all decisions with relevance scores
5. WHEN filtering content THEN the system SHALL preserve source information, dates, and model outputs for auditability
6. IF a URL fails NFL relevance checks THEN the system SHALL exclude it from storage but log the decision

### Requirement 3

**User Story:** As a data integrity manager, I want the system to prevent duplicate URLs and support incremental updates, so that the database remains clean and efficient.

#### Acceptance Criteria

1. WHEN ingesting news URLs THEN the system SHALL deduplicate by URL across all sources
2. WHEN processing sources THEN the system SHALL maintain per-source watermarks for incremental ingestion
3. WHEN a duplicate URL is detected THEN the system SHALL update metadata if newer but not create duplicate entries
4. WHEN the pipeline runs THEN the system SHALL only process URLs newer than the last successful watermark
5. WHEN ingestion completes successfully THEN the system SHALL update the watermark for that source
6. IF ingestion fails THEN the system SHALL preserve the previous watermark to enable retry

### Requirement 4

**User Story:** As a database administrator, I want all news URLs and metadata stored in Supabase with proper structure and indexing, so that they can be efficiently queried and analyzed.

#### Acceptance Criteria

1. WHEN storing news URLs THEN the system SHALL save them to Supabase with structured schema
2. WHEN storing URLs THEN the system SHALL include source, publisher, title, URL, publication date, description, and relevance score
3. WHEN storing filtering decisions THEN the system SHALL maintain audit trail of all processing decisions
4. WHEN creating database entries THEN the system SHALL ensure proper indexing for efficient queries
5. WHEN database operations fail THEN the system SHALL implement retry logic with exponential backoff
6. WHEN storing data THEN the system SHALL validate schema compliance before insertion
7. WHEN storing filtering decisions THEN the system SHALL save them in a dedicated Supabase table (e.g., `filter_decisions` or `pipeline_audit_log`) linked to each news URL row via foreign key
8. WHEN saving filtering decisions THEN the system SHALL include method (rule/llm), stage (rule or llm), confidence score, reasoning text, model identifier/version, and created_at timestamp to support quality metrics and analysis

### Requirement 5

**User Story:** As a system operator, I want comprehensive logging and monitoring capabilities, so that I can track pipeline performance and troubleshoot issues.

#### Acceptance Criteria

1. WHEN the pipeline runs THEN the system SHALL log start time, end time, and processing statistics
2. WHEN processing each source THEN the system SHALL log fetch attempts, success/failure rates, and item counts
3. WHEN filtering decisions are made THEN the system SHALL log the decision rationale and confidence scores
4. WHEN errors occur THEN the system SHALL log detailed error information with context
5. WHEN the pipeline completes THEN the system SHALL generate summary statistics for the run
6. IF critical errors occur THEN the system SHALL implement appropriate alerting mechanisms

### Requirement 6

**User Story:** As a compliance officer, I want the system to respect publisher terms of service and rate limits, so that we maintain ethical data collection practices.

#### Acceptance Criteria

1. WHEN fetching from sources THEN the system SHALL respect the max_parallel_fetches limit
2. WHEN making requests THEN the system SHALL use the configured user_agent string
3. WHEN accessing publisher content THEN the system SHALL only extract URLs and metadata from public feeds and sitemaps
4. WHEN processing sources THEN the system SHALL NOT download, scrape, or crawl full article content
5. WHEN rate limits are encountered THEN the system SHALL implement appropriate backoff strategies
6. WHEN sources are disabled THEN the system SHALL skip them entirely during processing

### Requirement 7

**User Story:** As a data analyst, I want the processed URL metadata to enable clustering and entity tracking, so that I can perform advanced analytics on NFL news trends.

#### Acceptance Criteria

1. WHEN storing news URLs THEN the system SHALL extract and normalize key entities from titles and descriptions (teams, players, etc.)
2. WHEN processing metadata THEN the system SHALL maintain consistent entity references across sources
3. WHEN storing data THEN the system SHALL include metadata that supports clustering algorithms
4. WHEN processing URLs THEN the system SHALL tag content with relevant categories and topics based on available metadata
5. WHEN entities are identified THEN the system SHALL link them to existing entity databases if available
6. WHEN storing processed data THEN the system SHALL maintain relationships between related news URLs

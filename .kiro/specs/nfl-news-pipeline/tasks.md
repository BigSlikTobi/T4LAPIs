# Implementation Plan

-
  1. [x] Set up project structure and core data models
  - Create directory structure for the NFL news pipeline components
  - Define core data models (NewsItem, ProcessedNewsItem, SourceWatermark,
    FilterResult)
  - Create base configuration classes for feeds.yaml parsing
  - _Requirements: 1.1, 1.6, 4.2_

-
  2. [x] Implement configuration management system
  - [x] 2.1 Create feeds.yaml parser and validator
    - Write ConfigManager class to parse feeds.yaml structure
    - Implement validation for required fields and data types
    - Add support for url_template construction with date placeholders
    - _Requirements: 1.1, 1.3, 1.5_

  - [x] 2.2 Add configuration loading and error handling
    - Implement configuration file loading with proper error messages
    - Add validation for enabled/disabled sources filtering
    - Create unit tests for configuration parsing edge cases
    - _Requirements: 1.1, 1.5, 5.4_

-
  3. [x] Build RSS feed processing capabilities
  - [x] 3.1 Create RSS processor with metadata extraction
    - Implement RSSProcessor class using feedparser library
    - Extract title, description, URL, and publication date from RSS entries
    - Handle various RSS feed formats and missing fields gracefully
    - _Requirements: 1.1, 1.6, 4.2_

  - [x] 3.2 Add RSS feed fetching with rate limiting
    - Implement HTTP requests with configured user agent and timeouts
    - Add retry logic with exponential backoff for network failures
    - Respect max_parallel_fetches configuration for concurrent processing
    - _Requirements: 1.6, 6.1, 6.2, 6.5_

  - [x] 3.3 Create RSS processor unit tests
    - Write tests with mock RSS feed responses
    - Test parsing of various RSS feed formats
    - Validate error handling for malformed feeds
    - _Requirements: 1.1, 5.4_

-
  4. [x] Build sitemap processing capabilities
  - [x] 4.1 Create sitemap processor with URL construction
    - Implement SitemapProcessor class for XML sitemap parsing
    - Add dynamic URL construction using url_template with current date
    - Extract URLs and metadata from sitemap entries
    - _Requirements: 1.2, 1.3, 1.4, 4.2_

  - [x] 4.2 Add date filtering and article limits
    - Implement days_back filtering to process only recent articles
    - Add max_articles limit enforcement for sitemap sources
    - Handle timezone conversion for publication date comparisons
    - _Requirements: 1.3, 1.4_

  - [x] 4.3 Create sitemap processor unit tests
    - Write tests with mock sitemap XML responses
    - Test URL template construction with various date formats
    - Validate date filtering and article limit enforcement
    - _Requirements: 1.2, 1.3, 1.4_

-
  5. [x] Implement NFL relevance filtering system
  - [x] 5.1 Create rule-based filter with NFL keywords
    - Implement RuleBasedFilter class with NFL team names and keywords
    - Add URL pattern matching for obvious NFL content
    - Create confidence scoring based on keyword matches and patterns
  - Persist each filtering decision (rule or LLM) to Supabase for metrics (FK to news URL entry)
  - _Requirements: 2.1, 2.4, 2.5, 4.3, 4.7_

  - [x] 5.2 Build LLM-assisted filtering for ambiguous cases
    - Implement LLMFilter class using existing LLM setup
    - Create prompts for NFL relevance analysis using only metadata
    - Add logic to determine when LLM filtering is needed
  - Persist LLM outputs (score, reasoning, model id) to Supabase linked to the article
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 4.3, 4.7_

  - [x] 5.3 Create comprehensive filter testing
    - Write unit tests with known NFL and non-NFL content examples
    - Test rule-based filtering accuracy and confidence scoring
    - Mock LLM responses for consistent testing of AI filtering
  - _Requirements: 2.1, 2.2, 2.4, 2.5, 4.7_

-
  6. [ ] Build database schema and storage layer
  - [ ] 6.1 Create Supabase database schema
    - Write SQL migration scripts for news_urls, source_watermarks, and
      pipeline_audit_log tables (including filter decisions with FK to news_urls)
    - Add proper indexes for efficient querying by date, source, and relevance
      score
    - Implement database initialization and migration logic
  - _Requirements: 4.1, 4.2, 4.4, 4.7_

  - [ ] 6.2 Implement storage manager with deduplication
    - Create StorageManager class using existing Supabase client
    - Implement URL-based deduplication logic across all sources
    - Add batch insert operations for efficient database writes
    - _Requirements: 3.1, 3.3, 4.1, 4.5_

  - [ ] 6.3 Add watermark management for incremental processing
    - Implement watermark storage and retrieval for each source
    - Add logic to update watermarks only on successful processing
    - Create watermark-based filtering to process only new items
    - _Requirements: 3.2, 3.4, 3.5, 3.6_

  - [ ] 6.4 Create storage layer unit tests
    - Write tests for deduplication logic with mock database responses
    - Test watermark management and incremental processing
    - Validate batch insert operations and error handling
    - _Requirements: 3.1, 3.2, 4.5_

-
  7. [ ] Implement comprehensive logging and audit system
  - [ ] 7.1 Create audit logging for all pipeline decisions
    - Implement audit log entries for fetch, filter, and store operations
    - Log all filtering decisions with scores and reasoning
    - Add structured logging for pipeline statistics and performance metrics
    - _Requirements: 2.4, 2.5, 5.1, 5.2, 5.3_

  - [ ] 7.2 Add error handling and monitoring capabilities
    - Implement error categorization and appropriate retry strategies
    - Add detailed error logging with context and stack traces
    - Create pipeline summary reports with success/failure statistics
    - _Requirements: 5.4, 5.6_

  - [ ] 7.3 Create logging system unit tests
    - Write tests for audit log entry creation and storage
    - Test error handling and retry logic with various failure scenarios
    - Validate pipeline reporting and statistics generation
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

-
  8. [ ] Build pipeline orchestrator and main execution logic
  - [ ] 8.1 Create main pipeline orchestrator
    - Implement NFLNewsPipeline class to coordinate all components
    - Add source processing logic with parallel execution support
    - Integrate all components (config, processors, filters, storage)
    - _Requirements: 1.1, 1.2, 6.1, 6.2_

  - [ ] 8.2 Add error handling and recovery mechanisms
    - Implement graceful error handling that continues processing other sources
    - Add pipeline-level retry logic and failure recovery
    - Create comprehensive error reporting and alerting
    - _Requirements: 5.4, 5.6, 6.5_

  - [ ] 8.3 Create end-to-end pipeline tests
    - Write integration tests using test feeds and mock responses
    - Test complete pipeline flow from configuration to storage
    - Validate error handling and recovery in various failure scenarios
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 4.1_

-
  9. [ ] Implement entity extraction and categorization
  - [ ] 9.1 Add entity extraction from news metadata
    - Create entity extraction logic for NFL teams, players, and topics
    - Implement entity normalization and linking to existing databases
    - Add category tagging based on extracted entities and content analysis
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ] 9.2 Create entity processing unit tests
    - Write tests for entity extraction accuracy with known content
    - Test entity normalization and database linking
    - Validate category tagging and relationship detection
    - _Requirements: 7.1, 7.2, 7.6_

-
  10. [ ] Add performance optimization and monitoring
  - [ ] 10.1 Implement caching and performance optimizations
    - Add LLM response caching to reduce API costs and latency
    - Implement connection pooling for database operations
    - Add memory management for large feed processing
    - _Requirements: 6.1, 6.2_

  - [ ] 10.2 Create performance monitoring and metrics
    - Implement metrics collection for processing times and success rates
    - Add performance reporting and bottleneck identification
    - Create monitoring dashboards for pipeline health
    - _Requirements: 5.1, 5.2, 5.3_

-
  11. [ ] Create CLI interface and deployment scripts
  - [ ] 11.1 Build command-line interface for pipeline execution
    - Create CLI script for manual pipeline execution and testing
    - Add options for single-source processing and dry-run mode
    - Implement configuration validation and status reporting commands
    - _Requirements: 5.1, 5.5_

  - [ ] 11.2 Add deployment and scheduling configuration
    - Create deployment scripts and configuration files
    - Add scheduling configuration for automated pipeline runs
    - Implement health check endpoints and monitoring integration
    - _Requirements: 5.5, 5.6_

-
  12. [ ] Final integration and comprehensive testing
  - [ ] 12.1 Perform end-to-end system testing
    - Test complete pipeline with real feeds in controlled environment
    - Validate all requirements are met with comprehensive test scenarios
    - Perform load testing with multiple sources and large data volumes
    - _Requirements: All requirements_

  - [ ] 12.2 Create documentation and deployment guide
    - Write comprehensive documentation for pipeline configuration and usage
    - Create troubleshooting guide and operational procedures
    - Add deployment instructions and monitoring setup guide
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

# Requirements Document

## Introduction

The Story Similarity Grouping system clusters related news stories about the same ongoing events (e.g., trades, injuries, contracts) by using URL-context LLM summaries and vector embeddings. The goal is to group similar stories, maintain group lifecycle states, and enable analytics and personalized content generation based on these groups.

## Requirements

### Requirement 1: Context Extraction

- The system SHALL extract contextual metadata from news URLs using LLMs without fetching full content
- The system SHALL store structured context summaries in the database
- The system SHALL track model identifiers and versioning for reproducibility

### Requirement 2: Embedding Generation

- The system SHALL generate vector embeddings for each context summary
- The system SHALL normalize embeddings and store them efficiently
- The system SHALL support configurable embedding models

### Requirement 3: Similarity Calculation

- The system SHALL compute cosine similarity between embeddings
- The system SHALL support threshold-based grouping with configurable margins
- The system SHALL allow centroid updates as groups evolve

### Requirement 4: Group Management

- The system SHALL assign new items to existing groups or create new groups when needed
- The system SHALL maintain group lifecycle states (created, active, stabilizing, archived)
- The system SHALL handle edge cases for borderline similarity and large groups

### Requirement 5: Storage & Indexing

- The system SHALL store embeddings and group data in appropriate tables
- The system SHALL use indexes for fast searches and updates
- The system SHALL ensure data integrity across related tables

### Requirement 6: Incremental Processing

- The system SHALL process new items incrementally based on timestamps or watermarks
- The system SHALL support backfilling for historical data
- The system SHALL ensure idempotency in assignment operations

### Requirement 7: Monitoring & Analytics

- The system SHALL log grouping decisions and metrics
- The system SHALL expose summaries for evaluating group quality
- The system SHALL monitor drift and embedding model performance

### Requirement 8: Security & Compliance

- The system SHALL process only public metadata and URLs
- The system SHALL not scrape full article content
- The system SHALL secure API keys and sensitive configurations

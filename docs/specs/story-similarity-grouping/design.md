# Design Document

## Architecture Overview

The Story Similarity Grouping system clusters related NFL news stories using LLM-extracted context and vector similarity. Components are modular to support evolution of models and algorithms without impacting storage and monitoring.

### High-Level Architecture

- URL Context Extractor: Summarizes URLs with LLMs using only public metadata
- Embedding Generator: Produces normalized embeddings for each context summary
- Similarity Calculator: Computes cosine similarity across embeddings
- Group Manager: Assigns items to groups and updates centroids
- Storage Manager: Persists embeddings, groups, and memberships
- Monitoring: Tracks metrics, decisions, and drift

## Components and Interfaces

### URLContextExtractor

- Input: URL, title, metadata
- Output: context_summary (text), model_id, created_at
- Notes: No full content fetching; metadata-only summaries

### EmbeddingGenerator

- Input: context_summary, model
- Output: vector_embedding (array[float])
- Notes: Model configurable via environment or config

### SimilarityCalculator

- Input: vector_embedding, centroid_embeddings
- Output: similarity_scores (array[float])
- Notes: Cosine similarity with thresholding

### GroupManager

- Input: embedding, group centroids, thresholds
- Output: group_id assignment or new group creation
- Notes: Updates centroid and state transitions

### StorageManager

- Input/Output: CRUD for story_embeddings, story_groups, story_group_members, context_summaries
- Notes: Uses indexes and foreign keys for joins

## Data Model

### context_summaries

- id (UUID)
- news_url_id (UUID)
- summary_text (text)
- model_id (text)
- created_at (timestamp)

### story_embeddings

- id (UUID)
- context_summary_id (UUID, fk -> context_summaries.id)
- embedding (vector)
- model_id (text)
- created_at (timestamp)

### story_groups

- id (UUID)
- title (text)
- centroid_embedding (vector)
- state (enum: created, active, stabilizing, archived)
- created_at (timestamp)
- updated_at (timestamp)

### story_group_members

- id (UUID)
- group_id (UUID, fk -> story_groups.id)
- story_embedding_id (UUID, fk -> story_embeddings.id)
- assigned_at (timestamp)

## Processing Flows

1. New URL metadata arrives
2. Create context summary (metadata-only)
3. Generate embedding
4. Compare to group centroids
5. Assign to best group if above threshold, else create new group
6. Update centroid and group state as needed

## Testing Strategy

- Unit tests for each component
- Integration tests for end-to-end grouping
- Backfill tests for historical runs
- Metrics validation for grouping quality

## Performance & Scaling

- Vector index on embeddings for similarity search
- Batch processing with configurable sizes
- Incremental runs with watermarks
- Asynchronous workers for embedding generation

## Security & Compliance

- Public metadata only (no scraping)
- Secure model keys and rotate regularly
- Access control for database tables

# Story Grouping Integration Guide

This guide explains how to use the story grouping functionality that has been integrated into the NFL News Pipeline as part of Task 12.

## Overview

The story grouping feature automatically clusters similar news stories based on semantic similarity. It integrates seamlessly with the existing pipeline as a post-processing step, running after news items are stored in the database.

## Configuration

### Enable Story Grouping

Story grouping can be enabled in two ways:

#### 1. Configuration File (feeds.yaml)

Add the following to your `feeds.yaml` file:

```yaml
defaults:
  user_agent: "T4LAPIs-NFLNewsPipeline/1.0"
  timeout_seconds: 15
  max_parallel_fetches: 4
  
  # Story grouping configuration
  enable_story_grouping: true
  story_grouping_max_parallelism: 4
  story_grouping_max_stories_per_run: 100
  story_grouping_reprocess_existing: false

sources:
  - name: "espn_nfl"
    type: "rss"
    url: "https://www.espn.com/espn/rss/nfl/news"
    enabled: true
    publisher: "ESPN"
    nfl_only: true
```

#### 2. Environment Variable

Set the environment variable to enable story grouping:

```bash
export NEWS_PIPELINE_ENABLE_STORY_GROUPING=1
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `enable_story_grouping` | Enable/disable story grouping | `false` |
| `story_grouping_max_parallelism` | Maximum parallel processing threads | `4` |
| `story_grouping_max_stories_per_run` | Maximum stories to process per run | `None` (unlimited) |
| `story_grouping_reprocess_existing` | Reprocess stories that already have embeddings | `false` |

## CLI Commands

The pipeline now includes four new CLI commands for story grouping:

### 1. Run Pipeline with Story Grouping

```bash
# Run the standard pipeline with story grouping enabled
python scripts/pipeline_cli.py run --enable-story-grouping

# With additional options
python scripts/pipeline_cli.py run --enable-story-grouping --dry-run --source espn
```

### 2. Manual Story Grouping

Process recent stories for grouping without running the full pipeline:

```bash
# Basic usage
python scripts/pipeline_cli.py group-stories

# With options
python scripts/pipeline_cli.py group-stories \
  --max-stories 50 \
  --max-parallelism 2 \
  --dry-run \
  --reprocess
```

**Options:**
- `--max-stories N`: Maximum number of stories to process (default: 100)
- `--max-parallelism N`: Maximum parallel processing threads (default: 4)
- `--dry-run`: Show what would be done without making changes
- `--reprocess`: Reprocess stories that already have embeddings

### 3. Story Grouping Status

Check the current status and statistics of story grouping:

```bash
python scripts/pipeline_cli.py group-status
```

This command shows:
- Database connectivity status
- Total stories with embeddings
- Total story groups
- Average group size
- Recent grouping activity

### 4. Batch Backfill

Process existing stories in batches for story grouping:

```bash
# Basic backfill
python scripts/pipeline_cli.py group-backfill

# With options
python scripts/pipeline_cli.py group-backfill \
  --batch-size 25 \
  --max-batches 10 \
  --dry-run \
  --resume-from story_id_123
```

**Options:**
- `--batch-size N`: Stories per batch (default: 50)
- `--max-batches N`: Maximum batches to process (default: unlimited)
- `--dry-run`: Show what would be processed without making changes
- `--resume-from ID`: Resume from specific story ID

### 5. Analytics Report

Generate analytics reports for story grouping performance:

```bash
# Text format report
python scripts/pipeline_cli.py group-report

# JSON format report
python scripts/pipeline_cli.py group-report \
  --format json \
  --days-back 14
```

**Options:**
- `--format FORMAT`: Output format (`text` or `json`, default: `text`)
- `--days-back N`: Number of days to include in report (default: 7)

## Standalone Batch Processor

For large-scale backfilling operations, use the standalone batch processor:

```bash
# Basic usage
python scripts/story_grouping_batch_processor.py --dry-run

# Advanced usage with resume capability
python scripts/story_grouping_batch_processor.py \
  --batch-size 25 \
  --max-batches 100 \
  --progress-file backfill_progress.json \
  --resume-from story_id_456 \
  --verbose
```

**Features:**
- **Progress Tracking**: Save progress to a file for resuming interrupted operations
- **Resume Capability**: Continue from where you left off
- **Batch Processing**: Efficient chunked processing of large datasets
- **Verbose Logging**: Detailed logging for monitoring progress

## Integration with Existing Pipeline

### How It Works

1. **Normal Pipeline Execution**: The pipeline runs normally, fetching, filtering, and storing news items
2. **Post-Processing Hook**: After items are successfully stored, story grouping runs automatically (if enabled)
3. **Story Processing**: Each new story is:
   - Analyzed for context using LLM URL context extraction
   - Converted to semantic embeddings
   - Compared against existing story groups
   - Assigned to the best matching group or creates a new group
4. **Group Management**: Story groups are maintained with centroids and metadata

### Performance Considerations

- **Incremental Processing**: Only new stories are processed, not existing ones (unless reprocess is enabled)
- **Centroid-First Similarity**: Initial similarity comparison uses group centroids for efficiency
- **Parallel Processing**: Configurable parallelism for embedding generation and similarity calculations
- **Resource Limits**: Configurable limits on processing scope and time

### Error Handling

- **Graceful Degradation**: Pipeline continues even if story grouping fails
- **Comprehensive Logging**: All errors are logged through the existing audit system
- **Retry Mechanisms**: Built-in retry logic for transient failures
- **Circuit Breakers**: Prevents cascading failures during high error rates

## Database Schema

Story grouping extends the existing database schema with these tables:

- `story_embeddings`: Stores semantic embeddings for each story
- `story_groups`: Manages story group metadata and centroids
- `story_group_members`: Links stories to their groups
- `context_summaries`: Caches LLM-generated context summaries

All tables maintain foreign key relationships with the existing `news_urls` table.

## Requirements Compliance

This implementation satisfies the requirements specified in Task 12:

### Requirement 6.1: Incremental Processing
- ✅ Only computes similarities against existing group centroids initially
- ✅ Performs detailed analysis only with candidate groups
- ✅ Implements parallel processing for embedding generation

### Requirement 6.5: Tools for Reprocessing
- ✅ Provides CLI commands for manual story grouping
- ✅ Includes batch processing tools for existing story backfill
- ✅ Supports resume capability for large operations

## Examples

### Complete Workflow Example

```bash
# 1. Run pipeline with story grouping for new stories
python scripts/pipeline_cli.py run --enable-story-grouping

# 2. Check status
python scripts/pipeline_cli.py group-status

# 3. Backfill existing stories in small batches
python scripts/pipeline_cli.py group-backfill --batch-size 25 --max-batches 5

# 4. Generate weekly analytics report
python scripts/pipeline_cli.py group-report --days-back 7

# 5. Process specific recent stories manually
python scripts/pipeline_cli.py group-stories --max-stories 20 --reprocess
```

### Configuration for Production

```yaml
defaults:
  # Standard pipeline settings
  user_agent: "T4LAPIs-NFLNewsPipeline/1.0"
  timeout_seconds: 30
  max_parallel_fetches: 6
  
  # Story grouping optimized for production
  enable_story_grouping: true
  story_grouping_max_parallelism: 6
  story_grouping_max_stories_per_run: 200
  story_grouping_reprocess_existing: false

sources:
  # Your production sources here
```

### Environment Variables for CI/CD

```bash
# Enable story grouping via environment
export NEWS_PIPELINE_ENABLE_STORY_GROUPING=1

# Disable for testing environments
export NEWS_PIPELINE_ENABLE_STORY_GROUPING=0
```

## Troubleshooting

### Common Issues

1. **Story grouping not running**: Check that it's enabled in configuration or environment variable
2. **Import errors**: Ensure all story grouping dependencies are installed
3. **Performance issues**: Reduce `story_grouping_max_parallelism` or `max_stories_per_run`
4. **Database errors**: Check database connectivity and schema migrations

### Debug Mode

Enable debug output to monitor story grouping operations:

```bash
export NEWS_PIPELINE_DEBUG=1
python scripts/pipeline_cli.py run --enable-story-grouping
```

This will show detailed information about:
- Story grouping initialization
- Number of items being processed
- Processing results and timing
- Error conditions

## Future Enhancements

Potential future improvements to the story grouping integration:

1. **Real-time Processing**: WebSocket-based real-time story grouping
2. **Advanced Analytics**: Machine learning-based group quality metrics
3. **User Interface**: Web-based dashboard for group management
4. **API Integration**: REST API endpoints for external systems
5. **Performance Optimization**: GPU-accelerated similarity calculations
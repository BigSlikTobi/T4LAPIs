# Story Grouping Configuration and Deployment Guide

This guide covers the configuration and deployment setup for the NFL News Pipeline Story Similarity Grouping feature.

## Overview

The Story Grouping feature extends the NFL News Pipeline to automatically group similar news stories using embedding-based similarity analysis. This document provides comprehensive guidance for configuring and deploying this feature.

## Configuration Management

### Configuration File Structure

The story grouping feature uses a dedicated YAML configuration file (`story_grouping_config.yaml`) with the following sections:

```yaml
# LLM Configuration for URL context extraction
llm:
  provider: "openai"  # or "google", "deepseek"
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"
  max_tokens: 500
  temperature: 0.1
  timeout_seconds: 30
  max_retries: 3

# Embedding Configuration
embedding:
  model_name: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  cache_ttl_hours: 24
  normalize_vectors: true

# Similarity Calculation Configuration
similarity:
  threshold: 0.8
  metric: "cosine"
  max_candidates: 100
  search_limit: 1000
  candidate_similarity_floor: 0.35

# Group Management Configuration
grouping:
  max_group_size: 50
  centroid_update_threshold: 5
  status_transition_hours: 24
  auto_tagging: true
  max_stories_per_run: null
  prioritize_recent: true
  prioritize_high_relevance: true
  reprocess_existing: false

# Performance Configuration
performance:
  parallel_processing: true
  max_workers: 4
  max_parallelism: 4
  batch_size: 10
  cache_size: 1000
  max_total_processing_time: null

# Database Configuration
database:
  connection_timeout_seconds: 30
  max_connections: 10
  batch_insert_size: 100
  vector_index_maintenance: true

# Monitoring and Logging Configuration
monitoring:
  log_level: "INFO"
  metrics_enabled: true
  cost_tracking_enabled: true
  performance_alerts_enabled: true
  group_quality_monitoring: true
```

### Configuration Sections Explained

#### LLM Configuration
- **provider**: LLM provider (openai, google, deepseek)
- **model**: Specific model to use for context extraction
- **api_key_env**: Environment variable name containing the API key
- **max_tokens**: Maximum tokens for LLM responses
- **temperature**: LLM temperature for context generation
- **timeout_seconds**: Request timeout for LLM calls
- **max_retries**: Number of retry attempts for failed requests

#### Embedding Configuration
- **model_name**: Sentence transformer model for embeddings
- **dimension**: Expected embedding vector dimension
- **batch_size**: Batch size for embedding generation
- **cache_ttl_hours**: Cache time-to-live for embeddings
- **normalize_vectors**: Whether to normalize embedding vectors

#### Similarity Configuration
- **threshold**: Minimum similarity score for grouping
- **metric**: Distance metric (cosine, euclidean, dot_product)
- **max_candidates**: Maximum candidate groups to consider
- **search_limit**: Maximum stories to search for similarity
- **candidate_similarity_floor**: Minimum similarity for candidate filtering

#### Grouping Configuration
- **max_group_size**: Maximum stories per group
- **centroid_update_threshold**: Stories needed to trigger centroid update
- **status_transition_hours**: Hours before group status transitions
- **auto_tagging**: Enable automatic group tagging
- **max_stories_per_run**: Limit stories processed per run
- **prioritize_recent**: Process recent stories first
- **prioritize_high_relevance**: Process high-relevance stories first
- **reprocess_existing**: Whether to reprocess already grouped stories

#### Performance Configuration
- **parallel_processing**: Enable parallel processing
- **max_workers**: Maximum worker threads
- **max_parallelism**: Maximum parallel operations
- **batch_size**: Batch size for operations
- **cache_size**: Maximum cache entries
- **max_total_processing_time**: Total processing time limit (seconds)

#### Database Configuration
- **connection_timeout_seconds**: Database connection timeout
- **max_connections**: Maximum database connections
- **batch_insert_size**: Batch size for database inserts
- **vector_index_maintenance**: Enable vector index maintenance

#### Monitoring Configuration
- **log_level**: Logging level (DEBUG, INFO, WARNING, ERROR)
- **metrics_enabled**: Enable metrics collection
- **cost_tracking_enabled**: Enable cost tracking
- **performance_alerts_enabled**: Enable performance alerts
- **group_quality_monitoring**: Enable group quality monitoring

## Environment Variables

### Required Variables

```bash
# Database Configuration (Required)
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your-anon-key

# LLM API Key (Choose one based on your provider)
OPENAI_API_KEY=your-openai-api-key
# OR
GOOGLE_API_KEY=your-google-api-key  
# OR
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### Optional Variables

```bash
# Story Grouping Feature Control
STORY_GROUPING_ENABLED=true
STORY_GROUPING_DRY_RUN=false
STORY_GROUPING_CONFIG_PATH=story_grouping_config.yaml

# Performance and Logging
LOG_LEVEL=INFO
DATABASE_TIMEOUT=30
LLM_TIMEOUT=30
MAX_PARALLEL_WORKERS=4

# Development Settings
DEBUG=true
```

### Environment Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file with your configuration:**
   ```bash
   # Required: Supabase database credentials
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-supabase-key
   
   # Required: LLM API key (choose your provider)
   OPENAI_API_KEY=sk-your-openai-key
   ```

3. **Validate environment setup:**
   ```bash
   python scripts/deploy_story_grouping.py validate
   ```

## Deployment Scripts

### Bash Deployment Script

The `scripts/deploy_story_grouping.sh` script provides comprehensive deployment validation:

```bash
# Validate deployment environment
./scripts/deploy_story_grouping.sh --dry-run

# Deploy to production environment
./scripts/deploy_story_grouping.sh --environment=prod

# Use custom configuration
./scripts/deploy_story_grouping.sh --config-path=custom_config.yaml
```

**Features:**
- Python environment validation
- Dependency checking
- Environment variable validation
- Configuration file validation
- Database connectivity testing
- Story grouping functionality testing

### Python Deployment Script

The `scripts/deploy_story_grouping.py` script provides detailed Python-based deployment:

```bash
# Run full validation
python scripts/deploy_story_grouping.py validate --verbose

# Interactive setup wizard
python scripts/deploy_story_grouping.py setup

# Test specific functionality
python scripts/deploy_story_grouping.py test

# Generate configuration file
python scripts/deploy_story_grouping.py config --config-path=my_config.yaml
```

**Commands:**
- `validate`: Complete deployment validation
- `setup`: Interactive setup wizard
- `config`: Configuration file generation and validation
- `test`: Story grouping functionality testing
- `deploy`: Full deployment validation and setup

## Deployment Process

### 1. Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Supabase database access
- LLM API access (OpenAI, Google, or DeepSeek)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/BigSlikTobi/T4LAPIs.git
cd T4LAPIs

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration Setup

```bash
# Copy configuration template
cp story_grouping_config.yaml my_story_config.yaml

# Copy environment template
cp .env.example .env

# Edit configuration files with your settings
# .env - database and API credentials
# my_story_config.yaml - story grouping parameters
```

### 4. Validation and Testing

```bash
# Run comprehensive validation
python scripts/deploy_story_grouping.py validate --verbose

# Test specific components
python scripts/deploy_story_grouping.py test

# Run interactive setup if needed
python scripts/deploy_story_grouping.py setup
```

### 5. Database Migration

Ensure story grouping database schema is set up:

```bash
# Run story grouping migration (if not already done)
python scripts/setup_story_grouping_db.py
```

### 6. Deployment Verification

```bash
# Final deployment validation
./scripts/deploy_story_grouping.sh

# Test story grouping end-to-end
python scripts/story_grouping_dry_run.py --config my_story_config.yaml
```

## Configuration Best Practices

### Production Configuration

For production deployments:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"  # Cost-effective and reliable
  timeout_seconds: 30
  max_retries: 3

embedding:
  model_name: "all-MiniLM-L6-v2"  # Good balance of speed and quality
  batch_size: 32
  cache_ttl_hours: 24

similarity:
  threshold: 0.8  # High threshold for precision
  max_candidates: 50  # Limit for performance
  candidate_similarity_floor: 0.4

performance:
  max_workers: 4  # Adjust based on available resources
  max_total_processing_time: 1800  # 30 minutes limit
  
monitoring:
  log_level: "INFO"
  metrics_enabled: true
  cost_tracking_enabled: true
```

### Development Configuration

For development and testing:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  timeout_seconds: 15
  max_retries: 2

similarity:
  threshold: 0.7  # Lower threshold for more grouping
  max_candidates: 20

grouping:
  max_stories_per_run: 100  # Limit for testing

performance:
  max_workers: 2
  max_total_processing_time: 600  # 10 minutes

monitoring:
  log_level: "DEBUG"
```

### Environment-Specific Configurations

Create separate configuration files for different environments:

- `story_grouping_config.dev.yaml` - Development
- `story_grouping_config.staging.yaml` - Staging  
- `story_grouping_config.prod.yaml` - Production

Use the `--config-path` option to specify the appropriate configuration:

```bash
# Development
python scripts/story_grouping_dry_run.py --config story_grouping_config.dev.yaml

# Production
python scripts/story_grouping_dry_run.py --config story_grouping_config.prod.yaml
```

## Monitoring and Maintenance

### Configuration Validation

Regularly validate your configuration:

```bash
# Validate current configuration
python scripts/deploy_story_grouping.py validate

# Check environment variables
python scripts/deploy_story_grouping.py validate --verbose
```

### Performance Monitoring

Monitor key metrics:
- Story processing throughput
- Group formation rates
- LLM API usage and costs
- Database performance
- Error rates and types

### Configuration Updates

When updating configuration:

1. Test changes in development environment
2. Validate configuration: `python scripts/deploy_story_grouping.py config`
3. Run validation: `python scripts/deploy_story_grouping.py validate`
4. Deploy gradually to staging then production

## Troubleshooting

### Common Issues

**Configuration validation fails:**
- Check YAML syntax
- Verify all required fields are present
- Validate parameter ranges and types

**Environment variable issues:**
- Verify .env file exists and is loaded
- Check variable names match configuration
- Ensure API keys are valid and have proper permissions

**Database connectivity issues:**
- Verify Supabase credentials
- Check network connectivity
- Ensure story grouping tables exist

**LLM API issues:**
- Verify API key is valid and active
- Check API usage limits and quotas
- Ensure proper provider configuration

### Getting Help

- Check logs for detailed error information
- Run validation with `--verbose` flag for detailed output
- Use the interactive setup wizard for guided configuration
- Review configuration file documentation and examples

## Integration with Existing Pipeline

The story grouping feature integrates seamlessly with the existing NFL news pipeline:

1. **Pipeline Integration**: Story grouping runs as a post-processing step after news ingestion
2. **Configuration Compatibility**: Uses existing configuration patterns and database connections
3. **Monitoring Integration**: Leverages existing logging and monitoring infrastructure
4. **CLI Integration**: Available through existing CLI tools and commands

For complete integration details, see the main pipeline documentation and the Technical Details guide.
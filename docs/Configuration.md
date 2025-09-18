# Configuration Guide

This guide explains how to use the two configuration layers in this repo.

## 1) feeds.yaml (root)

Purpose
- Defines news sources for the pipeline (RSS, sitemaps, or HTML pages)
- Supports top-level defaults (user agent, timeouts, parallel fetches)
- May include story-grouping toggles under `defaults`

Usage
```
python scripts/pipeline_cli.py run --config feeds.yaml --dry-run
```

Grouping toggles in feeds.yaml
```
defaults:
  enable_story_grouping: true
  story_grouping_max_parallelism: 4
  story_grouping_max_stories_per_run: 100
```

Example file
- See `docs/examples/feeds_with_story_grouping.yaml` for a complete, ready-to-copy example.

## 2) story_grouping_config.yaml (root)

Purpose
- Detailed settings for story similarity grouping (LLM provider/model, embeddings, thresholds, performance, monitoring, env vars)

Used by
- Grouping-aware scripts and the pipeline when grouping is enabled (via feeds.yaml or environment variable)

Environment variables
- Required: `SUPABASE_URL`, `SUPABASE_KEY`
- Optional (provider-specific): `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`

Run examples
```
# Pipeline with grouping (dry run)
python scripts/pipeline_cli.py run --config feeds.yaml --enable-story-grouping --dry-run

# Group-only operations (examples)
python scripts/pipeline_cli.py group-stories --max-stories 50 --dry-run
python scripts/pipeline_cli.py group-status
```

Troubleshooting
- Use `--dry-run` to avoid DB writes when experimenting
- Check `docs/Troubleshooting.md` for common issues

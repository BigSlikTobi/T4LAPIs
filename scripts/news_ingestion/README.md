# News Ingestion

Purpose
- Fetch and filter NFL news from configured RSS/Sitemaps using the orchestrated pipeline.

Key Scripts
- `pipeline_cli.py` — main entrypoint to run, validate, and inspect the pipeline.
- `news_fetch_demo.py` — quick demo to list and optionally filter items without DB writes.

Usage
- List sources: `python scripts/news_ingestion/pipeline_cli.py list-sources --config feeds.yaml`
- Dry-run pipeline: `python scripts/news_ingestion/pipeline_cli.py run --config feeds.yaml --dry-run --disable-llm`
- Demo fetch: `python scripts/news_ingestion/news_fetch_demo.py --config feeds.yaml --show 5 --filter`

Guidance
- Config lives at `feeds.yaml`. Keep publishers and defaults up to date.
- Use `--dry-run` to explore without DB writes; set `OPENAI_API_KEY` to enable LLM filters.
- Environment toggles: `NEWS_PIPELINE_ONLY_SOURCE`, `NEWS_PIPELINE_DEBUG`, `NEWS_PIPELINE_DISABLE_ENTITY_DICT`.

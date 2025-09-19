# News Ingestion

The news pipeline decomposed into discrete stages so every step can be optimised and debugged independently or stitched back together for an end-to-end run. Every stage has a dedicated CLI plus a top-level shim under `scripts/` for convenience.

## Stage-by-Stage CLIs

| Stage | CLI | Typical Output |
| --- | --- | --- |
| Source fetch | `python scripts/fetch_sources_cli.py --config feeds.yaml --use-watermarks --output fetched.json` | Raw RSS/Sitemap payload (`stories` + `sources`) |
| Context summaries | `python scripts/context_extraction_cli.py --input fetched.json --output contexts.json --mock` | `stories[*].summary_text` |
| Embeddings | `python scripts/embedding_cli.py --input contexts.json --output embeddings.json --mock` | `stories[*].embedding` |
| Similarity pairs | `python scripts/similarity_cli.py --input embeddings.json --output similarities.json --threshold 0.6` | `pairs[*].{story_id_a, story_id_b, similarity}` |
| Group assignment | `python scripts/grouping_cli.py --similarities similarities.json --embeddings embeddings.json --output groups.json --threshold 0.6` | `groups[*].members` + optional centroids |
| Full pipeline | `python scripts/full_story_pipeline_cli.py --config feeds.yaml` | Ingestion + context + embeddings + grouping (Supabase aware) |

All stage CLIs accept `--output` to persist structured JSON. You can pipe artefacts directly from one command to the next or inspect/edit the intermediate files as part of local optimisation. The `--mock` switches on the context/embedding commands generate deterministic placeholders so you can test workflows without hitting OpenAI/Google APIs.

### Fetch Stage (`fetch_sources_cli.py`)
- Respects Supabase watermarks when `--use-watermarks` is provided (requires Supabase credentials) but also works fully offline.
- Additional flags: `--source` to target a single feed, `--ignore-watermark` to bypass stored state, `--limit` for per-source caps.
- The output JSON always includes a flattened `stories` array with `story_id`, making downstream tooling straightforward.
- Use `--write-supabase` to push fetched items directly into `news_urls` (the CLI will synthesise minimal `ProcessedNewsItem` stubs for you).

### Context Stage (`context_extraction_cli.py`)
- Two modes:
  - **Offline:** `--input fetched.json [--output contexts.json] [--mock]`
  - **Pipeline-backed:** omit `--input` to re-use Supabase + the ingestion pipeline (supports `--dry-run`, `--disable-llm`, `--ignore-watermark`).
- Stores summaries in Supabase when run via the pipeline and not in dry-run mode; otherwise writes to JSON. Add `--write-supabase` to persist even in dry-run/offline flows.
- `--from-supabase` lets you export existing summaries (optionally filtered by `--story-ids` or `--limit`) without running extraction.
- Set `URL_CONTEXT_PROVIDER=google` if you want the extractor to prefer Gemini instead of OpenAI (falls back automatically when a provider isn’t available).

### Embedding Stage (`embedding_cli.py`)
- Consumes `contexts.json` and produces per-story embeddings.
- Use `--mock` to generate deterministic vectors (hash-based) when API keys are unavailable.
- Without `--mock`, supply `OPENAI_API_KEY` for OpenAI embeddings or the script will fall back to sentence-transformers.
- Flags: `--from-supabase` pulls context summaries straight from Supabase (`--story-ids` and `--limit` help scope the batch); `--write-supabase` persists generated embeddings back to `story_embeddings`.

### Similarity Stage (`similarity_cli.py`)
- Calculates pairwise similarity and keeps matches over `--threshold` (default 0.5). Use `--top-k` to limit emitted pairs.
- Supports the same metrics as the core pipeline (`cosine`, `euclidean`, `dot_product`).
- Flags: `--from-supabase` reads embeddings directly from the database; `--write-supabase` logs the resulting similarity payload to the audit log for downstream monitoring.

### Grouping Stage (`grouping_cli.py`)
- Builds connected components from the similarity pairs (override the cut-off with `--threshold`).
- Optionally supply `--embeddings` to emit centroid vectors per group.
- Flags: `--embeddings-from-supabase` fetches embeddings on-demand for centroid calculation; `--write-supabase` creates groups and memberships in Supabase using the supplied similarity pairs.

## End-to-End Example

```bash
# 1️⃣ Fetch
python scripts/fetch_sources_cli.py --config feeds.yaml --use-watermarks --output fetched.json

# 2️⃣ Context (mocked example)
python scripts/context_extraction_cli.py --input fetched.json --output contexts.json --mock

# 3️⃣ Embeddings
python scripts/embedding_cli.py --input contexts.json --output embeddings.json --mock

# 4️⃣ Similarity + Grouping
python scripts/similarity_cli.py --input embeddings.json --output similarities.json --threshold 0.6
python scripts/grouping_cli.py --similarities similarities.json --embeddings embeddings.json --output groups.json --threshold 0.6
```

Swap stage commands for their Supabase-backed equivalents (e.g., context/embedding via the pipeline) whenever you want the official storage side-effects.

## Legacy & Combined Entrypoints
- `python scripts/news_ingestion/pipeline_cli.py …` remains the canonical driver for ingestion + filtering.
- `python scripts/full_story_pipeline_cli.py …` chains ingestion and story grouping with Supabase writes, honouring defaults and feature flags in `feeds.yaml`.
- `python scripts/news_ingestion/news_fetch_demo.py …` still provides the lightweight demo fetcher.

## Tips & Environment Flags
- Config resides in `feeds.yaml`; defaults there influence both combined and staged runs.
- Useful environment toggles: `NEWS_PIPELINE_ONLY_SOURCE`, `NEWS_PIPELINE_DEBUG`, `NEWS_PIPELINE_DISABLE_ENTITY_DICT`, `NEWS_PIPELINE_DISABLE_STORY_GROUPING`.
- Supply `OPENAI_API_KEY` / `GOOGLE_API_KEY` when running live context or embedding stages.

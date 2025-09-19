# Story Grouping

Purpose
- Extract URL context, embed summaries, calculate similarity, and manage evolving story groups.

Key Scripts
- `story_grouping_cli_demo.py` — CLI showcase and examples.
- `story_grouping_dry_run.py` — end-to-end dry run without DB writes.
- `story_grouping_batch_processor.py` — backfill processor for existing items.
- `live_story_context_test.py` — fetch latest article and extract live context.
- `deploy_story_grouping.*` — deployment helpers.

Usage
- Dry run: `python scripts/story_grouping/story_grouping_dry_run.py --help`
- Batch backfill: `python scripts/story_grouping/story_grouping_batch_processor.py --dry-run --batch-size 25`
- Demo: `python scripts/story_grouping/story_grouping_cli_demo.py`

Guidance
- Configure thresholds in `story_grouping_config.yaml` (root).
- Requires `OPENAI_API_KEY` for context/embeddings; stores to Supabase when not dry-running.

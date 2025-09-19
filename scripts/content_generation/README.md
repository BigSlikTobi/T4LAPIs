# Content Generation

Purpose
- Detect trending entities and generate AI summaries of recent developments.

Key Scripts
- `trending_topic_detector.py` — finds trending players/teams over a lookback period.
- `trending_summary_generator.py` — produces LLM summaries for trending entities.

Usage
- Detect: `python scripts/content_generation/trending_topic_detector.py --hours 24 --top-n 10 --output-format json`
- Summarize: `python scripts/content_generation/trending_summary_generator.py --entity-ids "00-0033873,KC"`

Guidance
- Requires DB connections; LLM summaries need provider keys.
- Pipe detector output into generator with `--from-stdin` when chaining.

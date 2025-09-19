# Ops Utilities

Purpose
- Miscellaneous operational helpers for debugging, docs generation, and functionality checks.

Contents
- `helper_scripts/` — environment checks, data exploration, docs generation.
- `functionality_checks/` — quick manual tests for components (e.g., embeddings).

Usage Examples
- Env debug: `python scripts/helper_scripts/debug_env.py`
- Explore nfl_data_py: `python scripts/helper_scripts/explore_nfl_data.py`
- Embedding check: `python scripts/functionality_checks/test_embedding_generator.py --mode transformer`

Guidance
- These scripts are for local ops workflows; not part of automated pipelines.

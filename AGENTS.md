# Repository Guidelines

## Project Structure & Module Organization
- `src/nfl_news_pipeline/` owns ingestion, filtering, and story-grouping flows; place new pipeline logic here.
- `src/core/` offers shared database, logging, and entity helpers used by CLIs and the API.
- `scripts/` provides operational CLIs (`pipeline_cli.py`, `teams_cli.py`, `games_auto_update.py`) plus automation; mirror their naming.
- `api/` runs the FastAPI service (`python main.py`), while `content_generation/`, `docs/`, and `db/migrations/` hold AI writers, references, and schema updates.
- `tests/` shadows feature directories; add coverage next to the module you change.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt` installs dependencies.
- `python scripts/news_ingestion/pipeline_cli.py list-sources --config feeds.yaml` or `run --dry-run --disable-llm` exercises the news pipeline without writes (add `--batch-size` to throttle LLMs).
- `python scripts/teams_cli.py --dry-run` (and other `*_cli.py`) validates data loaders locally.
- `cd api && uvicorn main:app --reload` starts the REST API at `http://localhost:8000/docs`.
- `python -m pytest -q` runs the suite; use `pytest tests/test_story_grouping_integration.py` for targeted checks.

## Coding Style & Naming Conventions
- Use Python 3.11, 4-space indents, type hints, and dataclasses as seen in `src/nfl_news_pipeline/orchestrator/pipeline.py`.
- Keep modules and functions snake_case, classes PascalCase, and CLI entry points suffixed `_cli.py`.
- Order imports stdlib → third-party → local; reuse `src/core/utils/logging.get_logger` and document non-obvious behavior inline—no formatter is enforced, so match existing spacing.

## Testing Guidelines
- Pytest (async enabled via `pytest.ini`) is standard; follow the `test_<feature>.py` naming and reuse fixtures in `tests/conftest.py`.
- Prefer `--dry-run` flags or mocks to avoid hitting Supabase or LLMs; update integration suites such as `tests/test_story_grouping_integration.py` when orchestrators change.
- Run `python -m pytest -q` before pushing and capture notable command output in PRs when behavior shifts.

## Commit & Pull Request Guidelines
- Start every piece of work by creating a dedicated feature branch; never develop directly on `main`.
- Mirror git history: imperative subject, optional scope, PR number on merge (e.g., `Fix story grouping gating logic (#55)`).
- Keep commits focused and note schema or config implications in the body.
- When the branch is ready, open a detailed pull request summarizing changes, validation steps, and any follow-ups.
- Do not push commits straight to `main`; all changes flow through reviewed pull requests.
- PRs should explain motivation, list validation commands (pytest, CLI dry-runs, API smoke checks), link relevant docs, and attach payload samples when responses change; call out new env vars.

## Configuration & Secrets
- Store Supabase credentials and LLM keys outside the repo; inject via environment variables before running write paths.
- Toggle pipeline behavior with `NEWS_PIPELINE_ENABLE_STORY_GROUPING`, `NEWS_PIPELINE_ONLY_SOURCE`, `NEWS_PIPELINE_DEBUG`, and `NEWS_PIPELINE_DISABLE_ENTITY_DICT`.
- Keep `feeds.yaml` and `story_grouping_config.yaml` synchronized with new sources and update supporting docs under `docs/` when defaults shift.

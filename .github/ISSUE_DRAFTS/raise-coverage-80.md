# Raise test coverage to 80% across core modules

## Background
- Current overall coverage (excluding CLI/demo per .coveragerc): ~71% on Python 3.13 local run.
- Goal: Increase to ≥80% coverage focusing on core packages under `src/`.
- Out of scope: CLI/demo scripts under `scripts/`, `examples`, and any `demo_*` files (already excluded in `.coveragerc`).

## Why this matters
- Higher coverage improves confidence in refactors, CI stability, and contributor velocity.
- We’ve recently stabilized tasks and CI; now is a good time to lock in quality by closing coverage gaps.

## Scope
Focus on the lowest-covered modules first. From the latest report:

- Very low coverage
  - `src/core/llm/llm_setup.py` (~22%)
  - `src/core/data/transform.py` (~30%)
  - `src/core/utils/logging.py` (~39%)
  - `src/nfl_news_pipeline/entities/llm_openai.py` (~48%)
  - `src/nfl_news_pipeline/utils/cache.py` (~59%)
  - `src/nfl_news_pipeline/processors/sitemap.py` (~61%)
  - `src/nfl_news_pipeline/storage/embedding/storage.py` (~64%)
  - `src/nfl_news_pipeline/group_manager.py` (~69%)
  - `src/nfl_news_pipeline/filters/relevance.py` (~69%)
  - `src/nfl_news_pipeline/storage/group_manager.py` (~69%)

- Medium coverage (push over the line)
  - `src/nfl_news_pipeline/processors/rss.py` (~75%)
  - `src/nfl_news_pipeline/story_grouping/cache.py` (~74%)
  - `src/core/data/loaders/player_weekly_stats.py` (~62%)

Note: Numbers are from the most recent local run and may vary slightly in CI (Python 3.11).

## Plan of attack
- Add targeted unit tests for pure functions, edge cases, and error paths.
- Introduce small refactors where needed for testability (inject dependencies, split long functions).
- Add fixtures/mocks for external APIs (Supabase, OpenAI/Gemini/DeepSeek) to cover error handling paths.
- Ensure async code paths are exercised (pytest-asyncio already in place).
- Keep the `.coveragerc` exclusions for CLI/demo code.

## Deliverables
- New/updated tests under `tests/` covering the modules listed above.
- Optional minimal refactors (no behavior changes) to enable testing.
- Coverage report at or above 80% overall: `pytest --cov=src --cov-report=term-missing`.
- All tests green locally (3.13) and in CI (3.11).

## Acceptance criteria
- [ ] Overall coverage ≥80% (excluding excluded paths)
- [ ] No new flakiness; tests pass consistently on CI and locally
- [ ] Critical error/exception paths covered for low-coverage modules
- [ ] No public API/behavior changes without explicit review

## Nice to have
- [ ] Add `coverage html` artifact generation in CI for easy review
- [ ] Incremental CI threshold gate (e.g., 75% → 78% → 80%) to avoid one massive PR

## References
- `.coveragerc` now excludes CLI/demo paths
- `pytest.ini` suppresses noisy warnings for cleaner signal

## Labels
Suggested: `enhancement`, `testing`, `coverage`
# Developer Validation Scripts

This folder contains standalone scripts to validate common scenarios locally without running the full test suite.

Run any script directly with Python:

```bash
python tools/validation/test_configuration_integration.py
python tools/validation/test_simple_integration.py
python tools/validation/test_story_grouping_config.py
python tools/validation/test_fallback_query.py
python tools/validation/test_workflow.py
```

Notes
- These scripts are print-based checks for quick feedback; they are not pytest tests.
- Prefer `pytest` for unit/integration tests: `python -m pytest -q`.
- Some scripts may require environment variables (.env) or a live DB connection.

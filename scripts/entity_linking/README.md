# Entity Linking

Purpose
- Build an entity dictionary and link extracted entities (players/teams) to articles; supports LLM-assisted extraction.

Key Scripts
- `entity_dictionary_cli.py` — build/inspect dictionary.
- `llm_entity_linker.py` and `llm_entity_linker_cli.py` — extract and link via DeepSeek.
- `setup_entity_linking_db.py` — notes and index setup for Supabase.

Usage
- Dictionary: `python scripts/entity_linking/entity_dictionary_cli.py --search "Mahomes"`
- LLM extract test: `python scripts/entity_linking/llm_entity_linker_cli.py test --text "..."`
- Run linking: `python scripts/entity_linking/llm_entity_linker_cli.py run --batch-size 10`

Guidance
- Requires database access. LLM features need provider keys configured.
- See README for manual SQL function note (get_unlinked_articles).

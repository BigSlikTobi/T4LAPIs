 # NFL News Pipeline — Getting Started (Easy Setup)

This guide helps you run the NFL News Pipeline on your machine in a simple way. No database setup is required for a test run.

## 1) Requirements
- Python 3.11 or newer
- macOS, Linux, or Windows
- Git (optional, for cloning)

## 2) Clone and install
1. Clone the repository (or download the zip):
   git clone https://github.com/BigSlikTobi/T4LAPIs.git
   cd T4LAPIs
2. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # On Windows (PowerShell): .venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt

## 3) Quick dry-run (no DB writes)
Try the pipeline end-to-end without touching any database.
- List sources:
  python scripts/pipeline_cli.py list-sources --config feeds.yaml
- Run a single source with dry-run (example: ESPN):
  python scripts/pipeline_cli.py run --config feeds.yaml --source espn --dry-run --disable-llm

You should see a summary with “Dry-run stored items” preview. This confirms fetching, filtering, and processing work locally.

Tips:
- Use --disable-llm to skip AI calls and run only rule-based filtering (faster, cheaper).
- Omit --source to run all enabled sources (may be slower).

## 4) Optional: Enable LLM
If you want the AI filter for ambiguous items:
1. Set your API key in an .env file at the repo root:
   OPENAI_API_KEY=sk-...
   # Optional tuning
   OPENAI_TIMEOUT=10
   NEWS_PIPELINE_LLM_CACHE=1
   NEWS_PIPELINE_LLM_CACHE_TTL=86400
2. Re-run without --disable-llm:
   python scripts/pipeline_cli.py run --config feeds.yaml --source espn --dry-run

## 5) Connect a database (Supabase)
To persist results, configure Supabase and run without --dry-run.
1. Create a Supabase project and get:
   - SUPABASE_URL
   - SUPABASE_KEY (service role preferred for writes)
2. Put them in .env:
   SUPABASE_URL=...
   SUPABASE_KEY=...
3. Run a status check:
   python scripts/pipeline_cli.py status --config feeds.yaml
4. Run the pipeline (writes to DB):
   python scripts/pipeline_cli.py run --config feeds.yaml --source espn

Notes:
- The schema includes news_urls, source_watermarks, and pipeline_audit_log.
- The StorageManager handles deduplication by URL and watermarking by source.

## 6) Scheduling (CI)
We include a GitHub Actions workflow to run the pipeline on a schedule.
- File: .github/workflows/news-pipeline.yml
- What it does: validates, prints status, then runs the pipeline.
- To use it:
  1. In your GitHub repo settings, add these secrets:
     - OPENAI_API_KEY (optional)
     - SUPABASE_URL, SUPABASE_KEY (required for DB writes)
  2. Push the repo to GitHub and the workflow will run on schedule (and via “Run workflow”).

## 7) Troubleshooting
- “extract_content is not permitted by policy” → This is expected. We only use metadata (titles/descriptions/URLs), not full article scraping.
- No sources matched → Try a partial or case-insensitive name, e.g., --source espn. Use list-sources to see the exact names.
- LLM slow or failing → Use --disable-llm or set OPENAI_TIMEOUT in .env.
- DB errors → Run status to check connectivity, confirm SUPABASE_URL/KEY, and verify schema.

## 8) Useful commands
- Validate config:
  python scripts/pipeline_cli.py validate --config feeds.yaml
- Run all enabled sources (dry-run):
  python scripts/pipeline_cli.py run --config feeds.yaml --dry-run --disable-llm
- Run a single source with LLM:
  python scripts/pipeline_cli.py run --config feeds.yaml --source espn

That’s it! You can now fetch, filter, and optionally store NFL news items locally or in your database.

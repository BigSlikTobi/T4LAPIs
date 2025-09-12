# T4LAPIs - NFL Data Management System

A comprehensive Python-based system for fetching, transforming, and loading NFL data into Supabase databases. This project provides automated data pipelines, CLI tools, and robust data management capabilities for NFL analytics applications.

## 🏈 Overview

T4LAPIs is designed to handle complete NFL data workflows, from fetching raw data via the `nfl_data_py` library to loading processed data into Supabase databases. The system supports both one-time data loads and automated recurring updates through GitHub Actions workflows.

### Key Features

- **📊 Comprehensive NFL Data Coverage**: 24+ different NFL datasets including teams, players, games, statistics, injuries, and advanced analytics
- **🔄 Automated Data Pipelines**: GitHub Actions workflows for scheduled data updates
– **🤖 AI-Powered Content Generation**: Automated personalized summaries using Gemini 1.5 Flash + OpenAI gpt-5-nano entity extraction
- **� Trending Summary Generator**: AI-powered comprehensive summaries for trending NFL entities with pipeline integration
- **�🛠️ CLI Tools**: Command-line interfaces for manual data operations
- **📈 Smart Update Logic**: Intelligent detection of data gaps and incremental updates
- **🌐 Google Search Grounding**: Real-time NFL information retrieval for enhanced accuracy
- **🧪 Full Test Coverage**: Comprehensive test suite ensuring reliability (60+ tests including 34+ LLM tests)
- **🐳 Docker Support**: Containerized deployment and execution
- **🔧 Modular Architecture**: Separated concerns for fetching, transforming, and loading data
- **🚀 FastAPI REST API**: Complete CRUD operations for user and preference management with 7 endpoints

## 📁 Project Structure

```
T4LAPIs/
├── src/                        # Core application code
│   └── core/
│       ├── data/              # Data management modules
For a friendly step-by-step setup, read docs/Getting_Started.md.

### LLM-backed entity extraction
- The pipeline uses a small LLM to help extract players and teams from titles/descriptions, validated against our NFL dictionary.
- It’s enabled by default. To disable, set:
    - NEWS_PIPELINE_ENTITY_LLM=0
    - Or, to bypass LLM wiring in the orchestrator entirely: NEWS_PIPELINE_DISABLE_ENTITY_LLM=1
- Provide your LLM API key if needed (see docs/Getting_Started.md) to improve entity accuracy; without a key, the pipeline will fall back to rules and aliases.

Default model and controls:
- Uses OpenAI gpt-5-nano by default for entity extraction.
- Override with OPENAI_ENTITY_MODEL or OPENAI_MODEL. Configure timeout via OPENAI_TIMEOUT.

│       └── utils/             # Utility functions
├── content_generation/        # AI-powered content generation
│   ├── personal_summary_generator.py  # Gemini-powered personalized summaries
│   └── trending_summary_generator.py  # AI summaries for trending entities
├── api/                       # FastAPI REST API
│   ├── main.py               # Complete CRUD API with 7 endpoints
│   ├── Dockerfile            # Container configuration
│   ├── docker-compose.yml    # Development deployment
│   └── test_endpoints.py     # API testing scripts
├── scripts/                   # CLI tools and automation scripts
│   ├── trending_topic_detector.py      # Trending NFL entity detection
│   ├── llm_entity_linker_cli.py        # LLM entity linking
│   ├── teams_cli.py                    # Team data management
│   ├── players_cli.py                  # Player data management
│   └── ...                             # Other CLI tools
├── tests/                     # Test suite (50+ comprehensive tests)
├── docs/                      # Documentation (centralized)
├── .github/workflows/         # GitHub Actions automation
│   ├── games-auto-update.yml         # Automated game data updates
│   ├── player_weekly_stats_update.yml # Weekly stats automation
│   └── personal_summary_generation.yml # Hourly personalized summaries
├── examples/                  # Usage examples
├── injury_updates/            # Injury data specific tools
├── roster_updates/           # Roster data specific tools
└── Dockerfile                # Container configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Supabase account and database
- Environment variables configured (see [Environment Setup](#environment-setup))

### Installation

```bash
# Clone the repository
git clone https://github.com/BigSlikTobi/T4LAPIs.git
cd T4LAPIs

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Basic Usage

```bash
# Load team data
python scripts/teams_cli.py --dry-run

# Load player data for 2024 season
python scripts/players_cli.py 2024 --dry-run

# Load game schedules for 2024
python scripts/games_cli.py 2024 --dry-run

# Load weekly player statistics
python scripts/player_weekly_stats_cli.py 2024 --week 1 --dry-run

# Detect trending NFL entities
python scripts/trending_topic_detector.py --hours 168 --top-n 5

# Generate AI summaries for trending entities
python content_generation/trending_summary_generator.py --entity-ids "KC,00-0033873" --dry-run

# Pipeline trending detection → summary generation
python scripts/trending_topic_detector.py --entity-ids-only | python content_generation/trending_summary_generator.py --from-stdin

# Run LLM-enhanced entity linking
python scripts/llm_entity_linker_cli.py test --text "Patrick Mahomes threw for 300 yards as the Chiefs beat the 49ers."

# Process articles with LLM entity linking
python scripts/llm_entity_linker_cli.py run --batch-size 20 --max-batches 5
```

**⚠️ LLM Entity Linking Setup**: For optimal performance, manually create the `get_unlinked_articles` SQL function in Supabase (see [Manual Database Setup](#🔧-manual-database-setup-for-llm-entity-linking)).

## 🚀 FastAPI REST API

The project includes a complete FastAPI-based REST API for managing users and their NFL team/player preferences.

### API Features

✅ **Complete CRUD Operations**
- User management (create, delete) 
- Preference management (create, read, update, delete individual/bulk)
- UUID validation and comprehensive error handling
- CASCADE deletion support (deleting user removes all preferences)

✅ **7 API Endpoints**
- `POST /users/` - Create user
- `DELETE /users/{user_id}` - Delete user 
- `POST /users/{user_id}/preferences` - Set user preferences
- `GET /users/{user_id}/preferences` - Get user preferences
- `PUT /users/{user_id}/preferences/{preference_id}` - Update specific preference  
- `DELETE /users/{user_id}/preferences` - Delete all user preferences
- `DELETE /users/{user_id}/preferences/{preference_id}` - Delete specific preference

✅ **Production Ready**
- Docker containerization with security hardening
- Interactive API documentation (Swagger UI + ReDoc)
- Comprehensive error handling and validation
- Database integration with Supabase

### Quick Start API

#### 1. Activate Virtual Environment (REQUIRED)
```bash
# Navigate to project root
cd T4LAPIs

# Activate virtual environment - THIS IS REQUIRED
source venv/bin/activate
```

#### 2. Run the API

**Option A: Local Development (Recommended)**
```bash
# Make sure you're in the api directory and venv is activated
cd api
python main.py
```

**Option B: Docker (Production)**
```bash
# Build and run with Docker Compose
cd api
docker-compose up -d --build

# Or use the helper script
./docker.sh compose
```

#### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Create user
curl -X POST http://localhost:8000/users/

# Set preferences (replace {user_id} with actual UUID)
curl -X POST -H "Content-Type: application/json" \
  -d '{"entities": [{"entity_id": "KC", "type": "team"}]}' \
  http://localhost:8000/users/{user_id}/preferences
```

#### 4. Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

For complete API documentation, see: [📚 API Reference](docs/API_Reference.md)

## 📊 Available NFL Data

The system provides access to **24 different NFL datasets** with **841+ columns** across:

- **Core Tables**: Teams, Players, Games, Weekly Statistics
- **Advanced Stats**: Play-by-Play, Next Gen Stats, PFF Data
- **Personnel Data**: Draft picks, Combine results, Contracts, Depth charts
- **Specialized Data**: Injuries, Officials, Betting lines, Formation data

## 🧰 NFL News Pipeline CLI

Run validations and status:

- Validate config: `python scripts/pipeline_cli.py validate --config feeds.yaml`
- Status/health: `python scripts/pipeline_cli.py status --config feeds.yaml`

Run the pipeline:

- Dry-run (no DB writes): `python scripts/pipeline_cli.py run --config feeds.yaml --dry-run`
- Single source: `python scripts/pipeline_cli.py run --config feeds.yaml --source espn`
- Disable LLM: `python scripts/pipeline_cli.py run --disable-llm`

The pipeline respects SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, and cache flags (NEWS_PIPELINE_LLM_CACHE*).
- **Entity Linking**: LLM-enhanced extraction and linking of players and teams from text

For detailed information about available datasets, see: [📋 NFL Data Reference](docs/NFL_Data_Reference.md)

## 🔧 Core Components

### Data Management Pipeline

1. **[Fetch](src/core/data/fetch.py)**: Raw data retrieval from `nfl_data_py`
2. **[Transform](src/core/data/transform.py)**: Data cleaning, validation, and formatting
3. **[Load](src/core/data/loaders/)**: Database insertion with conflict resolution

### CLI Tools

Located in the `scripts/` directory, these tools provide command-line interfaces for:

- **trending_topic_detector.py**: Trending NFL entity detection and analysis
- **trending_summary_generator.py**: AI-powered comprehensive summaries for trending entities
- **teams_cli.py**: Team data management
- **players_cli.py**: Player roster management  
- **games_cli.py**: Game schedule management
- **player_weekly_stats_cli.py**: Weekly statistics management
- **llm_entity_linker_cli.py**: LLM-enhanced entity linking and extraction
- **entity_dictionary_cli.py**: Entity dictionary management and utilities
- **setup_entity_linking_db.py**: Database index creation (manual SQL function setup required)

**Note**: For optimal LLM entity linking performance, ensure the manual SQL function setup is completed (see [Manual Database Setup](#🔧-manual-database-setup-for-llm-entity-linking)).

For detailed CLI documentation, see: [🛠️ CLI Tools Guide](docs/CLI_Tools_Guide.md)

### Automated Workflows

GitHub Actions workflows provide scheduled data updates:

- **Games Data**: 6 times per week during NFL season
- **Player Stats**: Weekly after Monday Night Football
- **Player Rosters**: Weekly on Wednesdays
- **Team Data**: Monthly updates
- **Entity Linking**: Every 30 minutes between 16:30-00:30 UTC for article processing
- **🆕 Personalized Summaries**: Hourly generation (6 AM - 11 PM UTC) using Gemini 1.5 Flash

#### 🤖 Automated Personalized Summary Generation

The system now includes automated generation of personalized NFL summaries powered by Gemini 1.5 Flash:

- **Schedule**: Runs every hour during peak hours (6 AM - 11 PM UTC)
- **Intelligence**: Uses user preferences to generate tailored content
- **Fallback**: The generator can use alternative providers if configured
- **Manual Trigger**: Available for testing and on-demand generation

**Setup**: See [GitHub Actions Setup Guide](docs/GitHub_Actions_Setup.md) for configuration details.

For workflow details, see: [⚙️ Automation Workflows](docs/Automation_Workflows.md)

#### 🔥 Trending Summary Generator (NEW - Epic 2 Task 4)

The system now includes a powerful trending summary generator that creates comprehensive summaries for trending NFL entities identified by the trending topic detector.

**Features**:
- **🎯 Trending Detection**: Integrates with trending topic detector via pipeline
- **📰 Multi-Source Intelligence**: Combines recent articles and player statistics  
- **🤖 Gemini-Powered**: Uses Gemini 2.5 Flash for engaging, journalistic content
- **💾 Database Storage**: Saves to `trending_entity_updates` table
- **🔄 Pipeline Integration**: Seamless stdin/stdout connectivity
- **⚡ Flexible Input**: CLI args, files, or pipe from trending detector
- **🧪 Dry Run Support**: Preview generation without database saves

**Quick Start**:
```bash
# Generate summaries for specific entities
python content_generation/trending_summary_generator.py --entity-ids "00-0033873,KC,NYJ"

# Pipeline with trending detector (recommended)
python scripts/trending_topic_detector.py --entity-ids-only | python content_generation/trending_summary_generator.py --from-stdin

# Read from file
python content_generation/trending_summary_generator.py --input-file trending_entities.txt

# Preview mode (no database saves)
python content_generation/trending_summary_generator.py --entity-ids "KC" --dry-run

# Custom options
python content_generation/trending_summary_generator.py --entity-ids "KC" --hours 48 --llm-provider deepseek
```

**Example Output**:
```
📊 Trending Summary Generation Results:
   Entities processed: 3
   Summaries generated: 2  
   Articles analyzed: 186
   Stats analyzed: 5
   Errors encountered: 0
   Processing time: 18.2s
   LLM time: 14.7s
   Success rate: 66.7%
```

**Database Schema** (`trending_entity_updates`):
```json
{
    "id": "7ffdf99e-aa4d-4451-bb0c-47edcfbed795",
    "created_at": "2025-08-03T16:14:09.401587+00:00",
    "trending_content": "**Chiefs Dynasty Continues: Why Kansas City Remains NFL's Elite**\n\nThe Kansas City Chiefs are trending as they solidify their position as the NFL's premier franchise...",
    "source_articles": [],
    "source_starts": [],
    "player_ids": [], 
    "team_ids": ["KC"]
}
```

**Testing**:
```bash
# Run comprehensive test suite (32 test cases)
python -m pytest tests/test_trending_summary_generator.py -v
```

**Test Coverage**:
- ✅ TrendingSummary dataclass functionality
- ✅ Entity type determination and name retrieval
- ✅ Article and statistics fetching from database
- ✅ LLM integration with Gemini and DeepSeek fallback
- ✅ Prompt generation for players and teams
- ✅ Database storage with correct schema
- ✅ CLI argument parsing and multiple input methods
- ✅ Pipeline integration (stdin/stdout)
- ✅ Error handling and edge cases
- ✅ Dry run mode and configuration options

**CLI Options**:
- `--entity-ids`: Comma-separated entity IDs
- `--input-file`: File containing entity IDs  
- `--from-stdin`: Read from stdin (pipeline mode)
- `--hours`: Lookback period (default: 72)
- `--dry-run`: Preview mode without saves
- `--llm-provider`: Choose 'gemini' or 'deepseek'
- `--output-format`: 'summary' or 'json'
- `--verbose`: Enable debug logging

For detailed implementation, see `content_generation/trending_summary_generator.py` and the comprehensive test suite.

## 🐳 Docker Usage

The project includes Docker support for consistent execution environments:

```bash
# Build the image
docker build -t t4lapis-app .

# Run with environment file
docker run --rm --env-file .env t4lapis-app python scripts/teams_cli.py --dry-run

# Interactive shell for debugging
docker run --rm -it --env-file .env t4lapis-app bash
```

## ⚙️ Environment Setup

Required environment variables:

```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
DEEPSEEK_API_KEY=your_deepseek_api_key  # For LLM entity linking
GEMINI_API_KEY=your_gemini_api_key      # For AI content generation (preferred)
LOG_LEVEL=INFO  # Optional: DEBUG, INFO, WARNING, ERROR
```

## 🤖 AI Content Generation (NEW - Sprint 3)

The system now includes advanced AI-powered content generation capabilities using **Gemini 1.5 Flash** with optional Google Search grounding for creating personalized NFL summaries and trending topic detection.

### LLM Integration (`src/core/llm/llm_setup.py`)

Unified LLM setup supporting multiple AI providers:

#### Available Models
- **🌟 Gemini 1.5 Flash** (Primary): Fast, engaging content generation with optional Google Search grounding
- **🔧 DeepSeek Chat** (Fallback): Reliable entity extraction and backup content generation

#### Quick Setup

```python
from src.core.llm.llm_setup import initialize_model, generate_content_with_model

# Initialize Gemini with grounding
gemini_config = initialize_model("gemini", "flash", grounding_enabled=True)

# Initialize DeepSeek as fallback
deepseek_config = initialize_model("deepseek", "chat")

# Generate content
messages = [
    {"role": "system", "content": "You are an NFL expert."},
    {"role": "user", "content": "Summarize the latest Chiefs news."}
]

response = generate_content_with_model(gemini_config, messages)
```

### Personalized Summary Generator

Creates personalized AI summaries for each user based on their preferences, leveraging **Gemini's advanced reasoning** and **Google Search grounding** for accurate, engaging content.

#### Features
- **🎯 User-Centric**: Processes all users and their individual preferences
- **📊 Multi-Source Data**: Combines articles, player statistics, and team updates
- **🧠 Rolling Context**: Uses previous summaries as context for continuity
- **⚡ Gemini-Powered**: Faster generation with superior content quality
- **🔍 Google Grounding**: Real-time NFL information retrieval (when enabled)
- **⏰ Time-Aware**: Configurable lookback periods (default: 24 hours)
- **💾 Database Integration**: Stores results in `generated_updates` table
- **🔄 Comprehensive Tracking**: Full statistics and error reporting

#### Quick Start

```bash
# Run personalized summary generation (24-hour lookback)
python content_generation/personal_summary_generator.py

# The script will:
# 1. Process all users with preferences
# 2. Gather new articles and stats for each entity
# 3. Generate AI summaries using LLM
# 4. Store results in generated_updates table
```

#### Example Output
```
📊 Generation Results:
   Users processed: 15
   Preferences processed: 47
   Summaries generated: 42
   Errors encountered: 0
   Processing time: 127.3s
   LLM time: 89.2s
   Success rate: 89.4%
```

#### Database Schema

The `generated_updates` table stores personalized summaries:

```json
{
    "update_id": "f6b7a8fe-99b4-470c-9efe-129728a3e1d1",
    "user_id": "00b4b7d6-eabe-4179-955b-3b8a8ab32e95",
    "entity_id": "00-0033873", // Player ID or team abbreviation
    "entity_type": "player",   // "player" or "team"
    "created_at": "2025-08-02T20:13:52.312532+00:00",
    "update_content": "Patrick Mahomes continues to excel this season with 350 passing yards and 3 TDs in the latest Chiefs victory...",
    "source_article_ids": [1001, 1002, 1003],
    "source_stat_ids": ["00-0033873_2024_15"]
}
```

#### Testing

Comprehensive test suite with 25 test cases covering:
- ✅ LLM initialization and error handling
- ✅ User preference processing and validation  
- ✅ Article and statistics retrieval
- ✅ Summary generation with context
- ✅ Database operations and error scenarios
- ✅ Complete integration workflow

```bash
# Run tests for personalized summary generator
python -m pytest tests/test_personal_summary_generator.py -v
```

### Technical Implementation

#### Workflow
1. **User Discovery**: Fetch all users with active preferences
2. **Content Gathering**: Collect new articles and stats for each entity
3. **Context Building**: Retrieve previous summaries for rolling context
4. **AI Generation**: Create personalized summaries using DeepSeek LLM
5. **Storage**: Save generated content with source tracking

#### LLM Integration
- **Model**: DeepSeek Chat for high-quality content generation
- **Prompting**: Structured prompts with previous summary context
- **Temperature**: 0.7 for engaging, natural content
- **Token Management**: Smart truncation and content optimization

#### Error Handling
- **Graceful Degradation**: Continue processing on individual failures
- **Comprehensive Logging**: Detailed error information for debugging
- **Statistics Tracking**: Monitor success rates and performance metrics
- **Retry Logic**: Built-in resilience for LLM API calls

For detailed implementation information, see the source code and tests in `content_generation/`.

### 🔧 Manual Database Setup for LLM Entity Linking

**Important**: Due to Supabase limitations, the following SQL function must be created manually in the Supabase SQL Editor for optimal LLM entity linking performance:

1. Open your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the following SQL command:

```sql
CREATE OR REPLACE FUNCTION get_unlinked_articles(batch_limit INTEGER)
RETURNS SETOF "SourceArticles" AS $$
BEGIN
    RETURN QUERY
    SELECT sa.*
    FROM "SourceArticles" sa
    WHERE NOT EXISTS (
        SELECT 1
        FROM "article_entity_links" ael
        WHERE ael.article_id = sa.id
    )
    AND sa."contentType" IN ('news_article', 'news-round-up', 'topic_collection')
    AND sa."Content" IS NOT NULL
    AND sa."Content" != ''
    ORDER BY sa.id
    LIMIT batch_limit;
END;
$$ LANGUAGE plpgsql;
```

**Note**: Without this function, the system will automatically fall back to less efficient query methods, but functionality will remain intact.

## 🧪 Testing

The project includes comprehensive tests covering all major functionality:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_games_auto_update.py -v

# Run LLM-specific tests (34 tests total)
python -m pytest tests/test_llm_init.py tests/test_llm_entity_linker.py -v

# Run dedicated LLM test runner
python tests/run_llm_tests.py

# Run FastAPI tests (15+ tests)
python -m pytest tests/test_fastapi_basic.py tests/test_user_preference_api.py tests/test_crud_operations.py -v
```

### Test Coverage Summary
- **Core Data Pipeline**: 15+ tests for fetching, transforming, and loading NFL data
- **LLM Integration**: 34 comprehensive tests for entity linking and DeepSeek AI integration
- **FastAPI API**: 15+ tests for complete CRUD operations and error handling
- **CLI Tools**: Multiple tests for command-line interfaces
- **Database Operations**: Tests for all database interactions and conflict resolution

For detailed testing documentation, see: [🧪 Testing Guide](docs/Testing_Guide.md) and [🤖 LLM Test Coverage](docs/LLM_Test_Coverage.md)

## 📖 Documentation

### Core Documentation

- [📋 NFL Data Reference](docs/NFL_Data_Reference.md) - Complete data tables and columns reference
- [🛠️ CLI Tools Guide](docs/CLI_Tools_Guide.md) - Command-line interface documentation  
- [⚙️ Automation Workflows](docs/Automation_Workflows.md) - GitHub Actions workflows
- [� GitHub Actions Setup](docs/GitHub_Actions_Setup.md) - Automated personalized summary setup guide
- [�🧪 Testing Guide](docs/Testing_Guide.md) - Test suite documentation
- [🔧 Technical Details](docs/Technical_Details.md) - Architecture and implementation details
- [🚀 API Reference](docs/API_Reference.md) - Complete FastAPI REST API documentation
- [🤖 LLM Test Coverage](docs/LLM_Test_Coverage.md) - LLM functionality testing documentation

### Specialized Topics

- [🔄 Data Loaders](docs/Data_Loaders.md) - Database loading mechanisms
- [⚡ Auto-Update Scripts](docs/Auto_Update_Scripts.md) - Smart update logic
- [🚨 Troubleshooting](docs/Troubleshooting.md) - Common issues and solutions

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
Data Flow: NFL API → Fetch → Transform → Validate → Load → Supabase
                    ↓                                    ↓
            LLM Entity Linking → Article Processing → Entity Links
                    ↓
            FastAPI REST API → User/Preference Management → Database
```

### Key Design Principles

- **Separation of Concerns**: Distinct modules for fetching, transforming, and loading
- **Error Resilience**: Comprehensive error handling and logging
- **Data Integrity**: Validation and conflict resolution
- **AI Integration**: LLM-enhanced entity extraction with validation
- **API-First Design**: RESTful API with complete CRUD operations
- **Scalability**: Modular design supports easy extension
- **Maintainability**: Clear code structure and comprehensive tests

## 🎯 Project Status & Achievements

### ✅ Epic 2: User & Preference API (COMPLETED)

**All Tasks Successfully Completed:**

#### Task 4: FastAPI Project Setup ✅
- Modern FastAPI application with proper structure
- Comprehensive Pydantic v2 models
- CORS middleware and lifespan management
- Complete test suite (5 basic functionality tests)

#### Task 5: User & Preference Endpoints ✅
- **POST /users/** - Create new user with UUID generation
- **POST /users/{user_id}/preferences** - Set user preferences with validation
- **GET /users/{user_id}/preferences** - Retrieve user preferences
- Comprehensive validation and error handling
- Additional test suite (10 endpoint tests)

#### Task 6: Docker Containerization ✅
- Optimized Dockerfile with Python 3.13-slim
- Docker Compose configuration for easy deployment
- Security hardening (non-root user)
- Helper scripts for container management

#### Enhanced CRUD Operations ✅
- **DELETE /users/{user_id}** - Delete user with CASCADE preference deletion
- **PUT /users/{user_id}/preferences/{preference_id}** - Update specific preference
- **DELETE /users/{user_id}/preferences** - Delete all user preferences
- **DELETE /users/{user_id}/preferences/{preference_id}** - Delete specific preference

### 🚀 Production Ready Features
- **7 Complete API Endpoints** with full CRUD operations
- **34 LLM Tests** ensuring AI functionality reliability
- **15+ API Tests** covering all endpoints and error scenarios
- **Interactive Documentation** (Swagger UI + ReDoc)
- **Database Integration** with Supabase and CASCADE operations
- **Docker Support** for consistent deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [nfl_data_py](https://github.com/cooperdff/nfl_data_py) - For providing comprehensive NFL data access
- [Supabase](https://supabase.com/) - For the backend database platform
- OpenAI - For LLM-powered entity extraction capabilities
- NFL community - For maintaining and contributing to open-source NFL data

---

**Need Help?** Check the [documentation](docs/) or open an issue for support.

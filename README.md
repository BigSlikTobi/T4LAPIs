# T4LAPIs - NFL Data Management System

A comprehensive Python-based system for fetching, transforming, and loading NFL data into Supabase databases. This project provides automated data pipelines, CLI tools, and robust data management capabilities for NFL analytics applications.

## 🏈 Overview

T4LAPIs is designed to handle complete NFL data workflows, from fetching raw data via the `nfl_data_py` library to loading processed data into Supabase databases. The system supports both one-time data loads and automated recurring updates through GitHub Actions workflows.

### Key Features

- **📊 Comprehensive NFL Data Coverage**: 24+ different NFL datasets including teams, players, games, statistics, injuries, and advanced analytics
- **🔄 Automated Data Pipelines**: GitHub Actions workflows for scheduled data updates
- **🛠️ CLI Tools**: Command-line interfaces for manual data operations
- **📈 Smart Update Logic**: Intelligent detection of data gaps and incremental updates
- **� LLM-Enhanced Entity Linking**: DeepSeek AI integration for intelligent entity extraction and linking
- **�🧪 Full Test Coverage**: Comprehensive test suite ensuring reliability (34+ LLM tests included)
- **🐳 Docker Support**: Containerized deployment and execution
- **🔧 Modular Architecture**: Separated concerns for fetching, transforming, and loading data

## 📁 Project Structure

```
T4LAPIs/
├── src/                        # Core application code
│   └── core/
│       ├── data/              # Data management modules
│       ├── db/                # Database initialization
│       ├── llm/               # LLM integration and entity linking
│       └── utils/             # Utility functions
├── scripts/                   # CLI tools and automation scripts
├── tests/                     # Test suite
├── docs/                      # Documentation (see links below)
├── .github/workflows/         # GitHub Actions automation
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

# Run LLM-enhanced entity linking
python scripts/llm_entity_linker_cli.py test --text "Patrick Mahomes threw for 300 yards as the Chiefs beat the 49ers."

# Process articles with LLM entity linking
python scripts/llm_entity_linker_cli.py run --batch-size 20 --max-batches 5
```

## 📊 Available NFL Data

The system provides access to **24 different NFL datasets** with **841+ columns** across:

- **Core Tables**: Teams, Players, Games, Weekly Statistics
- **Advanced Stats**: Play-by-Play, Next Gen Stats, PFF Data
- **Personnel Data**: Draft picks, Combine results, Contracts, Depth charts
- **Specialized Data**: Injuries, Officials, Betting lines, Formation data
- **Entity Linking**: LLM-enhanced extraction and linking of players and teams from text

For detailed information about available datasets, see: [📋 NFL Data Reference](docs/NFL_Data_Reference.md)

## 🔧 Core Components

### Data Management Pipeline

1. **[Fetch](src/core/data/fetch.py)**: Raw data retrieval from `nfl_data_py`
2. **[Transform](src/core/data/transform.py)**: Data cleaning, validation, and formatting
3. **[Load](src/core/data/loaders/)**: Database insertion with conflict resolution

### CLI Tools

Located in the `scripts/` directory, these tools provide command-line interfaces for:

- **teams_cli.py**: Team data management
- **players_cli.py**: Player roster management  
- **games_cli.py**: Game schedule management
- **player_weekly_stats_cli.py**: Weekly statistics management
- **llm_entity_linker_cli.py**: LLM-enhanced entity linking and extraction
- **entity_dictionary_cli.py**: Entity dictionary management and utilities

For detailed CLI documentation, see: [🛠️ CLI Tools Guide](docs/CLI_Tools_Guide.md)

### Automated Workflows

GitHub Actions workflows provide scheduled data updates:

- **Games Data**: 6 times per week during NFL season
- **Player Stats**: Weekly after Monday Night Football
- **Player Rosters**: Weekly on Wednesdays
- **Team Data**: Monthly updates
- **Entity Linking**: Every 30 minutes between 16:30-00:30 UTC for article processing

For workflow details, see: [⚙️ Automation Workflows](docs/Automation_Workflows.md)

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
LOG_LEVEL=INFO  # Optional: DEBUG, INFO, WARNING, ERROR
```

## 🧪 Testing

The project includes comprehensive tests covering all major functionality:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_games_auto_update.py -v

# Run LLM-specific tests
python -m pytest tests/test_llm_init.py tests/test_llm_entity_linker.py -v

# Run dedicated LLM test runner
python tests/run_llm_tests.py
```

For testing details, see: [🧪 Testing Guide](docs/Testing_Guide.md)

## 📖 Documentation

### Core Documentation

- [📋 NFL Data Reference](docs/NFL_Data_Reference.md) - Complete data tables and columns reference
- [🛠️ CLI Tools Guide](docs/CLI_Tools_Guide.md) - Command-line interface documentation  
- [⚙️ Automation Workflows](docs/Automation_Workflows.md) - GitHub Actions workflows
- [🧪 Testing Guide](docs/Testing_Guide.md) - Test suite documentation
- [🔧 Technical Details](docs/Technical_Details.md) - Architecture and implementation details
- [🤖 LLM Test Coverage](tests/LLM_TEST_COVERAGE.md) - LLM functionality testing documentation

### Specialized Topics

- [🔄 Data Loaders](docs/Data_Loaders.md) - Database loading mechanisms
- [⚡ Auto-Update Scripts](docs/Auto_Update_Scripts.md) - Smart update logic
- [🚨 Troubleshooting](docs/Troubleshooting.md) - Common issues and solutions

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
Data Flow: NFL API → Fetch → Transform → Validate → Load → Supabase
                    ↓                                    ↓
                CLI Tools ← Auto Scripts ← GitHub Actions
                    ↓
            LLM Entity Linking → Article Processing → Entity Links
```

### Key Design Principles

- **Separation of Concerns**: Distinct modules for fetching, transforming, and loading
- **Error Resilience**: Comprehensive error handling and logging
- **Data Integrity**: Validation and conflict resolution
- **AI Integration**: LLM-enhanced entity extraction with validation
- **Scalability**: Modular design supports easy extension
- **Maintainability**: Clear code structure and comprehensive tests

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
- [DeepSeek AI](https://www.deepseek.com/) - For LLM-powered entity extraction capabilities
- NFL community - For maintaining and contributing to open-source NFL data

---

**Need Help?** Check the [documentation](docs/) or open an issue for support.

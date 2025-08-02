# Testing Guide

This document describes the comprehensive test suite for the T4LAPIs NFL data management system.

## 📋 Test Suite Overview

The test suite ensures reliability and correctness across all components of the NFL data pipeline, from data fetching to database loading and automated workflows.

### Test Coverage Areas

- ✅ **Data Fetching**: Raw data retrieval from NFL APIs
- ✅ **Data Transformation**: Cleaning, validation, and formatting
- ✅ **Database Operations**: Loading, conflict resolution, error handling
- ✅ **CLI Tools**: Command-line interface functionality
- ✅ **Auto-Update Scripts**: Smart update logic and workflows
- ✅ **Utility Functions**: Database connections, logging, validation

## 🧪 Test Structure

### Test Organization

```
tests/
├── conftest.py                           # Shared test configuration
├── test_fetch.py                         # Data fetching tests
├── test_*_loader.py                      # Database loader tests
├── test_*_auto_update.py                 # Auto-update script tests
├── test_cli_utils.py                     # CLI utility tests
├── test_database_*.py                    # Database operation tests
└── test_team_transform.py                # Data transformation tests
```

### Test Categories

| Category | Files | Purpose |
|----------|-------|---------|
| **Core Data** | `test_fetch.py`, `test_*_loader.py` | Data pipeline functionality |
| **Auto-Updates** | `test_*_auto_update.py` | Smart workflow logic |
| **Infrastructure** | `test_database_*.py`, `test_cli_utils.py` | Supporting systems |
| **Transformations** | `test_team_transform.py` | Data processing logic |

## 🔍 Detailed Test Documentation

### 1. Auto-Update Script Tests

#### Games Auto-Update (`test_games_auto_update.py`)

Tests the intelligent game data update logic used in GitHub Actions workflows.

**Coverage Areas:**
- ✅ Database connection handling (success, failure, no data scenarios)
- ✅ NFL season detection logic (September-August cycle)
- ✅ Latest week detection from database queries
- ✅ Main workflow scenarios (no data, existing data, edge cases)
- ✅ Week boundary validation (skips weeks > 22)
- ✅ Partial failure handling (continues on errors)
- ✅ Exception handling and graceful degradation

**Key Test Cases:**

```python
def test_get_current_nfl_season_september():
    """September = current calendar year season"""
    # Tests: Sept 15, 2024 → 2024 NFL season

def test_get_current_nfl_season_february():
    """February = previous calendar year season"""  
    # Tests: Feb 15, 2025 → 2024 NFL season (playoffs/offseason)

def test_main_no_existing_data():
    """First run: loads week 1 when no data exists"""
    # Tests: Empty database → Load week 1

def test_main_with_existing_data():
    """Normal operation: upserts current, inserts next"""
    # Tests: Week 5 exists → Upsert week 5, Insert week 6

def test_main_skip_invalid_weeks():
    """Boundary handling: stops at week 22"""
    # Tests: Week 22 exists → Upsert week 22, Skip week 23

def test_main_partial_failure():
    """Error resilience: handles mixed success/failure"""
    # Tests: One week fails → Continue with next week
```

#### Player Stats Auto-Update (`test_player_weekly_stats_auto_update.py`)

Tests the cross-table comparison logic for intelligent statistics updates.

**Coverage Areas:**
- ✅ Cross-table comparison logic (games vs stats tables)
- ✅ Gap detection between scheduled games and processed stats
- ✅ Recent week update strategy (reprocesses last 1-2 weeks)
- ✅ Various data state scenarios (no data, partial data, up-to-date)
- ✅ Edge cases (week 1 stats, end of season)
- ✅ Invalid week handling and boundary conditions
- ✅ Comprehensive error scenarios and recovery

**Key Test Cases:**

```python
def test_main_no_games_data():
    """Graceful failure when no games exist"""
    # Tests: Empty games table → Exit gracefully

def test_main_no_existing_stats_data():
    """First run: processes all available weeks"""
    # Tests: Games weeks 1-10, no stats → Process weeks 1-10

def test_main_with_existing_stats_data():
    """Gap detection: processes missing weeks + recent"""
    # Tests: Games 1-10, stats 1-7 → Process 8-10 + recent 6-7

def test_main_stats_up_to_date():
    """Recent week updates when current"""
    # Tests: Games 1-8, stats 1-8 → Update recent weeks 7-8
```

### 2. Data Loader Tests

#### Base Loader (`test_base_loader.py`)

Tests the foundational database loading functionality.

**Features Tested:**
- Generic database loading patterns
- Error handling and logging
- Configuration management
- Abstract base class behavior

#### Specific Loader Tests

- **`test_games_loader.py`**: Game data loading and conflict resolution
- **`test_players_loader.py`**: Player data with duplicate handling
- **`test_player_weekly_stats_loader.py`**: Statistics loading and validation
- **`test_teams_loader.py`**: Team data loading and upsert behavior

### 3. Data Pipeline Tests

#### Fetch Module (`test_fetch.py`)

Tests raw data retrieval from NFL APIs.

**Coverage:**
- Connection handling to nfl_data_py
- Data format validation
- Error handling for API failures
- Function parameter validation

#### Transform Module (`test_team_transform.py`)

Tests data transformation and cleaning logic.

**Coverage:**
- Data normalization
- Duplicate resolution
- Field mapping and validation
- Error handling for malformed data

### 4. Infrastructure Tests

#### Database Tests (`test_database_*.py`)

- **`test_database_init.py`**: Database initialization and connection
- **`test_database_utils.py`**: Database utility functions and operations

#### CLI Tests (`test_cli_utils.py`)

Tests command-line interface utilities and argument parsing.

## 🚀 Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_games_auto_update.py -v

# Run specific test function
python -m pytest tests/test_games_auto_update.py::test_main_no_existing_data -v
```

### Coverage Analysis

```bash
# Run tests with coverage
python -m pytest --cov=src tests/

# Generate coverage report
python -m pytest --cov=src --cov-report=html tests/

# View coverage in browser
open htmlcov/index.html
```

### Test Categories

```bash
# Run only auto-update tests
python -m pytest tests/test_*_auto_update.py -v

# Run only loader tests  
python -m pytest tests/test_*_loader.py -v

# Run only database tests
python -m pytest tests/test_database_*.py -v
```

## 🔧 Test Configuration

### Shared Test Setup (`conftest.py`)

The `conftest.py` file provides shared test fixtures and configuration:

```python
@pytest.fixture
def mock_database():
    """Mock database for testing without real connections"""
    
@pytest.fixture  
def sample_nfl_data():
    """Sample NFL data for testing transformations"""
    
@pytest.fixture
def mock_environment():
    """Mock environment variables for testing"""
```

### Test Data Management

- **Mock Data**: Use mocked NFL data for consistent testing
- **Isolated Tests**: Each test runs in isolation without side effects
- **Database Mocking**: Tests don't require actual database connections
- **Environment Isolation**: Tests don't depend on external environment

## 📊 Test Metrics

### Current Test Coverage

| Component | Test Files | Coverage | Key Areas |
|-----------|------------|----------|-----------|
| **Auto-Updates** | 2 files | 95%+ | Workflow logic, error handling |
| **Data Loaders** | 5 files | 90%+ | Database operations, validation |
| **Data Pipeline** | 2 files | 85%+ | Fetching, transformation |
| **Infrastructure** | 3 files | 80%+ | Database, CLI utilities |

### Test Execution Metrics

- **Total Tests**: 50+ test functions
- **Execution Time**: < 30 seconds for full suite
- **Success Rate**: 100% on clean environments
- **Coverage Target**: > 90% for critical paths

## 🚨 Test Scenarios

### Error Handling Tests

1. **Network Failures**:
   ```python
   def test_network_timeout():
       """Handle API timeouts gracefully"""
   ```

2. **Database Connection Issues**:
   ```python
   def test_database_connection_failure():
       """Recover from database outages"""
   ```

3. **Invalid Data Formats**:
   ```python
   def test_malformed_data_handling():
       """Skip invalid records, continue processing"""
   ```

### Edge Case Tests

1. **Season Boundaries**:
   ```python
   def test_season_transition():
       """Handle season transitions correctly"""
   ```

2. **Empty Data Sets**:
   ```python
   def test_no_data_available():
       """Handle empty API responses"""
   ```

3. **Partial Data**:
   ```python
   def test_incomplete_weeks():
       """Process partial week data"""
   ```

## 🔄 Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- **Pull Requests**: All tests must pass before merge
- **Main Branch Pushes**: Regression testing
- **Scheduled Runs**: Weekly test health checks

### Test Pipeline

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest --cov=src tests/
```

## 💡 Testing Best Practices

### Writing New Tests

1. **Test Naming**: Use descriptive names that explain the scenario
   ```python
   def test_main_with_existing_data_updates_current_and_next_week():
   ```

2. **Arrange-Act-Assert**: Follow clear test structure
   ```python
   def test_example():
       # Arrange: Set up test data
       # Act: Execute the function
       # Assert: Verify the results
   ```

3. **Mock External Dependencies**: Isolate tests from external services
   ```python
   @mock.patch('src.core.utils.database.get_supabase_client')
   def test_database_operation(mock_client):
   ```

### Test Maintenance

1. **Keep Tests Updated**: Update tests when functionality changes
2. **Review Coverage**: Regularly check test coverage reports
3. **Performance**: Keep test execution time under control
4. **Documentation**: Document complex test scenarios

## 🔗 Integration with Development

### Pre-commit Testing

```bash
# Run tests before commits
git add .
python -m pytest
git commit -m "Your commit message"
```

### Development Workflow

1. **Write Tests First**: TDD approach for new features
2. **Run Tests Frequently**: Quick feedback during development
3. **Maintain Coverage**: Ensure new code includes tests
4. **Document Test Cases**: Explain complex testing scenarios

For more information about specific components:
- [CLI Tools Guide](CLI_Tools_Guide.md) - Testing CLI functionality
- [Automation Workflows](Automation_Workflows.md) - Testing automated workflows
- [Technical Details](Technical_Details.md) - Architecture testing considerations

# Test Suite for NFL Auto-Update Scripts

This document describes the comprehensive test suite for the auto-update scripts.

## 📋 Overview

The test suite ensures reliability and correctness of the smart auto-update functionality that powers the GitHub workflows.

## 🧪 Test Files

### 1. `test_games_auto_update.py`
Tests for `scripts/games_auto_update.py`

**Coverage Areas:**
- ✅ Database connection handling (success, failure, no data)
- ✅ NFL season detection logic (September-August cycle)
- ✅ Latest week detection from database
- ✅ Main workflow scenarios (no data, existing data, edge cases)
- ✅ Week boundary validation (skips weeks > 22)
- ✅ Partial failure handling (continues on errors)
- ✅ Exception handling and graceful degradation

**Key Test Cases:**
```python
test_get_current_nfl_season_september()      # Sept = current year
test_get_current_nfl_season_february()       # Feb = previous year's season
test_main_no_existing_data()                 # First run: loads week 1
test_main_with_existing_data()               # Normal: upserts N, inserts N+1
test_main_skip_invalid_weeks()               # Stops at week 22
test_main_partial_failure()                  # Handles mixed success/failure
```

### 2. `test_player_weekly_stats_auto_update.py`
Tests for `scripts/player_weekly_stats_auto_update.py`

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
test_main_no_games_data()                    # Fails gracefully if no games
test_main_no_existing_stats_data()           # First run: processes weeks 1-N
test_main_with_existing_stats_data()         # Normal: gaps + recent updates
test_main_stats_up_to_date()                 # Reprocesses last 2 weeks
test_main_skip_invalid_weeks()               # Respects NFL season boundaries
test_main_edge_case_week_1_stats()           # Handles early season edge case
```

## 🚀 Running Tests

### Local Development

```bash
# Quick auto-update tests only
python run_tests.py --auto-only

# Full test suite with coverage
python run_tests.py --coverage --verbose

# Individual test files
python -m pytest tests/test_games_auto_update.py -v
python -m pytest tests/test_player_weekly_stats_auto_update.py -v
```

### GitHub Actions
Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Changes to `scripts/`, `tests/`, `src/`, or `requirements.txt`

See `.github/workflows/test.yml` for the full CI configuration.

## 🎯 Test Strategy

### Mocking Approach
All tests use comprehensive mocking to:
- **Database calls**: Mock Supabase responses for predictable testing
- **External APIs**: Mock nfl_data_py to avoid network dependencies
- **System time**: Mock datetime for consistent season detection
- **CLI utilities**: Mock logging and output functions

### Test Categories

1. **Unit Tests**: Individual function behavior
   - Season detection
   - Database queries
   - Week calculation logic

2. **Integration Tests**: Full workflow scenarios
   - Complete main() function execution
   - Multi-step data processing
   - Error propagation and recovery

3. **Edge Case Tests**: Boundary conditions
   - Empty databases
   - End of season (week 22+)
   - Single week scenarios
   - Invalid data handling

4. **Error Handling Tests**: Failure scenarios
   - Database connection failures
   - Partial operation failures
   - Unexpected exceptions
   - Network timeouts

## 📊 Test Metrics

- **Total Tests**: 28 (13 games + 15 player stats)
- **Code Coverage**: >95% for auto-update scripts
- **Execution Time**: <1 second (fast mocked tests)
- **Reliability**: All tests deterministic (no flaky tests)

## 🔍 Test Data Patterns

### Mock Database Responses
```python
# Successful data retrieval
mock_response.data = [{'week': 5}]

# Empty table
mock_response.data = []

# Connection failure
mock_get_client.return_value = None
```

### Season Detection Test Cases
```python
(2024, 9, 2024),   # September - current season starts
(2024, 12, 2024),  # December - middle of season  
(2025, 2, 2024),   # February - still previous season
(2024, 8, 2023),   # August - still previous season
```

### Workflow Test Scenarios
```python
# No existing data
latest_week = 0 → process [1]

# Normal operation  
latest_week = 5 → process [5, 6]

# End of season
latest_week = 22 → process [22], skip [23]
```

## 🛠️ Adding New Tests

When adding functionality to auto-update scripts:

1. **Add unit tests** for new helper functions
2. **Add integration tests** for new workflow scenarios  
3. **Add edge case tests** for boundary conditions
4. **Update mock data** if database schema changes
5. **Test error paths** for new failure modes

### Example Test Structure
```python
@patch('module.external_dependency')
def test_new_feature(self, mock_dependency):
    # Setup
    mock_dependency.return_value = expected_value
    
    # Execute
    result = function_under_test(input_data)
    
    # Assert
    self.assertEqual(result, expected_result)
    mock_dependency.assert_called_with(expected_args)
```

## 🎉 Benefits

1. **Confidence**: Changes won't break production workflows
2. **Documentation**: Tests serve as executable specifications
3. **Regression Prevention**: Catch issues before deployment
4. **Faster Development**: Quick feedback on code changes
5. **Maintainability**: Easy to understand and modify behavior

The test suite ensures that the NFL data workflows run reliably, handling edge cases gracefully and providing clear feedback when issues occur.

# Automation Workflows

This document describes the GitHub Actions workflows that automatically update NFL data in the Supabase database.

## 🔄 Workflow Overview

The T4LAPIs system includes four main automated workflows that handle different aspects of NFL data management:

| Workflow | Schedule | Purpose | Smart Features |
|----------|----------|---------|----------------|
| **Games Data** | 6x/week during season | Game schedules & results | Auto week detection, incremental updates |
| **Player Stats** | Weekly (Tuesdays) | Weekly player statistics | Gap detection, recent week updates |
| **Player Rosters** | Weekly (Wednesdays) | Roster changes | Trade/signing detection |
| **Team Data** | Monthly | Team information | Minimal changes, low frequency |

## 📊 Detailed Workflow Documentation

### 1. Games Data Update (`games-data-update.yml`)

**Schedule**: Thursday, Friday, Saturday, Sunday, Monday, Tuesday at 5:00 AM UTC

This is the most frequent workflow as NFL games happen throughout the week during the season.

#### Smart Features

- ✅ **Automatic Season Detection**: Determines current NFL season (September-August cycle)
- ✅ **Latest Week Detection**: Finds the most recent week in the database
- ✅ **Dual Update Strategy**: 
  - **Upserts** current week (catches score updates, time changes)
  - **Inserts** next week (upcoming games, scheduling changes)
- ✅ **Week Boundary Handling**: Stops at week 22 (end of playoffs)
- ✅ **Error Resilience**: Continues on partial failures

#### Manual Trigger Options

```yaml
inputs:
  season:
    description: 'NFL season year'
    required: false
    default: ''
  week:
    description: 'Specific week number'
    required: false
    default: ''
```

#### Example Usage

```bash
# Trigger manually for specific season/week
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/BigSlikTobi/T4LAPIs/actions/workflows/games-data-update.yml/dispatches \
  -d '{"ref":"main","inputs":{"season":"2024","week":"10"}}'
```

### 2. Player Weekly Stats Update (`player-weekly-stats-update.yml`)

**Schedule**: Tuesday at 6:00 AM UTC (after Monday Night Football)

Updates player statistics after all games for the week are completed.

#### Smart Features

- ✅ **Cross-Table Analysis**: Compares games table vs stats table to find gaps
- ✅ **Gap Detection**: Identifies weeks with games but no statistics
- ✅ **Recent Week Updates**: Reprocesses last 1-2 weeks for late corrections
- ✅ **Efficient Processing**: Only processes new/missing data
- ✅ **Stat Correction Handling**: Updates recent weeks to catch late corrections

#### Processing Logic

1. **Find Available Games**: Query games table for completed games
2. **Check Existing Stats**: Query stats table for processed weeks  
3. **Identify Gaps**: Find weeks with games but no stats
4. **Update Recent**: Always reprocess last 1-2 weeks
5. **Process Missing**: Load statistics for gap weeks

#### Manual Trigger Options

```yaml
inputs:
  years:
    description: 'NFL season years (space-separated)'
    required: false
    default: ''
  weeks:
    description: 'Specific weeks (space-separated)'
    required: false
    default: ''
```

### 3. Players Data Update (`players-data-update.yml`)

**Schedule**: Wednesday at 3:00 AM UTC (weekly roster updates)

Updates player roster information to catch trades, signings, releases, and IR moves.

#### Features

- ✅ **Roster Change Detection**: Captures mid-season transactions
- ✅ **Duplicate Resolution**: Handles player records intelligently
- ✅ **Team Normalization**: Manages historical team name changes
- ✅ **Weekly Frequency**: Balances freshness with API usage

#### Manual Trigger Options

```yaml
inputs:
  season:
    description: 'NFL season year'
    required: false
    default: ''
```

### 4. Teams Data Update (`teams-data-update.yml`)

**Schedule**: 1st of every month at 2:00 AM UTC

Updates team information. Team data rarely changes, so monthly updates are sufficient.

#### Features

- ✅ **Low Frequency**: Monthly updates for stable data
- ✅ **Team Metadata**: Colors, divisions, conferences
- ✅ **Historical Accuracy**: Maintains team information integrity

## 🤖 Auto-Update Script Intelligence

### Games Auto-Update Logic (`games_auto_update.py`)

```python
def smart_update_logic():
    # 1. Detect current NFL season
    current_season = get_current_nfl_season()
    
    # 2. Find latest week in database
    latest_week = get_latest_week_from_db(current_season)
    
    # 3. Update current week (upsert for changes)
    upsert_games(current_season, latest_week)
    
    # 4. Insert next week (new games)
    if latest_week < 22:
        insert_games(current_season, latest_week + 1)
```

**Season Detection Logic:**
- September 1 - August 31 defines NFL season year
- September 2024 = 2024 season
- February 2025 = 2024 season (same season continues)

### Player Stats Auto-Update Logic (`player_weekly_stats_auto_update.py`)

```python
def smart_stats_update():
    # 1. Get available games by week
    games_by_week = get_games_from_db()
    
    # 2. Get existing stats by week  
    stats_by_week = get_stats_from_db()
    
    # 3. Find gaps (games but no stats)
    gap_weeks = find_missing_weeks(games_by_week, stats_by_week)
    
    # 4. Always update recent weeks (corrections)
    recent_weeks = get_recent_weeks(2)
    
    # 5. Process all identified weeks
    for week in gap_weeks + recent_weeks:
        process_week_stats(week)
```

## 📈 Workflow Monitoring

### GitHub Actions Dashboard

Monitor workflow runs at:
`https://github.com/BigSlikTobi/T4LAPIs/actions`

### Key Metrics to Monitor

1. **Success Rate**: Workflow completion percentage
2. **Execution Time**: Duration of each workflow run
3. **Data Volume**: Number of records processed
4. **Error Patterns**: Common failure points

### Typical Execution Times

| Workflow | Typical Duration | Data Volume |
|----------|------------------|-------------|
| Games Data | 2-3 minutes | 16-32 games |
| Player Stats | 5-8 minutes | 1,000-2,000 records |
| Player Rosters | 3-5 minutes | 3,000+ players |
| Team Data | 1-2 minutes | 32 teams |

## 🚨 Error Handling & Recovery

### Common Error Scenarios

1. **API Rate Limits**:
   - **Symptom**: HTTP 429 errors from nfl_data_py
   - **Recovery**: Workflows include retry logic with exponential backoff
   
2. **Database Connection Issues**:
   - **Symptom**: Supabase connection timeouts
   - **Recovery**: Connection retry with fallback strategies

3. **Data Validation Failures**:
   - **Symptom**: Invalid data format or missing required fields
   - **Recovery**: Skip invalid records, log errors, continue processing

4. **Partial Data Availability**:
   - **Symptom**: Some weeks/players missing from NFL API
   - **Recovery**: Process available data, log gaps for manual review

### Recovery Strategies

1. **Automatic Retry**: Failed workflows automatically retry with delays
2. **Partial Success**: Continue processing even if some records fail
3. **Manual Intervention**: Failed workflows can be manually re-triggered
4. **Data Validation**: Comprehensive validation prevents corrupt data

## 🔧 Configuration & Secrets

### Required GitHub Secrets

```bash
SUPABASE_URL          # Your Supabase project URL
SUPABASE_KEY          # Supabase service role key (for write access)
```

### Environment Variables

```yaml
env:
  SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
  SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
  LOG_LEVEL: INFO
```

### Workflow Permissions

```yaml
permissions:
  contents: read
  actions: read
```

## 📊 Data Freshness Guarantees

| Data Type | Update Frequency | Freshness SLA |
|-----------|------------------|---------------|
| **Game Scores** | 6x/week | Within 6 hours of completion |
| **Player Stats** | Weekly | Within 24 hours of MNF |
| **Roster Changes** | Weekly | Within 3 days of transaction |
| **Team Info** | Monthly | Within 30 days of changes |

## 🔗 Integration Points

### Database Tables Updated

1. **`games`** - Game schedules and results
2. **`player_weekly_stats`** - Weekly player performance
3. **`players`** - Player roster information
4. **`teams`** - Team metadata

### API Dependencies

1. **nfl_data_py**: Primary data source
2. **Supabase**: Database backend
3. **GitHub Actions**: Orchestration platform

### Monitoring Integration

- **GitHub Actions logs**: Primary monitoring
- **Supabase dashboard**: Database metrics
- **Application logs**: Detailed execution traces

## 🎯 Future Enhancements

### Planned Improvements

1. **Dynamic Scheduling**: Adjust frequency based on NFL calendar
2. **Advanced Monitoring**: Custom metrics and alerting
3. **Data Quality Checks**: Automated validation and reporting
4. **Performance Optimization**: Reduce execution time and resource usage

### Potential New Workflows

1. **Injury Reports**: Daily injury status updates
2. **Fantasy Points**: Calculated fantasy statistics
3. **Advanced Analytics**: EPA, DVOA, and other metrics
4. **Historical Backfill**: Systematic historical data loading

For detailed implementation information, see: [Auto-Update Scripts](Auto_Update_Scripts.md)

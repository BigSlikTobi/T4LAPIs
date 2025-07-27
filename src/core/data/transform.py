"""
Data transformation module for converting NFL data to database schema.
This module provides classes to transform raw NFL data into the format expected by our database.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseDataTransformer(ABC):
    """
    Abstract base class for all NFL data transformers.
    
    Provides a common interface and shared functionality for transforming,
    validating, and sanitizing NFL data before database insertion.
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def transform(self, raw_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Main transformation pipeline that processes raw data through all stages.
        
        Args:
            raw_df: Raw DataFrame from nfl_data_py
            
        Returns:
            List of validated and sanitized records ready for database insertion
            
        Raises:
            ValueError: If required columns are missing or transformation fails
        """
        self.logger.info(f"Starting transformation pipeline for {len(raw_df)} records")
        
        # Step 1: Validate input columns
        self._validate_required_columns(raw_df)
        
        # Step 2: Transform records
        transformed_records = self._transform_records(raw_df)
        
        # Step 3: Validate and sanitize records
        validated_records = self._validate_and_sanitize_records(transformed_records)
        
        self.logger.info(f"Transformation pipeline completed: {len(validated_records)} valid records")
        return validated_records
    
    @abstractmethod
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for this data type."""
        pass
    
    @abstractmethod
    def _transform_single_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Transform a single row into a database record."""
        pass
    
    @abstractmethod
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a single transformed record."""
        pass
    
    def _validate_required_columns(self, raw_df: pd.DataFrame) -> None:
        """Validate that all required columns are present in the DataFrame."""
        required_columns = self._get_required_columns()
        missing_columns = [col for col in required_columns if col not in raw_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _transform_records(self, raw_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform all records in the DataFrame."""
        self.logger.info(f"Transforming {len(raw_df)} records")
        transformed_records = []
        
        for _, row in raw_df.iterrows():
            try:
                record = self._transform_single_record(row)
                if record is not None:
                    transformed_records.append(record)
            except Exception as e:
                self.logger.warning(f"Failed to transform record: {e}")
                continue
        
        self.logger.info(f"Successfully transformed {len(transformed_records)} records")
        return transformed_records
    
    def _validate_and_sanitize_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sanitize all transformed records."""
        self.logger.info(f"Validating and sanitizing {len(records)} records")
        validated_records = []
        
        for record in records:
            try:
                if self._validate_record(record):
                    sanitized_record = self._sanitize_record(record)
                    validated_records.append(sanitized_record)
                else:
                    self.logger.warning(f"Invalid record skipped: {record.get('id', 'unknown')}")
            except Exception as e:
                self.logger.error(f"Error validating/sanitizing record: {e}")
                continue
        
        self.logger.info(f"Validated and sanitized {len(validated_records)} records")
        return validated_records
    
    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a single record by cleaning up None/NaN values.
        Can be overridden by subclasses for custom sanitization.
        """
        sanitized = record.copy()
        for field in sanitized:
            if pd.isna(sanitized[field]) or sanitized[field] == 'nan':
                sanitized[field] = None
        return sanitized

class TeamDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL team data.
    
    Transforms raw team data from nfl_data_py into the format expected
    by our teams database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for team data."""
        return [
            'team_abbr', 'team_name', 'team_id', 'team_nick', 
            'team_conf', 'team_division', 'team_color', 'team_color2', 
            'team_color3', 'team_color4'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Transform a single team row into a database record."""
        team_record = {
            "team_abbr": row['team_abbr'],
            "team_name": row['team_name'], 
            "team_conference": row['team_conf'],
            "team_division": row['team_division'],
            "team_nfl_data_py_id": int(row['team_id']),  # Convert to int
            "team_nick": row['team_nick'],
            "team_color": row['team_color'],
            "team_color2": row['team_color2'],
            "team_color3": row['team_color3'],
            "team_color4": row['team_color4']
        }
        
        # Skip if critical data is missing
        if not team_record['team_abbr'] or not team_record['team_name']:
            self.logger.warning(f"Skipping team with missing abbr/name: {row.to_dict()}")
            return None
            
        return team_record
    
    def _validate_record(self, team_record: Dict[str, Any]) -> bool:
        """Validate a single team record for completeness and correctness."""
        try:
            # Required fields
            required_fields = ['team_abbr', 'team_name', 'team_conference', 
                              'team_division', 'team_nfl_data_py_id', 'team_nick']
            
            # Check all required fields are present and non-empty
            for field in required_fields:
                if field not in team_record or not team_record[field]:
                    self.logger.warning(f"Team record missing or empty required field '{field}': {team_record}")
                    return False
            
            # Validate team_abbr is 2-3 characters
            if not (2 <= len(team_record['team_abbr']) <= 3):
                self.logger.warning(f"Invalid team_abbr length: {team_record['team_abbr']}")
                return False
                
            # Validate conference
            if team_record['team_conference'] not in ['AFC', 'NFC']:
                self.logger.warning(f"Invalid conference: {team_record['team_conference']}")
                return False
                
            # Validate team_nfl_data_py_id is positive integer
            if not isinstance(team_record['team_nfl_data_py_id'], int) or team_record['team_nfl_data_py_id'] <= 0:
                self.logger.warning(f"Invalid team_nfl_data_py_id: {team_record['team_nfl_data_py_id']}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating team record: {e}")
            return False
    
    def _sanitize_record(self, team_record: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize team record with team-specific logic."""
        # Call parent sanitization first
        sanitized = super()._sanitize_record(team_record)
        
        # Clean up any potential None values in color fields
        for color_field in ['team_color', 'team_color2', 'team_color3', 'team_color4']:
            if sanitized.get(color_field) is None or pd.isna(sanitized.get(color_field)):
                sanitized[color_field] = '#000000'  # Default to black
        
        return sanitized

class PlayerDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL player data.
    
    Transforms raw player data from nfl_data_py into the format expected
    by our players database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for player data."""
        return [
            'player_id', 'player_name', 'first_name', 'last_name', 
            'birth_date', 'height', 'weight', 'college', 'position', 
            'rookie_year', 'season', 'football_name', 'jersey_number', 
            'status', 'team', 'years_exp', 'headshot_url'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Transform a single player row into a database record."""
        # Handle birth_date conversion
        birth_date = None
        if pd.notna(row.get('birth_date')):
            try:
                birth_date = pd.to_datetime(row['birth_date']).date().isoformat()
            except:
                birth_date = None
        
        # Handle draft information
        draft_year = None
        draft_round = None
        draft_pick = None
        draft_team = None
        
        if pd.notna(row.get('entry_year')):
            draft_year = int(row['entry_year'])
        elif pd.notna(row.get('rookie_year')):
            draft_year = int(row['rookie_year'])
            
        if pd.notna(row.get('draft_number')):
            draft_pick = int(row['draft_number'])
            # Estimate round from draft pick (approximate)
            if draft_pick <= 32:
                draft_round = 1
            elif draft_pick <= 64:
                draft_round = 2
            elif draft_pick <= 96:
                draft_round = 3
            elif draft_pick <= 128:
                draft_round = 4
            elif draft_pick <= 160:
                draft_round = 5
            elif draft_pick <= 192:
                draft_round = 6
            else:
                draft_round = 7
                
        if pd.notna(row.get('draft_club')):
            draft_team = str(row['draft_club'])
        
        # Determine position group
        position_group = self._determine_position_group(row.get('position'))
        
        player_record = {
            "player_id": str(row['player_id']),  # This is the gsis_id
            "full_name": str(row['player_name']) if pd.notna(row['player_name']) else None,
            "birthdate": birth_date,
            "height": int(row['height']) if pd.notna(row['height']) else None,
            "weight": int(row['weight']) if pd.notna(row['weight']) else None,
            "college": str(row['college']) if pd.notna(row['college']) else None,
            "position": str(row['position']) if pd.notna(row['position']) else None,
            "rookie_year": int(row['rookie_year']) if pd.notna(row['rookie_year']) else None,
            "last_active_season": int(row['season']) if pd.notna(row['season']) else None,
            "display_name": str(row['player_name']) if pd.notna(row['player_name']) else None,
            "common_first_name": str(row['first_name']) if pd.notna(row['first_name']) else None,
            "first_name": str(row['first_name']) if pd.notna(row['first_name']) else None,
            "last_name": str(row['last_name']) if pd.notna(row['last_name']) else None,
            "short_name": None,  # Not available in nfl_data_py
            "football_name": str(row['football_name']) if pd.notna(row['football_name']) else None,
            "suffix": None,  # Not available in nfl_data_py
            "position_group": position_group,
            "headshot": str(row['headshot_url']) if pd.notna(row['headshot_url']) else None,
            "college_conference": None,  # Not available in nfl_data_py
            "jersey_number": str(int(row['jersey_number'])) if pd.notna(row['jersey_number']) else None,
            "status": str(row['status']) if pd.notna(row['status']) else "Unknown",
            "latest_team": str(row['team']) if pd.notna(row['team']) else None,
            "years_of_experience": int(row['years_exp']) if pd.notna(row['years_exp']) else None,
            "draft_year": draft_year,
            "draft_round": draft_round,
            "draft_pick": draft_pick,
            "draft_team": draft_team
        }
        
        # Skip if critical data is missing
        if not player_record['player_id'] or not player_record['full_name']:
            self.logger.warning(f"Skipping player with missing ID/name: {row.to_dict()}")
            return None
            
        return player_record
    
    def _determine_position_group(self, position: Any) -> str:
        """Determine position group from player position."""
        if pd.isna(position):
            return "Unknown"
            
        pos = str(position).upper()
        if pos in ['QB']:
            return "Offense"
        elif pos in ['RB', 'FB', 'HB']:
            return "Offense"
        elif pos in ['WR']:
            return "Offense"
        elif pos in ['TE']:
            return "Offense"
        elif pos in ['OL', 'T', 'G', 'C', 'LT', 'RT', 'LG', 'RG']:
            return "Offense"
        elif pos in ['DL', 'DE', 'DT', 'NT']:
            return "Defense"
        elif pos in ['LB', 'ILB', 'OLB', 'MLB']:
            return "Defense"
        elif pos in ['DB', 'CB', 'S', 'SS', 'FS']:
            return "Defense"
        elif pos in ['K', 'P', 'LS']:
            return "Special Teams"
        else:
            return "Unknown"
    
    def _validate_record(self, player_record: Dict[str, Any]) -> bool:
        """Validate a single player record for completeness and correctness."""
        try:
            # Required fields
            required_fields = ['player_id', 'full_name']
            
            # Check critical required fields are present and non-empty
            for field in required_fields:
                if field not in player_record or not player_record[field]:
                    self.logger.warning(f"Player record missing or empty required field '{field}': {player_record}")
                    return False
            
            # Validate player_id format (should be like "00-0028830")
            player_id = player_record['player_id']
            if not isinstance(player_id, str) or len(player_id) < 8:
                self.logger.warning(f"Invalid player_id format: {player_id}")
                return False
            
            # Validate years if present
            if player_record.get('rookie_year') and not (1920 <= player_record['rookie_year'] <= 2030):
                self.logger.warning(f"Invalid rookie_year: {player_record['rookie_year']}")
                return False
                
            if player_record.get('last_active_season') and not (1920 <= player_record['last_active_season'] <= 2030):
                self.logger.warning(f"Invalid last_active_season: {player_record['last_active_season']}")
                return False
            
            # Validate height/weight if present
            if player_record.get('height') and not (60 <= player_record['height'] <= 90):  # inches
                self.logger.warning(f"Invalid height: {player_record['height']}")
                return False
                
            if player_record.get('weight') and not (150 <= player_record['weight'] <= 400):  # pounds
                self.logger.warning(f"Invalid weight: {player_record['weight']}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating player record: {e}")
            return False

class GameDataTransformer(BaseDataTransformer):
    """
    Handles transformation, validation, and sanitization of NFL game data.
    
    Transforms raw game data from nfl_data_py into the format expected
    by our games database table.
    """
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for game data."""
        return [
            'game_id', 'season', 'week', 'home_team', 'away_team', 
            'home_score', 'away_score', 'game_type', 'weekday', 
            'location', 'stadium', 'gameday', 'gametime'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Transform a single game row into a database record."""
        # Handle date and time conversion
        gameday = None
        if pd.notna(row.get('gameday')):
            try:
                gameday = pd.to_datetime(row['gameday']).date().isoformat()
            except:
                gameday = None
        
        # Handle time conversion
        gametime = None
        if pd.notna(row.get('gametime')):
            try:
                # Convert "20:20" format to "20:20:00"
                time_str = str(row['gametime'])
                if len(time_str) == 5 and ':' in time_str:  # "HH:MM" format
                    gametime = f"{time_str}:00"
                else:
                    gametime = time_str
            except:
                gametime = None
        
        game_record = {
            "game_id": str(row['game_id']),
            "season": int(row['season']) if pd.notna(row['season']) else None,
            "week": int(row['week']) if pd.notna(row['week']) else None,
            "home_team_abbr": str(row['home_team']) if pd.notna(row['home_team']) else None,
            "away_team_abbr": str(row['away_team']) if pd.notna(row['away_team']) else None,
            "home_score": int(row['home_score']) if pd.notna(row['home_score']) else None,
            "away_score": int(row['away_score']) if pd.notna(row['away_score']) else None,
            "game_type": str(row['game_type']) if pd.notna(row['game_type']) else None,
            "weekday": str(row['weekday']) if pd.notna(row['weekday']) else None,
            "location": str(row['location']) if pd.notna(row['location']) else None,
            "stadium": str(row['stadium']) if pd.notna(row['stadium']) else None,
            "gameday": gameday,
            "gametime": gametime
        }
        
        # Skip if critical data is missing
        if not game_record['game_id'] or not game_record['season'] or not game_record['week']:
            self.logger.warning(f"Skipping game with missing critical data: {row.to_dict()}")
            return None
            
        return game_record
    
    def _validate_record(self, game_record: Dict[str, Any]) -> bool:
        """Validate a single game record for completeness and correctness."""
        try:
            # Required fields
            required_fields = ['game_id', 'season', 'week', 'home_team_abbr', 'away_team_abbr']
            
            # Check critical required fields are present and non-empty
            for field in required_fields:
                if field not in game_record or not game_record[field]:
                    self.logger.warning(f"Game record missing or empty required field '{field}': {game_record}")
                    return False
            
            # Validate game_id format (should be like "2023_01_DET_KC")
            game_id = game_record['game_id']
            if not isinstance(game_id, str) or len(game_id) < 10:
                self.logger.warning(f"Invalid game_id format: {game_id}")
                return False
            
            # Validate season range
            if game_record.get('season') and not (1920 <= game_record['season'] <= 2030):
                self.logger.warning(f"Invalid season: {game_record['season']}")
                return False
                
            # Validate week range
            if game_record.get('week') and not (1 <= game_record['week'] <= 22):  # Including playoffs
                self.logger.warning(f"Invalid week: {game_record['week']}")
                return False
            
            # Validate team abbreviations
            for team_field in ['home_team_abbr', 'away_team_abbr']:
                team_abbr = game_record.get(team_field)
                if team_abbr and not (2 <= len(team_abbr) <= 3):
                    self.logger.warning(f"Invalid {team_field} length: {team_abbr}")
                    return False
            
            # Validate scores if present
            for score_field in ['home_score', 'away_score']:
                score = game_record.get(score_field)
                if score is not None and not (0 <= score <= 100):  # Reasonable score range
                    self.logger.warning(f"Invalid {score_field}: {score}")
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating game record: {e}")
            return False


class PlayerWeeklyStatsDataTransformer(BaseDataTransformer):
    """Transform weekly player stats data from nfl_data_py for database storage."""
    
    def __init__(self, db_manager):
        super().__init__()
        self.entity_type = "player_weekly_stats"
        self.db_manager = db_manager
    
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for weekly stats data."""
        return [
            'player_id', 'season', 'week', 'recent_team', 'opponent_team'
        ]
    
    def _transform_single_record(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Transform a single weekly stats row into a database record."""
        try:
            # Generate game_id using season, week, and teams
            game_id = self._generate_game_id(
                season=row.get('season'),
                week=row.get('week'),
                recent_team=row.get('recent_team'),
                opponent_team=row.get('opponent_team')
            )
            
            if not game_id:
                self.logger.warning(f"Could not generate game_id for record: {row.get('player_id', 'Unknown')}")
                return None
            
            # Transform the record to match database schema
            transformed = {
                'stat_id': f"{row.get('player_id', '')}_{game_id}_{row.get('week', '')}",
                'player_id': row.get('player_id'),
                'game_id': game_id,
                'completions': row.get('completions'),
                'passing_yards': row.get('passing_yards'),
                'passing_tds': row.get('passing_tds'),
                'interceptions': row.get('interceptions'),
                'carries': row.get('carries'),
                'rushing_yards': row.get('rushing_yards'),
                'rushing_tds': row.get('rushing_tds'),
                'receptions': row.get('receptions'),
                'receiving_yards': row.get('receiving_yards'),
                'receiving_tds': row.get('receiving_tds'),
                'fumbles_lost': row.get('fumbles_lost'),
                'fantasy_points': row.get('fantasy_points'),
                'attempts': row.get('attempts'),
                'sacks': row.get('sacks'),
                'sack_yards': row.get('sack_yards'),
                'sack_fumbles': row.get('sack_fumbles'),
                'sack_fumbles_loss': row.get('sack_fumbles_lost'),  # Note: mapping sack_fumbles_lost to sack_fumbles_loss
                'passing_yards_after_catch': row.get('passing_yards_after_catch'),
                'passing_first_downs': row.get('passing_first_downs'),
                'passing_epa': row.get('passing_epa'),
                'rushing_fumbles': row.get('rushing_fumbles'),
                'rushing_fumbles_lost': row.get('rushing_fumbles_lost'),
                'rushing_first_downs': row.get('rushing_first_downs'),
                'rushing_epa': row.get('rushing_epa'),
                'targets': row.get('targets'),
                'receiving_fumbles': row.get('receiving_fumbles'),
                'receiving_fumbles_loss': row.get('receiving_fumbles_lost'),  # Note: mapping receiving_fumbles_lost to receiving_fumbles_loss
                'receiving _first_downs': row.get('receiving_first_downs'),  # Note: keeping the space as in your schema
                'receiving_epa': row.get('receiving_epa'),
                'air_yards_share': row.get('air_yards_share')
            }
            
            # Convert numeric fields and handle None values
            numeric_fields = [
                'completions', 'passing_yards', 'passing_tds', 'interceptions', 'carries',
                'rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds',
                'fumbles_lost', 'fantasy_points', 'attempts', 'sacks', 'sack_yards', 'sack_fumbles',
                'sack_fumbles_loss', 'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
                'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa',
                'targets', 'receiving_fumbles', 'receiving_fumbles_loss', 'receiving _first_downs',
                'receiving_epa', 'air_yards_share'
            ]
            
            for field in numeric_fields:
                value = transformed.get(field)
                if pd.isna(value) or value is None:
                    transformed[field] = None
                else:
                    try:
                        transformed[field] = float(value) if '.' in str(value) else int(value)
                    except (ValueError, TypeError):
                        transformed[field] = None
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming weekly stats record: {e}")
            return None
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a transformed weekly stats record."""
        try:
            # Required fields
            required_fields = ['stat_id', 'player_id', 'game_id']
            for field in required_fields:
                if not record.get(field):
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate fields that should be non-negative (attempts, completions, etc.)
            # Note: yards can be negative due to sacks, tackles for loss, etc.
            non_negative_fields = [
                'completions', 'attempts', 'passing_tds', 'carries', 'rushing_tds', 
                'receptions', 'targets', 'receiving_tds'
            ]
            
            for field in non_negative_fields:
                value = record.get(field)
                if value is not None and value < 0:
                    self.logger.warning(f"Negative value for {field}: {value}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating weekly stats record: {e}")
            return False
    
    def fetch_data(self, **kwargs):
        """Fetch weekly player stats data from nfl_data_py."""
        try:
            import nfl_data_py as nfl
            
            # Extract parameters
            years = kwargs.get('years', [2024])
            weeks = kwargs.get('weeks', None)
            
            if not isinstance(years, list):
                years = [years]
                
            self.logger.info(f"Fetching weekly stats data for years: {years}, weeks: {weeks}")
            
            # Fetch weekly data
            data = nfl.import_weekly_data(years=years, columns=None)
            
            if data is None or data.empty:
                self.logger.warning("No weekly stats data retrieved")
                return pd.DataFrame()
            
            # Filter by weeks if specified
            if weeks is not None:
                if not isinstance(weeks, list):
                    weeks = [weeks]
                data = data[data['week'].isin(weeks)]
                
            self.logger.info(f"Retrieved {len(data)} weekly stats records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching weekly stats data: {e}")
            return pd.DataFrame()
    
    def transform_record(self, record):
        """Legacy method for compatibility - delegates to _transform_single_record."""
        if isinstance(record, dict):
            # Convert dict to pandas Series for consistency
            record = pd.Series(record)
        return self._transform_single_record(record)
    
    def validate_record(self, record):
        """Legacy method for compatibility - delegates to _validate_record."""
        return self._validate_record(record)
    
    def _generate_game_id(self, season, week, recent_team, opponent_team):
        """Generate game_id by looking up the corresponding game."""
        try:
            if not all([season, week, recent_team, opponent_team]):
                return None
            
            # Query the games table to find the matching game using Supabase API
            # Check if recent_team is home and opponent_team is away
            response1 = self.db_manager.table('games').select('game_id').eq('season', season).eq('week', week).eq('home_team_abbr', recent_team).eq('away_team_abbr', opponent_team).execute()
            
            if response1.data and len(response1.data) > 0:
                return response1.data[0]['game_id']
            
            # Check if recent_team is away and opponent_team is home
            response2 = self.db_manager.table('games').select('game_id').eq('season', season).eq('week', week).eq('home_team_abbr', opponent_team).eq('away_team_abbr', recent_team).execute()
            
            if response2.data and len(response2.data) > 0:
                return response2.data[0]['game_id']
            
            self.logger.warning(f"No game found for season={season}, week={week}, teams={recent_team} vs {opponent_team}")
            return None
                
        except Exception as e:
            self.logger.error(f"Error generating game_id: {e}")
            return None


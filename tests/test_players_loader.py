"""Tests for the players data loader."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.core.data.loaders.players import PlayersDataLoader


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client."""
    mock_client = Mock()
    mock_table = Mock()
    mock_client.table.return_value = mock_table
    return mock_client


@pytest.fixture
def players_loader(mock_supabase_client):
    """Players data loader instance with mocked client."""
    with patch('src.core.data.loaders.players.get_supabase_client', return_value=mock_supabase_client):
        return PlayersDataLoader()


@pytest.fixture
def sample_raw_player_data():
    """Sample raw player data from nfl_data_py."""
    return pd.DataFrame([
        {
            'player_id': 'P001',
            'player_name': 'Patrick Mahomes',
            'team': 'KC',
            'position': 'QB',
            'birth_date': '1995-09-17',
            'draft_year': '2017',
            'draft_round': '1',
            'draft_pick': '10',
            'weight': '230',
            'height': '74',
            'college': 'Texas Tech',
            'years_of_experience': '7'
        },
        {
            'player_id': 'P002',
            'player_name': 'Travis Kelce',
            'team': 'KC',
            'position': 'TE',
            'birth_date': '1989-10-05',
            'draft_year': '2013',
            'draft_round': '3',
            'draft_pick': '63',
            'weight': '250',
            'height': '77',
            'college': 'Cincinnati',
            'years_of_experience': '11'
        }
    ])


@pytest.fixture
def sample_transformed_player_data():
    """Sample transformed player data."""
    return [
        {
            'player_id': 'P001',
            'player_name': 'Patrick Mahomes',
            'latest_team': 'KC',
            'position': 'QB',
            'position_group': 'QB',
            'birth_date': '1995-09-17',
            'draft_team': 'KC',
            'draft_year': 2017,
            'draft_round': 1,
            'draft_pick': 10,
            'weight': 230,
            'height_inches': 74,
            'college': 'Texas Tech',
            'years_of_experience': 7
        },
        {
            'player_id': 'P002',
            'player_name': 'Travis Kelce',
            'latest_team': 'KC',
            'position': 'TE',
            'position_group': 'Receiving',
            'birth_date': '1989-10-05',
            'draft_team': 'KC',
            'draft_year': 2013,
            'draft_round': 3,
            'draft_pick': 63,
            'weight': 250,
            'height_inches': 77,
            'college': 'Cincinnati',
            'years_of_experience': 11
        }
    ]


class TestPlayersDataLoader:
    """Test cases for PlayersDataLoader."""
    
    def test_init(self, mock_supabase_client):
        """Test loader initialization."""
        with patch('src.core.data.loaders.players.get_supabase_client', return_value=mock_supabase_client):
            loader = PlayersDataLoader()
            assert loader.supabase == mock_supabase_client
            assert loader.table_name == "players"
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    @patch('src.core.data.loaders.players.PlayerDataTransformer')
    def test_load_players_success(self, mock_transformer_class, mock_fetch, 
                                 players_loader, sample_raw_player_data, 
                                 sample_transformed_player_data):
        """Test successful player data loading."""
        # Setup mocks
        mock_fetch.return_value = sample_raw_player_data
        
        # Mock the transformer instance and its transform method
        mock_transformer = Mock()
        mock_transformer.transform.return_value = sample_transformed_player_data
        mock_transformer_class.return_value = mock_transformer
        
        # Mock database operations
        players_loader.supabase.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=sample_transformed_player_data
        )
        
        # Execute
        result = players_loader.load_players(season=2024)
        
        # Verify
        assert result["success"] is True
        assert result["season"] == 2024
        assert result["total_fetched"] == 2
        assert result["total_validated"] == 2
        assert result["upsert_result"]["affected_rows"] == 2
        
        # Verify function calls
        mock_fetch.assert_called_once_with([2024])
        mock_transformer_class.assert_called_once()
        mock_transformer.transform.assert_called_once()
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    def test_load_players_no_data(self, mock_fetch, players_loader):
        """Test handling when no player data is found."""
        mock_fetch.return_value = pd.DataFrame()
        
        result = players_loader.load_players(season=2024)
        
        assert result["success"] is False
        assert "No data found" in result["message"]
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    @patch('src.core.data.loaders.players.PlayerDataTransformer')
    def test_load_players_dry_run(self, mock_transformer_class, mock_fetch, 
                                 players_loader, sample_raw_player_data, 
                                 sample_transformed_player_data):
        """Test dry run mode."""
        # Setup mocks
        mock_fetch.return_value = sample_raw_player_data
        
        # Mock the transformer instance and its transform method
        mock_transformer = Mock()
        mock_transformer.transform.return_value = sample_transformed_player_data
        mock_transformer_class.return_value = mock_transformer
        
        # Execute dry run
        result = players_loader.load_players(season=2024, dry_run=True)
        
        # Verify
        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["would_upsert"] == 2
        assert "sample_record" in result
        
        # Verify no database operations
        players_loader.supabase.table.assert_not_called()
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    @patch('src.core.data.loaders.players.PlayerDataTransformer')
    def test_load_players_with_clear(self, mock_transformer_class, mock_fetch, 
                                    players_loader, sample_raw_player_data, 
                                    sample_transformed_player_data):
        """Test loading with table clearing."""
        # Setup mocks
        mock_fetch.return_value = sample_raw_player_data
        
        # Mock the transformer instance and its transform method
        mock_transformer = Mock()
        mock_transformer.transform.return_value = sample_transformed_player_data
        mock_transformer_class.return_value = mock_transformer
        
        # Mock clear operation
        players_loader.supabase.table.return_value.delete.return_value.neq.return_value.execute.return_value = Mock()
        players_loader.supabase.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=sample_transformed_player_data
        )
        
        # Execute with clear
        result = players_loader.load_players(season=2024, clear_table=True)
        
        # Verify
        assert result["success"] is True
        assert result["cleared_table"] is True
        
        # Verify clear was called
        players_loader.supabase.table.return_value.delete.assert_called_once()
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    @patch('src.core.data.loaders.players.PlayerDataTransformer')
    def test_load_players_validation_failures(self, mock_transformer_class, 
                                             mock_fetch, players_loader, 
                                             sample_raw_player_data, 
                                             sample_transformed_player_data):
        """Test handling of validation failures."""
        # Setup mocks - simulate transformer returning only one valid player
        mock_fetch.return_value = sample_raw_player_data
        
        # Mock transformer to return only one valid player (simulating validation failure)
        mock_transformer = Mock()
        mock_transformer.transform.return_value = [sample_transformed_player_data[1]]  # Only second player
        mock_transformer_class.return_value = mock_transformer
        
        # Mock database operations
        players_loader.supabase.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=[sample_transformed_player_data[1]]  # Only second player
        )
        
        # Execute
        result = players_loader.load_players(season=2024)
        
        # Verify
        assert result["success"] is True
        assert result["total_fetched"] == 2
        assert result["total_validated"] == 1  # Only one passed validation
    
    @patch('src.core.data.loaders.players.fetch_player_data')
    def test_load_players_exception_handling(self, mock_fetch, players_loader):
        """Test exception handling during load."""
        mock_fetch.side_effect = Exception("Database connection failed")
        
        result = players_loader.load_players(season=2024)
        
        assert result["success"] is False
        assert "Database connection failed" in result["error"]
    
    def test_clear_table(self, players_loader):
        """Test table clearing functionality."""
        # Mock successful clear
        players_loader.supabase.table.return_value.delete.return_value.neq.return_value.execute.return_value = Mock()
        
        # Should not raise exception
        players_loader._clear_table()
        
        # Verify delete was called
        players_loader.supabase.table.assert_called_with("players")
        players_loader.supabase.table.return_value.delete.assert_called_once()
    
    def test_clear_table_exception(self, players_loader):
        """Test exception handling during table clear."""
        # Mock exception during clear
        players_loader.supabase.table.return_value.delete.return_value.neq.return_value.execute.side_effect = Exception("Clear failed")
        
        with pytest.raises(Exception, match="Clear failed"):
            players_loader._clear_table()
    
    def test_upsert_players_success(self, players_loader, sample_transformed_player_data):
        """Test successful player upsert."""
        # Mock successful upsert
        players_loader.supabase.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=sample_transformed_player_data
        )
        
        result = players_loader._upsert_players(sample_transformed_player_data)
        
        assert result["operation"] == "upsert"
        assert result["affected_rows"] == 2
        assert result["status"] == "success"
        
        # Verify upsert was called with correct parameters
        players_loader.supabase.table.return_value.upsert.assert_called_once_with(
            sample_transformed_player_data,
            on_conflict="player_id"
        )
    
    def test_upsert_players_exception(self, players_loader, sample_transformed_player_data):
        """Test exception handling during upsert."""
        # Mock exception during upsert
        players_loader.supabase.table.return_value.upsert.return_value.execute.side_effect = Exception("Upsert failed")
        
        with pytest.raises(Exception, match="Upsert failed"):
            players_loader._upsert_players(sample_transformed_player_data)
    
    def test_get_player_count_success(self, players_loader):
        """Test getting player count."""
        # Mock successful count
        players_loader.supabase.table.return_value.select.return_value.execute.return_value = Mock(count=150)
        
        count = players_loader.get_player_count()
        
        assert count == 150
        players_loader.supabase.table.return_value.select.assert_called_once_with("player_id", count="exact")
    
    def test_get_player_count_exception(self, players_loader):
        """Test exception handling when getting player count."""
        # Mock exception
        players_loader.supabase.table.return_value.select.return_value.execute.side_effect = Exception("Count failed")
        
        count = players_loader.get_player_count()
        
        assert count == 0
    
    def test_get_players_by_team_success(self, players_loader, sample_transformed_player_data):
        """Test getting players by team."""
        # Mock successful query
        kc_players = [p for p in sample_transformed_player_data if p['latest_team'] == 'KC']
        players_loader.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
            data=kc_players
        )
        
        result = players_loader.get_players_by_team('KC')
        
        assert len(result) == 2
        assert all(p['latest_team'] == 'KC' for p in result)
        
        # Verify query
        players_loader.supabase.table.return_value.select.return_value.eq.assert_called_once_with("latest_team", "KC")
    
    def test_get_players_by_team_exception(self, players_loader):
        """Test exception handling when getting players by team."""
        # Mock exception
        players_loader.supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = Exception("Query failed")
        
        result = players_loader.get_players_by_team('KC')
        
        assert result == []


class TestPlayersLoaderCLI:
    """Test cases for the CLI interface."""
    
    def test_cli_basic_usage(self):
        """Test basic CLI usage."""
        # Test would require sys.argv mocking for full CLI test
        # This tests the loader initialization
        from src.core.data.loaders.players import PlayersDataLoader
        from unittest.mock import Mock
        
        mock_client = Mock()
        with patch('src.core.data.loaders.players.get_supabase_client', return_value=mock_client):
            loader = PlayersDataLoader()
            assert loader.supabase == mock_client
    
    def test_cli_dry_run(self):
        """Test CLI dry run functionality."""
        # This would test the actual CLI with argument parsing
        # For now, verify the loader works correctly with dry_run
        from src.core.data.loaders.players import PlayersDataLoader
        from unittest.mock import Mock
        
        mock_client = Mock()
        with patch('src.core.data.loaders.players.get_supabase_client', return_value=mock_client):
            loader = PlayersDataLoader()
        
        # Test that dry_run parameter exists and can be used
        # (actual dry run testing done in main test class)
        assert hasattr(loader, 'load_players')
        assert loader.table_name == "players"

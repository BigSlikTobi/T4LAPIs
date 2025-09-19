"""Tests for the rosters CLI interface."""

import unittest
from unittest.mock import MagicMock, patch

import scripts.data_loaders.rosters_cli as rosters_cli


class TestRostersCLI(unittest.TestCase):
    """CLI interaction tests."""

    @patch("scripts.data_loaders.rosters_cli.setup_cli_logging")
    @patch("scripts.data_loaders.rosters_cli.print_results")
    @patch("scripts.data_loaders.rosters_cli.RostersDataLoader")
    @patch("scripts.data_loaders.rosters_cli.setup_cli_parser")
    def test_main_success(self, mock_parser, mock_loader_cls, mock_print_results, mock_setup_logging):
        """CLI should call loader with parsed arguments."""
        mock_args = MagicMock()
        mock_args.season = 2025
        mock_args.dry_run = False
        mock_args.clear = False
        mock_args.include_records = False
        mock_args.verbose = False
        mock_args.log_level = "INFO"

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_loader = MagicMock()
        mock_loader.load_data.return_value = {"success": True, "version": 4}
        mock_loader_cls.return_value = mock_loader

        exit_code = rosters_cli.main()

        self.assertEqual(exit_code, 0)
        mock_loader.load_data.assert_called_once_with(
            season=2025,
            dry_run=False,
            clear_table=False,
            include_records=False,
        )
        mock_print_results.assert_called_once()
        mock_setup_logging.assert_called_once_with(mock_args)

    @patch("scripts.data_loaders.rosters_cli.setup_cli_logging")
    @patch("scripts.data_loaders.rosters_cli.print_results")
    @patch("scripts.data_loaders.rosters_cli.RostersDataLoader")
    @patch("scripts.data_loaders.rosters_cli.setup_cli_parser")
    def test_main_logs_skipped_records(self, mock_parser, mock_loader_cls, mock_print_results, mock_setup_logging):
        """Skipped entries should be rendered in console output."""
        mock_args = MagicMock()
        mock_args.season = 2025
        mock_args.dry_run = False
        mock_args.clear = False
        mock_args.include_records = False
        mock_args.verbose = False
        mock_args.log_level = "INFO"

        mock_parser_instance = MagicMock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance

        mock_loader = MagicMock()
        mock_loader.load_data.return_value = {
            "success": True,
            "version": 7,
            "skipped": [{"reason": "duplicate_slug", "slug": "wr-player-kc-na"}],
        }
        mock_loader_cls.return_value = mock_loader

        with patch("builtins.print") as mock_print:
            exit_code = rosters_cli.main()

        self.assertEqual(exit_code, 0)
        printed_messages = [call.args[0] for call in mock_print.call_args_list if call.args]
        self.assertIn("Skipped records: 1", printed_messages)
        detail_lines = [msg for msg in printed_messages if msg.startswith("  - [1]")]
        self.assertTrue(detail_lines)


if __name__ == "__main__":
    unittest.main()

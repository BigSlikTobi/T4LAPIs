"""Tests for CLI utilities."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import argparse
import sys
from io import StringIO

from src.core.utils.cli import setup_cli_parser, setup_cli_logging, print_results, handle_cli_errors


class TestCLIUtilities(unittest.TestCase):
    """Test cases for CLI utility functions."""

    def test_setup_cli_parser(self):
        """Test CLI parser setup."""
        parser = setup_cli_parser("Test description")
        
        self.assertIsInstance(parser, argparse.ArgumentParser)
        # Test that common arguments are added
        args = parser.parse_args(['--dry-run', '--verbose'])
        self.assertTrue(args.dry_run)
        self.assertTrue(args.verbose)

    def test_setup_cli_parser_with_custom_args(self):
        """Test CLI parser with additional custom arguments."""
        parser = setup_cli_parser("Test description")
        parser.add_argument("season", type=int, help="Season year")
        
        args = parser.parse_args(['2024', '--dry-run'])
        self.assertEqual(args.season, 2024)
        self.assertTrue(args.dry_run)

    @patch('src.core.utils.cli.setup_logging')
    def test_setup_cli_logging_default(self, mock_setup_logging):
        """Test CLI logging setup with default level.""" 
        args = Mock()
        args.verbose = False
        args.log_level = 'INFO'
        
        setup_cli_logging(args)
        
        mock_setup_logging.assert_called_once_with(level='INFO')

    @patch('src.core.utils.cli.setup_logging')
    def test_setup_cli_logging_verbose(self, mock_setup_logging):
        """Test CLI logging setup with verbose mode."""
        args = Mock()
        args.verbose = True
        args.log_level = 'INFO'
        
        setup_cli_logging(args)
        
        mock_setup_logging.assert_called_once_with(level='DEBUG')

    @patch('src.core.utils.cli.setup_logging')
    def test_setup_cli_logging_with_level(self, mock_setup_logging):
        """Test CLI logging setup with specific level."""
        args = Mock()
        args.verbose = False
        args.log_level = 'WARNING'
        
        setup_cli_logging(args)
        
        mock_setup_logging.assert_called_once_with(level='WARNING')

    @patch('src.core.utils.cli.setup_logging')
    def test_setup_cli_logging_with_defaults(self, mock_setup_logging):
        """Test CLI logging setup with defaults."""
        args = Mock()
        args.verbose = False
        args.log_level = 'INFO'  # Default should be INFO, not None
        
        setup_cli_logging(args)
        
        mock_setup_logging.assert_called_once_with(level='INFO')

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_results_success(self, mock_stdout):
        """Test printing successful results."""
        result = {
            'success': True,
            'total_fetched': 10,
            'total_validated': 9,
            'upsert_result': {'affected_rows': 8}
        }
        
        print_results(result, "test operation", dry_run=False)
        
        output = mock_stdout.getvalue()
        self.assertIn("✅ Successfully completed test operation", output)
        self.assertIn("Fetched: 10", output)
        self.assertIn("Validated: 9", output)
        self.assertIn("Upserted: 8 records", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_results_failure(self, mock_stdout):
        """Test printing failed results."""
        result = {
            'success': False,
            'error': 'Something went wrong'
        }
        
        print_results(result, "test operation", dry_run=False)
        
        output = mock_stdout.getvalue()
        self.assertIn("❌ Test operation failed: Something went wrong", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_results_dry_run(self, mock_stdout):
        """Test printing dry run results."""
        result = {
            'success': True,
            'dry_run': True,
            'would_upsert': 9,
            'would_clear': False
        }
        
        print_results(result, "test operation", dry_run=True)
        
        output = mock_stdout.getvalue()
        self.assertIn("DRY RUN - Would perform test operation", output)
        self.assertIn("Would upsert 9 records", output)
        self.assertIn("Would clear table: False", output)

    def test_handle_cli_errors_decorator_success(self):
        """Test CLI error handler with successful function."""
        @handle_cli_errors
        def test_function():
            return True
        
        result = test_function()
        self.assertEqual(result, 0)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_cli_errors_decorator_exception(self, mock_stdout):
        """Test CLI error handler with exception."""
        @handle_cli_errors
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        
        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("Unexpected error: Test error", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_handle_cli_errors_decorator_keyboard_interrupt(self, mock_stdout):
        """Test CLI error handler with keyboard interrupt."""
        @handle_cli_errors
        def test_function():
            raise KeyboardInterrupt()
        
        result = test_function()
        
        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        self.assertIn("Operation cancelled by user", output)
        output = mock_stdout.getvalue()
        self.assertIn("Operation cancelled by user", output)


if __name__ == '__main__':
    unittest.main()

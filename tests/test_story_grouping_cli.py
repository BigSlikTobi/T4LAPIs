"""Tests for story grouping CLI functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from io import StringIO
import tempfile
import textwrap

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.pipeline_cli as cli


class TestStoryGroupingCLI(unittest.TestCase):
    """Test cases for story grouping CLI commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_file = self.temp_dir / "test_feeds.yaml"
        
        # Create a minimal test config
        config_content = textwrap.dedent("""
            defaults:
              user_agent: "test"
              timeout_seconds: 5
              max_parallel_fetches: 2
              enable_story_grouping: true
              story_grouping_max_parallelism: 2
            sources:
              - name: test_source
                type: rss
                url: "https://example.com/rss"
                enabled: true
                publisher: "Test Publisher"
        """).strip()
        
        self.config_file.write_text(config_content)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_parser_includes_story_grouping_commands(self):
        """Test that the parser includes story grouping commands."""
        parser = cli.build_parser()
        
        # Test that we can parse story grouping commands
        with self.assertRaises(SystemExit):  # argparse exits on help
            parser.parse_args(["group-stories", "--help"])
            
        with self.assertRaises(SystemExit):
            parser.parse_args(["group-status", "--help"])
            
        with self.assertRaises(SystemExit):
            parser.parse_args(["group-backfill", "--help"])
            
        with self.assertRaises(SystemExit):
            parser.parse_args(["group-report", "--help"])

    def test_run_command_with_story_grouping_flag(self):
        """Test run command with --enable-story-grouping flag."""
        parser = cli.build_parser()
        args = parser.parse_args([
            "run", 
            "--config", str(self.config_file),
            "--enable-story-grouping",
            "--dry-run"
        ])
        
        self.assertEqual(args.cmd, "run")
        self.assertTrue(args.enable_story_grouping)
        self.assertTrue(args.dry_run)

    def test_group_stories_command_parsing(self):
        """Test group-stories command argument parsing."""
        parser = cli.build_parser()
        args = parser.parse_args([
            "group-stories",
            "--config", str(self.config_file),
            "--max-stories", "50",
            "--max-parallelism", "2",
            "--dry-run",
            "--reprocess"
        ])
        
        self.assertEqual(args.cmd, "group-stories")
        self.assertEqual(args.max_stories, 50)
        self.assertEqual(args.max_parallelism, 2)
        self.assertTrue(args.dry_run)
        self.assertTrue(args.reprocess)

    def test_group_backfill_command_parsing(self):
        """Test group-backfill command argument parsing."""
        parser = cli.build_parser()
        args = parser.parse_args([
            "group-backfill",
            "--config", str(self.config_file),
            "--batch-size", "25",
            "--max-batches", "10",
            "--resume-from", "story_123"
        ])
        
        self.assertEqual(args.cmd, "group-backfill")
        self.assertEqual(args.batch_size, 25)
        self.assertEqual(args.max_batches, 10)
        self.assertEqual(args.resume_from, "story_123")

    def test_group_report_command_parsing(self):
        """Test group-report command argument parsing."""
        parser = cli.build_parser()
        args = parser.parse_args([
            "group-report",
            "--config", str(self.config_file),
            "--format", "json",
            "--days-back", "14"
        ])
        
        self.assertEqual(args.cmd, "group-report")
        self.assertEqual(args.format, "json")
        self.assertEqual(args.days_back, 14)

    @patch('scripts.pipeline_cli._build_storage')
    @patch('scripts.pipeline_cli._build_story_grouping_orchestrator')
    def test_cmd_group_stories_dry_run(self, mock_orchestrator, mock_storage):
        """Test group-stories command in dry-run mode."""
        # Setup mocks
        mock_storage.return_value = Mock()
        mock_orchestrator_instance = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Mock the orchestrator settings
        mock_settings = Mock()
        mock_orchestrator_instance.settings = mock_settings
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = cli.cmd_group_stories(
                str(self.config_file),
                max_stories=10,
                max_parallelism=2,
                dry_run=True,
                reprocess=False
            )
            
            # Should return 0 for success
            self.assertEqual(result, 0)
            
            # Check that storage was built with dry_run=True
            mock_storage.assert_called_once_with(True)
            
            # Check that orchestrator was initialized
            mock_orchestrator.assert_called_once()
            
            # Verify output contains expected messages
            output = captured_output.getvalue()
            self.assertIn("Running story grouping", output)
            self.assertIn("dry_run=True", output)
            
        finally:
            sys.stdout = old_stdout

    @patch('scripts.pipeline_cli._build_storage')
    def test_cmd_group_status(self, mock_storage):
        """Test group-status command."""
        mock_storage.return_value = Mock()
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = cli.cmd_group_status(str(self.config_file))
            
            self.assertEqual(result, 0)
            
            output = captured_output.getvalue()
            self.assertIn("Story Grouping Status", output)
            self.assertIn("Database connectivity", output)
            
        finally:
            sys.stdout = old_stdout

    @patch('scripts.pipeline_cli._build_storage')
    @patch('scripts.pipeline_cli._build_story_grouping_orchestrator')
    def test_cmd_group_backfill_dry_run(self, mock_orchestrator, mock_storage):
        """Test group-backfill command in dry-run mode."""
        mock_storage.return_value = Mock()
        mock_orchestrator.return_value = Mock()
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = cli.cmd_group_backfill(
                str(self.config_file),
                batch_size=25,
                max_batches=5,
                dry_run=True,
                resume_from=None
            )
            
            self.assertEqual(result, 0)
            
            output = captured_output.getvalue()
            self.assertIn("Running story grouping backfill", output)
            self.assertIn("batch_size=25", output)
            self.assertIn("max_batches=5", output)
            
        finally:
            sys.stdout = old_stdout

    @patch('scripts.pipeline_cli._build_storage')
    def test_cmd_group_report_text_format(self, mock_storage):
        """Test group-report command with text format."""
        mock_storage.return_value = Mock()
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = cli.cmd_group_report(
                str(self.config_file),
                format_type="text",
                days_back=7
            )
            
            self.assertEqual(result, 0)
            
            output = captured_output.getvalue()
            self.assertIn("Story Grouping Analytics Report", output)
            self.assertIn("Report Period: 7 days", output)
            
        finally:
            sys.stdout = old_stdout

    @patch('scripts.pipeline_cli._build_storage')
    def test_cmd_group_report_json_format(self, mock_storage):
        """Test group-report command with JSON format."""
        mock_storage.return_value = Mock()
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            result = cli.cmd_group_report(
                str(self.config_file),
                format_type="json",
                days_back=14
            )
            
            self.assertEqual(result, 0)
            
            output = captured_output.getvalue()
            # Should contain JSON output
            self.assertIn('"report_generated":', output)
            self.assertIn('"days": 14', output)
            
        finally:
            sys.stdout = old_stdout

    def test_main_routes_to_group_commands(self):
        """Test that main() correctly routes to group commands."""
        # Test group-stories
        with patch('scripts.pipeline_cli.cmd_group_stories') as mock_cmd:
            mock_cmd.return_value = 0
            result = cli.main([
                "group-stories", 
                "--config", str(self.config_file),
                "--dry-run"
            ])
            self.assertEqual(result, 0)
            mock_cmd.assert_called_once()

        # Test group-status
        with patch('scripts.pipeline_cli.cmd_group_status') as mock_cmd:
            mock_cmd.return_value = 0
            result = cli.main([
                "group-status",
                "--config", str(self.config_file)
            ])
            self.assertEqual(result, 0)
            mock_cmd.assert_called_once()

        # Test group-backfill
        with patch('scripts.pipeline_cli.cmd_group_backfill') as mock_cmd:
            mock_cmd.return_value = 0
            result = cli.main([
                "group-backfill",
                "--config", str(self.config_file),
                "--dry-run"
            ])
            self.assertEqual(result, 0)
            mock_cmd.assert_called_once()

        # Test group-report
        with patch('scripts.pipeline_cli.cmd_group_report') as mock_cmd:
            mock_cmd.return_value = 0
            result = cli.main([
                "group-report",
                "--config", str(self.config_file)
            ])
            self.assertEqual(result, 0)
            mock_cmd.assert_called_once()

    @patch('os.environ')
    def test_run_command_sets_environment_variables(self, mock_environ):
        """Test that run command sets appropriate environment variables."""
        mock_environ.__setitem__ = Mock()
        
        with patch('scripts.pipeline_cli._build_storage') as mock_storage, \
             patch('scripts.pipeline_cli.ConfigManager') as mock_config_manager, \
             patch('scripts.pipeline_cli.NFLNewsPipeline') as mock_pipeline:
            
            mock_storage.return_value = Mock()
            mock_config_manager.return_value = Mock()
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            # Mock pipeline run result
            mock_summary = Mock()
            mock_summary.sources = 1
            mock_summary.fetched_items = 10
            mock_summary.filtered_in = 8
            mock_summary.inserted = 5
            mock_summary.updated = 3
            mock_summary.errors = 0
            mock_summary.store_errors = 0
            mock_summary.duration_ms = 1000
            mock_pipeline_instance.run.return_value = mock_summary
            
            result = cli.cmd_run(
                str(self.config_file),
                source=None,
                dry_run=True,
                disable_llm=True,
                llm_timeout=30.0,
                enable_story_grouping=True
            )
            
            self.assertEqual(result, 0)
            
            # Verify environment variables were set
            expected_calls = [
                unittest.mock.call("NEWS_PIPELINE_DISABLE_LLM", "1"),
                unittest.mock.call("OPENAI_TIMEOUT", "30.0"),
                unittest.mock.call("NEWS_PIPELINE_ENABLE_STORY_GROUPING", "1")
            ]
            
            for call in expected_calls:
                self.assertIn(call, mock_environ.__setitem__.call_args_list)


class TestStoryGroupingBatchProcessor(unittest.TestCase):
    """Test cases for the standalone batch processor."""

    def test_batch_processor_script_syntax(self):
        """Test that the batch processor script has valid syntax."""
        script_path = ROOT / "scripts" / "story_grouping_batch_processor.py"
        self.assertTrue(script_path.exists())
        
        # Test that the script compiles without syntax errors
        import py_compile
        try:
            py_compile.compile(str(script_path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"Batch processor script has syntax errors: {e}")

    def test_batch_processor_help(self):
        """Test that the batch processor shows help correctly."""
        # This would test the argparse help functionality
        # In a real test environment, you could run the script with --help
        # and check the output
        pass  # Placeholder for integration test


if __name__ == '__main__':
    unittest.main()
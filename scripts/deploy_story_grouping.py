#!/usr/bin/env python3
"""
Story Grouping Feature Deployment and Configuration Script
==========================================================

This script provides comprehensive deployment and configuration management
for the NFL News Pipeline Story Grouping feature.

Key Features:
- Environment validation and setup
- Configuration validation and generation
- Database schema verification
- Deployment readiness checks
- Interactive setup wizard

Usage:
    python scripts/deploy_story_grouping.py [command] [options]

Commands:
    validate    - Validate deployment environment and configuration
    setup       - Interactive setup wizard for first-time deployment
    config      - Generate or validate configuration files
    test        - Test story grouping functionality
    deploy      - Full deployment validation and setup

Options:
    --config-path PATH    - Custom configuration file path
    --environment ENV     - Deployment environment (dev, staging, prod)
    --dry-run            - Validation only, no changes
    --verbose            - Detailed output
    --help               - Show help message

Examples:
    python scripts/deploy_story_grouping.py validate
    python scripts/deploy_story_grouping.py setup --environment=prod
    python scripts/deploy_story_grouping.py config --config-path=custom.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from nfl_news_pipeline.story_grouping_config import (
        StoryGroupingConfigManager,
        StoryGroupingConfig,
        StoryGroupingConfigError,
    )
    from nfl_news_pipeline.storage.database_manager import get_supabase_client
except ImportError as e:
    print(f"Error importing story grouping modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class DeploymentValidator:
    """Validates deployment environment and configuration for story grouping."""
    
    def __init__(self, config_path: Optional[Path] = None, verbose: bool = False):
        self.config_path = config_path or PROJECT_ROOT / "story_grouping_config.yaml"
        self.verbose = verbose
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        return logger
        
    def _print_status(self, message: str, status: str = "info") -> None:
        """Print colored status message."""
        color_map = {
            "info": Colors.BLUE,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED,
        }
        color = color_map.get(status, Colors.NC)
        status_text = status.upper()
        print(f"{color}[{status_text}]{Colors.NC} {message}")
        
    def check_python_environment(self) -> bool:
        """Check Python version and virtual environment."""
        self._print_status("Checking Python environment...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            self._print_status(
                f"Python 3.11+ required. Found: {sys.version_info.major}.{sys.version_info.minor}",
                "error"
            )
            return False
            
        # Check if in virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self._print_status("Virtual environment detected", "success")
        else:
            self._print_status("No virtual environment detected - recommended to use one", "warning")
            
        self._print_status(f"Python {sys.version_info.major}.{sys.version_info.minor} OK", "success")
        return True
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        self._print_status("Checking dependencies...")
        
        required_modules = [
            "yaml",
            "supabase", 
            "numpy",
            "sklearn",
            "sentence_transformers",
            "openai",
        ]
        
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
                
        if missing_modules:
            self._print_status(f"Missing dependencies: {', '.join(missing_modules)}", "error")
            self._print_status("Install with: pip install -r requirements.txt", "info")
            return False
            
        self._print_status("All required dependencies installed", "success")
        return True
        
    def check_environment_variables(self) -> Tuple[bool, Dict[str, bool]]:
        """Check environment variables setup."""
        self._print_status("Checking environment variables...")
        
        # Load .env file if it exists
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            self._print_status(f"Loading environment from {env_file}", "info")
            from dotenv import load_dotenv
            load_dotenv(env_file)
        
        required_vars = {
            "SUPABASE_URL": os.getenv("SUPABASE_URL"),
            "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        }
        
        # At least one LLM API key required
        llm_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"), 
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        }
        
        missing_required = [k for k, v in required_vars.items() if not v]
        has_llm_key = any(llm_keys.values())
        
        env_status = {}
        all_good = True
        
        for var, value in required_vars.items():
            env_status[var] = bool(value)
            if not value:
                self._print_status(f"Missing required variable: {var}", "error")
                all_good = False
                
        if not has_llm_key:
            self._print_status("Missing LLM API key (need one of: OPENAI_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY)", "error")
            all_good = False
        else:
            active_keys = [k for k, v in llm_keys.items() if v]
            self._print_status(f"LLM API keys available: {', '.join(active_keys)}", "success")
            
        if all_good:
            self._print_status("Environment variables OK", "success")
            
        return all_good, env_status
        
    def check_configuration(self) -> bool:
        """Check and validate configuration file."""
        self._print_status(f"Checking configuration: {self.config_path}")
        
        try:
            manager = StoryGroupingConfigManager(self.config_path)
            config = manager.load_config()
            
            # Check environment variable validation
            issues = manager.validate_environment()
            if issues:
                self._print_status("Configuration environment issues:", "warning")
                for issue in issues:
                    print(f"  - {issue}")
                    
            self._print_status("Configuration loaded and validated", "success")
            
            if self.verbose:
                self._print_status("Configuration summary:", "info")
                print(f"  LLM Provider: {config.llm.provider}")
                print(f"  LLM Model: {config.llm.model}")
                print(f"  Embedding Model: {config.embedding.model_name}")
                print(f"  Similarity Threshold: {config.similarity.threshold}")
                print(f"  Max Group Size: {config.grouping.max_group_size}")
                
            return True
            
        except StoryGroupingConfigError as e:
            self._print_status(f"Configuration error: {e}", "error")
            return False
        except Exception as e:
            self._print_status(f"Configuration check failed: {e}", "error")
            return False
            
    def check_database_connectivity(self) -> bool:
        """Check database connectivity and schema."""
        self._print_status("Checking database connectivity...")
        
        try:
            client = get_supabase_client()
            
            # Test basic connectivity
            response = client.table('news_items').select('id').limit(1).execute()
            self._print_status("Database connection OK", "success")
            
            # Check story grouping tables
            try:
                response = client.table('story_groups').select('id').limit(1).execute()
                self._print_status("Story grouping schema available", "success")
                return True
            except Exception:
                self._print_status("Story grouping schema not found - migration may be needed", "warning")
                return False
                
        except Exception as e:
            self._print_status(f"Database connectivity failed: {e}", "error")
            return False
            
    def test_story_grouping_functionality(self) -> bool:
        """Test story grouping functionality."""
        self._print_status("Testing story grouping functionality...")
        
        try:
            # Import orchestrator components
            from nfl_news_pipeline.orchestrator.story_grouping import (
                StoryGroupingOrchestrator,
                StoryGroupingSettings
            )
            
            # Load configuration
            manager = StoryGroupingConfigManager(self.config_path)
            config = manager.load_config()
            settings = config.get_orchestrator_settings()
            
            # Test orchestrator initialization
            client = get_supabase_client()
            orchestrator = StoryGroupingOrchestrator(client, settings)
            
            self._print_status("Story grouping orchestrator initialized", "success")
            
            if self.verbose:
                self._print_status("Orchestrator settings:", "info")
                print(f"  Max Parallelism: {settings.max_parallelism}")
                print(f"  Max Candidates: {settings.max_candidates}")
                print(f"  Similarity Floor: {settings.candidate_similarity_floor}")
                
            return True
            
        except Exception as e:
            self._print_status(f"Story grouping functionality test failed: {e}", "error")
            return False
            
    def run_full_validation(self) -> bool:
        """Run complete deployment validation."""
        self._print_status("Running full deployment validation...", "info")
        print("=" * 60)
        
        checks = [
            ("Python Environment", self.check_python_environment),
            ("Dependencies", self.check_dependencies),
            ("Environment Variables", lambda: self.check_environment_variables()[0]),
            ("Configuration", self.check_configuration),
            ("Database", self.check_database_connectivity),
            ("Story Grouping", self.test_story_grouping_functionality),
        ]
        
        results = {}
        all_passed = True
        
        for check_name, check_func in checks:
            print()
            try:
                result = check_func()
                results[check_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                self._print_status(f"{check_name} check failed with exception: {e}", "error")
                results[check_name] = False
                all_passed = False
                
        # Print summary
        print("\n" + "=" * 60)
        self._print_status("Validation Summary:", "info")
        
        for check_name, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            
        if all_passed:
            self._print_status("All validation checks passed! 🎉", "success")
            self._print_status("Story Grouping feature is ready for deployment", "success")
        else:
            self._print_status("Some validation checks failed", "error")
            self._print_status("Please fix the issues above before deploying", "error")
            
        return all_passed


def create_default_config(config_path: Path) -> bool:
    """Create default configuration file."""
    try:
        if config_path.exists():
            print(f"Configuration file already exists: {config_path}")
            return True
            
        # Copy from default config
        default_config_path = PROJECT_ROOT / "story_grouping_config.yaml"
        if default_config_path.exists():
            import shutil
            shutil.copy2(default_config_path, config_path)
            print(f"Created configuration file: {config_path}")
            return True
        else:
            print("Default configuration template not found")
            return False
            
    except Exception as e:
        print(f"Failed to create configuration file: {e}")
        return False


def interactive_setup():
    """Interactive setup wizard."""
    print(f"{Colors.BOLD}🚀 Story Grouping Feature Setup Wizard{Colors.NC}")
    print("=" * 50)
    
    # Check if .env exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"\n{Colors.YELLOW}No .env file found{Colors.NC}")
        create_env = input("Create .env file from template? (y/N): ").lower().strip()
        if create_env == 'y':
            import shutil
            shutil.copy2(PROJECT_ROOT / ".env.example", env_file)
            print(f"{Colors.GREEN}Created .env file{Colors.NC}")
            print(f"{Colors.BLUE}Please edit {env_file} with your configuration{Colors.NC}")
    
    # Ask about configuration
    config_path = PROJECT_ROOT / "story_grouping_config.yaml"
    if not config_path.exists():
        print(f"\n{Colors.YELLOW}No story grouping configuration found{Colors.NC}")
        create_config = input("Create default configuration? (y/N): ").lower().strip()
        if create_config == 'y':
            create_default_config(config_path)
    
    # Run validation
    print(f"\n{Colors.BLUE}Running deployment validation...{Colors.NC}")
    validator = DeploymentValidator(config_path, verbose=True)
    success = validator.run_full_validation()
    
    if success:
        print(f"\n{Colors.GREEN}✅ Setup completed successfully!{Colors.NC}")
        print(f"\n{Colors.BLUE}Next steps:{Colors.NC}")
        print("1. Test story grouping: python scripts/story_grouping_dry_run.py")
        print("2. Run full pipeline with story grouping enabled")
        print("3. Monitor logs and adjust configuration as needed")
    else:
        print(f"\n{Colors.RED}❌ Setup validation failed{Colors.NC}")
        print("Please fix the issues above and run the setup again")


def main():
    """Main entry point for deployment script."""
    parser = argparse.ArgumentParser(
        description="Story Grouping Feature Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="validate",
        choices=["validate", "setup", "config", "test", "deploy"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--environment",
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Deployment environment"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validation only, no changes"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Detailed output"
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DeploymentValidator(args.config_path, args.verbose)
    
    # Execute command
    if args.command == "validate" or args.command == "deploy":
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)
        
    elif args.command == "setup":
        interactive_setup()
        
    elif args.command == "config":
        config_path = args.config_path or PROJECT_ROOT / "story_grouping_config.yaml"
        if create_default_config(config_path):
            validator = DeploymentValidator(config_path, args.verbose)
            success = validator.check_configuration()
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)
            
    elif args.command == "test":
        success = validator.test_story_grouping_functionality()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
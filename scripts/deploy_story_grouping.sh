#!/bin/bash
"""
Story Grouping Feature Deployment Script
=========================================

This script helps deploy and configure the NFL News Pipeline Story Grouping feature.
It validates environment setup, initializes configuration, and provides deployment guidance.

Usage:
    ./scripts/deploy_story_grouping.sh [--dry-run] [--config-path=path] [--environment=env]

Options:
    --dry-run                   Validate setup without making changes
    --config-path=PATH         Path to story grouping config file (default: story_grouping_config.yaml)
    --environment=ENV          Deployment environment (dev, staging, prod)
    --help                     Show this help message

Requirements:
    - Python 3.11+
    - All dependencies installed (pip install -r requirements.txt)
    - Environment variables configured (.env file or system environment)
    - Supabase database setup and accessible
"""

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_CONFIG="story_grouping_config.yaml"
DRY_RUN=false
ENVIRONMENT="dev"
CONFIG_PATH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config-path=*)
            CONFIG_PATH="${1#*=}"
            shift
            ;;
        --environment=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Story Grouping Feature Deployment Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run                   Validate setup without making changes"
            echo "  --config-path=PATH         Path to story grouping config file"
            echo "  --environment=ENV          Deployment environment (dev, staging, prod)"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run               # Validate configuration"
            echo "  $0 --environment=prod      # Deploy to production"
            echo "  $0 --config-path=custom.yaml  # Use custom config"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default config path if not provided
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$DEFAULT_CONFIG"
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available and correct version
check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        return 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    required_version="3.11"
    
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
        log_error "Python 3.11+ is required. Found: $python_version"
        return 1
    fi
    
    log_success "Python $python_version is installed and compatible"
    return 0
}

# Check if required dependencies are installed
check_dependencies() {
    log_info "Checking Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        log_error "requirements.txt not found in project root"
        return 1
    fi
    
    # Try importing key modules
    if ! python3 -c "import yaml, supabase, numpy, sklearn" 2>/dev/null; then
        log_warning "Some dependencies may be missing. Run: pip install -r requirements.txt"
        if [[ "$DRY_RUN" == "false" ]]; then
            read -p "Install dependencies now? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Installing dependencies..."
                pip install -r requirements.txt
                log_success "Dependencies installed"
            fi
        fi
    else
        log_success "Required dependencies are available"
    fi
    
    return 0
}

# Check environment variables
check_environment_variables() {
    log_info "Checking environment variables..."
    
    local missing_vars=()
    local optional_vars=()
    
    # Required variables
    if [[ -z "${SUPABASE_URL:-}" ]]; then
        missing_vars+=("SUPABASE_URL")
    fi
    
    if [[ -z "${SUPABASE_KEY:-}" ]]; then
        missing_vars+=("SUPABASE_KEY")
    fi
    
    # LLM API keys (at least one required)
    if [[ -z "${OPENAI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
        missing_vars+=("OPENAI_API_KEY or GOOGLE_API_KEY or DEEPSEEK_API_KEY")
    fi
    
    # Check for .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        log_success "Found .env file"
        source "$PROJECT_ROOT/.env"
    else
        log_warning "No .env file found. Copy .env.example to .env and configure it."
    fi
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        return 1
    else
        log_success "All required environment variables are set"
    fi
    
    return 0
}

# Check configuration file
check_configuration() {
    log_info "Checking story grouping configuration..."
    
    if [[ ! -f "$CONFIG_PATH" ]]; then
        log_warning "Configuration file not found: $CONFIG_PATH"
        log_info "Creating default configuration file..."
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Copy the default config if it doesn't exist
            cp "$PROJECT_ROOT/story_grouping_config.yaml" "$CONFIG_PATH" 2>/dev/null || {
                log_error "Could not create configuration file"
                return 1
            }
            log_success "Created configuration file: $CONFIG_PATH"
        fi
    else
        log_success "Configuration file found: $CONFIG_PATH"
    fi
    
    # Validate configuration using Python
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from nfl_news_pipeline.story_grouping_config import StoryGroupingConfigManager
    manager = StoryGroupingConfigManager('$CONFIG_PATH')
    config = manager.load_config()
    issues = manager.validate_environment()
    if issues:
        print('Configuration validation issues:')
        for issue in issues:
            print(f'  - {issue}')
        sys.exit(1)
    else:
        print('Configuration is valid')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
" || {
        log_error "Configuration validation failed"
        return 1
    }
    
    log_success "Configuration is valid"
    return 0
}

# Check database connectivity
check_database() {
    log_info "Checking database connectivity..."
    
    cd "$PROJECT_ROOT"
    python3 -c "
import os
import sys
try:
    from supabase import create_client
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    
    if not url or not key:
        print('Database credentials not found in environment')
        sys.exit(1)
    
    client = create_client(url, key)
    
    # Test basic connectivity
    response = client.table('news_items').select('id').limit(1).execute()
    print(f'Database connection successful')
    
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" || {
        log_error "Database connectivity check failed"
        return 1
    }
    
    log_success "Database is accessible"
    return 0
}

# Run story grouping migration check
check_migration() {
    log_info "Checking story grouping database schema..."
    
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from nfl_news_pipeline.storage.story_group_storage import GroupStorageManager
    from nfl_news_pipeline.storage.database_manager import get_supabase_client
    
    client = get_supabase_client()
    storage = GroupStorageManager(client)
    
    # Test if story grouping tables exist
    response = client.table('story_groups').select('id').limit(1).execute()
    print('Story grouping tables are available')
    
except Exception as e:
    print(f'Story grouping schema check failed: {e}')
    print('Please ensure story grouping database migration has been run')
    sys.exit(1)
" || {
        log_error "Story grouping database schema is not set up"
        log_info "Run the story grouping migration first"
        return 1
    }
    
    log_success "Story grouping database schema is ready"
    return 0
}

# Test story grouping functionality
test_story_grouping() {
    log_info "Testing story grouping functionality..."
    
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from nfl_news_pipeline.story_grouping_config import get_story_grouping_config
    from nfl_news_pipeline.orchestrator.story_grouping import StoryGroupingOrchestrator
    from nfl_news_pipeline.storage.database_manager import get_supabase_client
    
    # Load configuration
    config = get_story_grouping_config('$CONFIG_PATH')
    settings = config.get_orchestrator_settings()
    
    # Initialize orchestrator
    client = get_supabase_client()
    orchestrator = StoryGroupingOrchestrator(client, settings)
    
    print('Story grouping orchestrator initialized successfully')
    print(f'Configuration: {settings.max_parallelism} max parallelism, {settings.max_candidates} max candidates')
    
except Exception as e:
    print(f'Story grouping test failed: {e}')
    sys.exit(1)
" || {
        log_error "Story grouping functionality test failed"
        return 1
    }
    
    log_success "Story grouping functionality test passed"
    return 0
}

# Deploy or validate based on mode
deploy_story_grouping() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "=== DRY RUN MODE - VALIDATION ONLY ==="
    else
        log_info "=== DEPLOYING STORY GROUPING FEATURE ==="
    fi
    
    log_info "Environment: $ENVIRONMENT"
    log_info "Configuration file: $CONFIG_PATH"
    echo
    
    # Run all checks
    local checks=(
        "check_python"
        "check_dependencies" 
        "check_environment_variables"
        "check_configuration"
        "check_database"
        "check_migration"
        "test_story_grouping"
    )
    
    local failed_checks=()
    
    for check in "${checks[@]}"; do
        if ! $check; then
            failed_checks+=("$check")
        fi
        echo
    done
    
    # Summary
    echo "==============================================="
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        log_success "All deployment checks passed!"
        if [[ "$DRY_RUN" == "false" ]]; then
            log_success "Story Grouping feature is ready for use"
            echo
            log_info "Next steps:"
            echo "  1. Run story grouping: python scripts/story_grouping_dry_run.py"
            echo "  2. Check logs for processing results"
            echo "  3. Monitor performance and adjust configuration as needed"
        else
            log_success "Story Grouping feature deployment validation passed"
            log_info "Run without --dry-run to deploy"
        fi
    else
        log_error "Deployment validation failed. Please fix the following issues:"
        for check in "${failed_checks[@]}"; do
            echo "  - $check"
        done
        exit 1
    fi
}

# Main execution
main() {
    echo "🚀 Story Grouping Feature Deployment"
    echo "======================================"
    echo
    
    deploy_story_grouping
}

# Run the script
main "$@"
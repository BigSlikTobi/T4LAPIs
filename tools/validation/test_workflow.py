#!/usr/bin/env python3
"""
Test script for validating the Personalized Summary Generation workflow.

This script simulates the GitHub Actions workflow steps locally to ensure
everything works correctly before deploying.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load environment variables from a local .env if present to mimic GitHub Actions env injection
load_dotenv(os.path.join(project_root, ".env"), override=False)

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_environment_variables():
    """Test that required environment variables are set."""
    print("🔍 Testing environment variables...")

    # Required for database access
    base_required = ['SUPABASE_URL', 'SUPABASE_KEY']

    # Gemini key can be provided as GEMINI_API_KEY (preferred) or GOOGLE_API_KEY (alias)
    gemini_keys = ['GEMINI_API_KEY', 'GOOGLE_API_KEY']

    # DeepSeek is used as fallback provider
    deepseek_required = ['DEEPSEEK_API_KEY']

    missing_vars = []

    # Check base required
    for var in base_required:
        if not os.getenv(var):
            missing_vars.append(var)

    # Check Gemini/Goggle AI key presence
    if not any(os.getenv(k) for k in gemini_keys):
        missing_vars.append('GEMINI_API_KEY (or GOOGLE_API_KEY)')

    # Check DeepSeek
    for var in deepseek_required:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("   Set these variables before running the workflow (you can place them in a .env at repo root):")
        # Provide concrete export hints
        if 'SUPABASE_URL' in missing_vars:
            print("   export SUPABASE_URL=your_supabase_url_here")
        if 'SUPABASE_KEY' in missing_vars:
            print("   export SUPABASE_KEY=your_supabase_key_here")
        if any('GEMINI_API_KEY' in mv for mv in missing_vars):
            print("   # Prefer GEMINI_API_KEY; GOOGLE_API_KEY is accepted as an alias")
            print("   export GEMINI_API_KEY=your_gemini_api_key_here")
            print("   # or")
            print("   export GOOGLE_API_KEY=your_google_api_key_here")
        if 'DEEPSEEK_API_KEY' in missing_vars:
            print("   export DEEPSEEK_API_KEY=your_deepseek_api_key_here")
        return False

    print("✅ All required environment variables are set")
    return True

def test_dependencies():
    """Test that required dependencies are available."""
    print("📦 Testing dependencies...")
    
    try:
        from src.core.db.database_init import get_supabase_client
        from src.core.llm.llm_setup import initialize_model
        from content_generation.personal_summary_generator import PersonalizedSummaryGenerator
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def test_database_connection():
    """Test database connectivity."""
    print("🗃️ Testing database connection...")
    
    try:
        from src.core.db.database_init import get_supabase_client
        client = get_supabase_client()
        
        # Test simple query
        response = client.table('users').select('user_id').limit(1).execute()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_llm_connections():
    """Test LLM connections."""
    print("🤖 Testing LLM connections...")
    
    try:
        from src.core.llm.llm_setup import initialize_model
        
        # Test Gemini
        try:
            gemini_config = initialize_model('gemini', 'flash', grounding_enabled=False)
            print("✅ Gemini connection successful")
            gemini_ok = True
        except Exception as e:
            print(f"⚠️ Gemini connection failed: {e}")
            gemini_ok = False
        
        # Test DeepSeek
        try:
            deepseek_config = initialize_model('deepseek', 'chat')
            print("✅ DeepSeek connection successful")
            deepseek_ok = True
        except Exception as e:
            print(f"❌ DeepSeek connection failed: {e}")
            deepseek_ok = False
        
        if not deepseek_ok:
            print("❌ DeepSeek is required as fallback - check DEEPSEEK_API_KEY")
            return False
        
        if gemini_ok:
            print("🎉 Primary Gemini + Fallback DeepSeek ready")
        else:
            print("⚠️ Using DeepSeek only (Gemini unavailable)")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM setup failed: {e}")
        return False

def test_summary_generator():
    """Test the PersonalizedSummaryGenerator without running full generation."""
    print("🎯 Testing PersonalizedSummaryGenerator initialization...")
    
    try:
        from content_generation.personal_summary_generator import PersonalizedSummaryGenerator
        
        # Initialize with minimal settings
        generator = PersonalizedSummaryGenerator(lookback_hours=1)  # Short lookback for test
        
        # Test LLM initialization
        if generator.initialize_llm():
            print("✅ PersonalizedSummaryGenerator LLM initialization successful")
        else:
            print("❌ PersonalizedSummaryGenerator LLM initialization failed")
            return False
        
        # Test database managers
        try:
            users = generator.get_all_users_with_preferences()
            print(f"✅ Found {len(users)} users with preferences")
        except Exception as e:
            print(f"❌ Error fetching users: {e}")
            return False
        
        print("✅ PersonalizedSummaryGenerator ready for workflow")
        return True
        
    except Exception as e:
        print(f"❌ PersonalizedSummaryGenerator test failed: {e}")
        return False

def run_workflow_simulation():
    """Simulate the full workflow (dry run)."""
    print("🚀 Running workflow simulation...")
    
    try:
        from content_generation.personal_summary_generator import PersonalizedSummaryGenerator
        
        # Create generator
        generator = PersonalizedSummaryGenerator(lookback_hours=1)  # Short test period
        
        # Initialize LLM
        if not generator.initialize_llm():
            print("❌ Failed to initialize LLM in simulation")
            return False
        
        # Get users (don't generate summaries, just test data flow)
        users = generator.get_all_users_with_preferences()
        
        print(f"🎯 Simulation Results:")
        print(f"   👥 Users with preferences: {len(users)}")
        
        total_preferences = sum(len(user.get('preferences', [])) for user in users)
        print(f"   ⭐ Total preferences: {total_preferences}")
        
        if total_preferences > 0:
            print("✅ Workflow simulation successful - ready for production")
            return True
        else:
            print("⚠️ No preferences found - workflow will run but generate no summaries")
            return True
        
    except Exception as e:
        print(f"❌ Workflow simulation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 GitHub Actions Workflow Validation")
    print("=" * 50)
    print(f"🕐 Test time: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    setup_logging()
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Dependencies", test_dependencies),
        ("Database Connection", test_database_connection),
        ("LLM Connections", test_llm_connections),
        ("Summary Generator", test_summary_generator),
        ("Workflow Simulation", run_workflow_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✅' if success else '❌'} {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Results Summary")
    print("-" * 30)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Workflow is ready for deployment.")
        print("\n🚀 Next steps:")
        print("1. Commit the workflow file to GitHub")
        print("2. Add required secrets to GitHub repository:")
        print("   - SUPABASE_URL")
        print("   - SUPABASE_KEY") 
        print("   - GEMINI_API_KEY")
        print("   - DEEPSEEK_API_KEY")
        print("3. Test with manual workflow_dispatch trigger")
        print("4. Monitor scheduled runs")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
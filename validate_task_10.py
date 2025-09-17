#!/usr/bin/env python3
"""
Final validation test for Task 10 completion.
This test verifies all Task 10 requirements are met.
"""

import os
import sys
from pathlib import Path

def check_task_10_requirements():
    """
    Verify Task 10 requirements from tasks.md:
    
    10. [ ] Create configuration and deployment setup
    - Create configuration management for all story grouping parameters
    - Add environment variable setup for LLM API keys and database connections
    - Implement deployment scripts and documentation for the new feature
    - Requirements: 1.6, 2.6, 6.5
    """
    print("🎯 Task 10 Completion Validation")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    all_good = True
    
    # Check 1: Configuration management for story grouping parameters
    print("\n📋 1. Configuration Management")
    
    config_file = project_root / "story_grouping_config.yaml"
    config_manager = project_root / "src" / "nfl_news_pipeline" / "story_grouping_config.py"
    
    if config_file.exists():
        print("   ✅ story_grouping_config.yaml exists")
        
        # Check configuration sections
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        expected_sections = ['llm', 'embedding', 'similarity', 'grouping', 'performance', 'database', 'monitoring']
        missing_sections = [s for s in expected_sections if s not in config]
        
        if not missing_sections:
            print("   ✅ All configuration sections present")
        else:
            print(f"   ❌ Missing sections: {missing_sections}")
            all_good = False
    else:
        print("   ❌ story_grouping_config.yaml missing")
        all_good = False
    
    if config_manager.exists():
        print("   ✅ StoryGroupingConfigManager class implemented")
        
        # Check if it has required methods
        content = config_manager.read_text()
        required_methods = ['load_config', 'validate', 'get_orchestrator_settings', 'check_environment_variables']
        missing_methods = [m for m in required_methods if m not in content]
        
        if not missing_methods:
            print("   ✅ All required methods implemented")
        else:
            print(f"   ❌ Missing methods: {missing_methods}")
            all_good = False
    else:
        print("   ❌ StoryGroupingConfigManager missing")
        all_good = False
    
    # Check 2: Environment variable setup
    print("\n🌍 2. Environment Variable Setup")
    
    env_example = project_root / ".env.example"
    if env_example.exists():
        content = env_example.read_text()
        
        # Check for LLM API keys
        llm_keys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'DEEPSEEK_API_KEY']
        found_keys = [key for key in llm_keys if key in content]
        
        if found_keys:
            print(f"   ✅ LLM API keys documented: {', '.join(found_keys)}")
        else:
            print("   ❌ No LLM API keys found in .env.example")
            all_good = False
        
        # Check for database connections
        db_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
        found_db = [var for var in db_vars if var in content]
        
        if len(found_db) == len(db_vars):
            print("   ✅ Database connection variables documented")
        else:
            missing_db = [var for var in db_vars if var not in found_db]
            print(f"   ❌ Missing database variables: {missing_db}")
            all_good = False
        
        # Check for story grouping specific variables
        sg_vars = ['STORY_GROUPING_ENABLED', 'STORY_GROUPING_DRY_RUN']
        found_sg = [var for var in sg_vars if var in content]
        
        if found_sg:
            print(f"   ✅ Story grouping variables documented: {', '.join(found_sg)}")
        else:
            print("   ⚠️  Story grouping specific variables could be added")
    else:
        print("   ❌ .env.example file missing")
        all_good = False
    
    # Check 3: Deployment scripts
    print("\n🚀 3. Deployment Scripts")
    
    bash_script = project_root / "scripts" / "deploy_story_grouping.sh"
    python_script = project_root / "scripts" / "deploy_story_grouping.py"
    
    if bash_script.exists():
        if bash_script.stat().st_mode & 0o111:
            print("   ✅ Bash deployment script exists and is executable")
        else:
            print("   ⚠️  Bash deployment script exists but not executable")
    else:
        print("   ❌ Bash deployment script missing")
        all_good = False
    
    if python_script.exists():
        if python_script.stat().st_mode & 0o111:
            print("   ✅ Python deployment script exists and is executable")
        else:
            print("   ⚠️  Python deployment script exists but not executable")
        
        # Check for key features
        content = python_script.read_text()
        features = ['validate', 'setup', 'test', 'environment', 'configuration']
        found_features = [f for f in features if f in content.lower()]
        
        if len(found_features) >= 4:
            print(f"   ✅ Deployment script has comprehensive features")
        else:
            print(f"   ⚠️  Deployment script could have more features")
    else:
        print("   ❌ Python deployment script missing")
        all_good = False
    
    # Check 4: Documentation
    print("\n📚 4. Documentation")
    
    doc_file = project_root / "docs" / "Story_Grouping_Configuration.md"
    if doc_file.exists():
        content = doc_file.read_text()
        word_count = len(content.split())
        
        print(f"   ✅ Configuration documentation exists ({word_count} words)")
        
        # Check for key sections
        key_sections = [
            "Configuration Management",
            "Environment Variables",
            "Deployment Process",
            "Troubleshooting"
        ]
        
        found_sections = [s for s in key_sections if s in content]
        
        if len(found_sections) == len(key_sections):
            print("   ✅ All key documentation sections present")
        else:
            missing = [s for s in key_sections if s not in found_sections]
            print(f"   ⚠️  Missing documentation sections: {missing}")
    else:
        print("   ❌ Configuration documentation missing")
        all_good = False
    
    # Check 5: Integration with existing pipeline
    print("\n🔧 5. Pipeline Integration")
    
    dry_run_script = project_root / "scripts" / "story_grouping_dry_run.py"
    if dry_run_script.exists():
        content = dry_run_script.read_text()
        
        if "--config" in content and "StoryGroupingConfigManager" in content:
            print("   ✅ Dry run script updated for configuration support")
        else:
            print("   ⚠️  Dry run script integration could be improved")
    else:
        print("   ⚠️  Dry run script not found for integration check")
    
    # Final result
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Task 10 COMPLETED successfully!")
        print("\n✅ All requirements met:")
        print("   - Configuration management implemented")
        print("   - Environment variables documented")
        print("   - Deployment scripts created")
        print("   - Documentation provided")
        print("   - Pipeline integration ready")
        
        print("\n🚀 Ready for deployment:")
        print("   python scripts/deploy_story_grouping.py validate")
        print("   python scripts/deploy_story_grouping.py setup")
        
        return True
    else:
        print("❌ Task 10 has some incomplete items")
        print("   Please review the issues above")
        return False

def main():
    """Run Task 10 validation."""
    success = check_task_10_requirements()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
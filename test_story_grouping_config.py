#!/usr/bin/env python3
"""Simple test for story grouping configuration without heavy dependencies."""

import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Simple mock of the configuration classes for testing
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 500
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass 
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    cache_ttl_hours: int = 24
    normalize_vectors: bool = True

@dataclass
class SimilarityConfig:
    threshold: float = 0.8
    metric: str = "cosine"
    max_candidates: int = 100
    search_limit: int = 1000
    candidate_similarity_floor: float = 0.35

@dataclass
class StoryGroupingConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)

def test_config_loading():
    """Test configuration loading from YAML file."""
    config_path = Path("story_grouping_config.yaml")
    
    print("🧪 Testing Story Grouping Configuration")
    print("=" * 50)
    
    # Test 1: Check if config file exists
    if config_path.exists():
        print("✅ Configuration file exists")
    else:
        print("❌ Configuration file not found")
        return False
    
    # Test 2: Load and parse YAML
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        print("✅ YAML file parsed successfully")
    except Exception as e:
        print(f"❌ YAML parsing failed: {e}")
        return False
    
    # Test 3: Validate structure
    expected_sections = ['llm', 'embedding', 'similarity', 'grouping', 'performance', 'database', 'monitoring']
    missing_sections = []
    
    for section in expected_sections:
        if section not in raw_config:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"❌ Missing configuration sections: {missing_sections}")
        return False
    else:
        print("✅ All expected configuration sections present")
    
    # Test 4: Validate key parameters
    try:
        llm_config = raw_config.get('llm', {})
        assert llm_config.get('provider') in ['openai', 'google', 'deepseek'], "Invalid LLM provider"
        assert isinstance(llm_config.get('max_tokens', 500), int), "max_tokens must be integer"
        assert 0.0 <= llm_config.get('temperature', 0.1) <= 2.0, "temperature must be 0.0-2.0"
        
        similarity_config = raw_config.get('similarity', {})
        assert 0.0 <= similarity_config.get('threshold', 0.8) <= 1.0, "threshold must be 0.0-1.0"
        assert similarity_config.get('metric') in ['cosine', 'euclidean', 'dot_product'], "Invalid similarity metric"
        
        print("✅ Configuration parameters validated")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    
    # Test 5: Environment variables documentation
    env_vars = raw_config.get('environment_variables', {})
    if 'required' in env_vars and 'optional' in env_vars:
        print("✅ Environment variables documented")
        print(f"   Required: {len(env_vars['required'])} variables")
        print(f"   Optional: {len(env_vars['optional'])} variables")
    else:
        print("⚠️  Environment variables section incomplete")
    
    # Test 6: Display configuration summary
    print("\n📋 Configuration Summary:")
    print(f"   LLM Provider: {llm_config.get('provider', 'N/A')}")
    print(f"   LLM Model: {llm_config.get('model', 'N/A')}")
    print(f"   Embedding Model: {raw_config.get('embedding', {}).get('model_name', 'N/A')}")
    print(f"   Similarity Threshold: {raw_config.get('similarity', {}).get('threshold', 'N/A')}")
    print(f"   Max Group Size: {raw_config.get('grouping', {}).get('max_group_size', 'N/A')}")
    
    return True

def test_environment_file():
    """Test .env.example file for completeness."""
    print("\n🔧 Testing Environment Configuration")
    print("=" * 40)
    
    env_file = Path(".env.example")
    if not env_file.exists():
        print("❌ .env.example file not found")
        return False
    
    content = env_file.read_text()
    
    # Check for required environment variables
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_KEY', 
        'OPENAI_API_KEY',
        'STORY_GROUPING_ENABLED',
        'LOG_LEVEL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    else:
        print("✅ All key environment variables documented")
    
    return True

def test_deployment_scripts():
    """Test deployment scripts exist and are executable."""
    print("\n🚀 Testing Deployment Scripts")
    print("=" * 35)
    
    scripts = [
        Path("scripts/deploy_story_grouping.sh"),
        Path("scripts/deploy_story_grouping.py")
    ]
    
    all_good = True
    for script in scripts:
        if script.exists():
            if script.suffix == '.sh' and not (script.stat().st_mode & 0o111):
                print(f"⚠️  {script.name} exists but not executable")
            else:
                print(f"✅ {script.name} available")
        else:
            print(f"❌ {script.name} not found")
            all_good = False
    
    return all_good

def test_documentation():
    """Test documentation exists."""
    print("\n📚 Testing Documentation")
    print("=" * 30)
    
    doc_file = Path("docs/Story_Grouping_Configuration.md")
    if doc_file.exists():
        content = doc_file.read_text()
        word_count = len(content.split())
        print(f"✅ Configuration documentation exists ({word_count} words)")
        
        # Check for key sections
        key_sections = [
            "Configuration Management",
            "Environment Variables", 
            "Deployment Process",
            "Troubleshooting"
        ]
        
        missing_sections = []
        for section in key_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️  Missing documentation sections: {missing_sections}")
        else:
            print("✅ All key documentation sections present")
        
        return True
    else:
        print("❌ Configuration documentation not found")
        return False

def main():
    """Run all tests."""
    print("🧪 Story Grouping Configuration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Environment File", test_environment_file), 
        ("Deployment Scripts", test_deployment_scripts),
        ("Documentation", test_documentation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
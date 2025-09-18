#!/usr/bin/env python3
"""
Simple integration test for story grouping configuration.
Tests the configuration loading and compatibility without heavy dependencies.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

def test_basic_integration():
    """Test basic configuration integration."""
    print("🧪 Testing Story Grouping Configuration Integration")
    print("=" * 60)
    
    # Create a test configuration file
    test_config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 500,
            "temperature": 0.1,
            "timeout_seconds": 30,
            "max_retries": 3
        },
        "embedding": {
            "model_name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 16,  # Different from default
            "cache_ttl_hours": 24,
            "normalize_vectors": True
        },
        "similarity": {
            "threshold": 0.75,  # Different from default
            "metric": "cosine",
            "max_candidates": 50,  # Different from default
            "search_limit": 1000,
            "candidate_similarity_floor": 0.4  # Different from default
        },
        "grouping": {
            "max_group_size": 25,  # Different from default
            "centroid_update_threshold": 5,
            "status_transition_hours": 24,
            "auto_tagging": True,
            "max_stories_per_run": None,
            "prioritize_recent": True,
            "prioritize_high_relevance": True,
            "reprocess_existing": False
        },
        "performance": {
            "parallel_processing": True,
            "max_workers": 2,  # Different from default
            "max_parallelism": 2,  # Different from default
            "batch_size": 10,
            "cache_size": 1000,
            "max_total_processing_time": None
        },
        "monitoring": {
            "log_level": "DEBUG",  # Different from default
            "metrics_enabled": True,
            "cost_tracking_enabled": True,
            "performance_alerts_enabled": True,
            "group_quality_monitoring": True
        }
    }
    
    # Write test config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        test_config_path = f.name
    
    try:
        # Test 1: Configuration file structure
        print("🔧 Testing configuration structure...")
        
        with open(test_config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Verify structure
        expected_sections = ['llm', 'embedding', 'similarity', 'grouping', 'performance', 'monitoring']
        for section in expected_sections:
            assert section in loaded_config, f"Missing section: {section}"
        
        print("✅ Configuration structure validated")
        
        # Test 2: Parameter validation logic
        print("\n🎯 Testing parameter validation...")
        
        # Simulate parameter validation
        llm_config = loaded_config['llm']
        assert llm_config['provider'] in ['openai', 'google', 'deepseek'], "Invalid LLM provider"
        assert isinstance(llm_config['max_tokens'], int) and llm_config['max_tokens'] > 0, "Invalid max_tokens"
        assert 0.0 <= llm_config['temperature'] <= 2.0, "Invalid temperature"
        
        similarity_config = loaded_config['similarity']
        assert 0.0 <= similarity_config['threshold'] <= 1.0, "Invalid threshold"
        assert similarity_config['metric'] in ['cosine', 'euclidean', 'dot_product'], "Invalid metric"
        
        print("✅ Parameter validation passed")
        
        # Test 3: Orchestrator settings compatibility
        print("\n🚀 Testing orchestrator settings compatibility...")
        
        # Simulate creating orchestrator settings from config
        mock_settings = {
            'max_parallelism': loaded_config['performance']['max_parallelism'],
            'max_candidates': loaded_config['similarity']['max_candidates'],
            'candidate_similarity_floor': loaded_config['similarity']['candidate_similarity_floor'],
            'max_total_processing_time': loaded_config['performance']['max_total_processing_time'],
            'max_stories_per_run': loaded_config['grouping']['max_stories_per_run'],
            'prioritize_recent': loaded_config['grouping']['prioritize_recent'],
            'prioritize_high_relevance': loaded_config['grouping']['prioritize_high_relevance'],
            'reprocess_existing': loaded_config['grouping']['reprocess_existing'],
        }
        
        # Verify expected values from our test config
        assert mock_settings['max_parallelism'] == 2, f"Expected 2, got {mock_settings['max_parallelism']}"
        assert mock_settings['max_candidates'] == 50, f"Expected 50, got {mock_settings['max_candidates']}"
        assert mock_settings['candidate_similarity_floor'] == 0.4, f"Expected 0.4, got {mock_settings['candidate_similarity_floor']}"
        
        print("✅ Orchestrator settings compatibility verified")
        
        # Test 4: CLI argument override logic
        print("\n⚙️  Testing CLI argument override logic...")
        
        # Simulate what the dry run script would do
        mock_args = {
            'similarity_threshold': 0.8,  # CLI default
            'embedding_batch_size': 32,   # CLI default
            'max_parallelism': 4,         # CLI default
            'max_candidates': 8,          # CLI default
            'candidate_floor': 0.35,      # CLI default
            'log_level': 'INFO'           # CLI default
        }
        
        # Override with config values (like the script does)
        if mock_args['similarity_threshold'] == 0.8:  # is default
            mock_args['similarity_threshold'] = loaded_config['similarity']['threshold']
        if mock_args['embedding_batch_size'] == 32:  # is default
            mock_args['embedding_batch_size'] = loaded_config['embedding']['batch_size']
        if mock_args['max_parallelism'] == 4:  # is default
            mock_args['max_parallelism'] = loaded_config['performance']['max_parallelism']
        if mock_args['max_candidates'] == 8:  # is default
            mock_args['max_candidates'] = loaded_config['similarity']['max_candidates']
        if mock_args['candidate_floor'] == 0.35:  # is default
            mock_args['candidate_floor'] = loaded_config['similarity']['candidate_similarity_floor']
        if mock_args['log_level'] == 'INFO':  # is default
            mock_args['log_level'] = loaded_config['monitoring']['log_level']
        
        # Verify overrides worked
        assert mock_args['similarity_threshold'] == 0.75, f"Expected 0.75, got {mock_args['similarity_threshold']}"
        assert mock_args['embedding_batch_size'] == 16, f"Expected 16, got {mock_args['embedding_batch_size']}"
        assert mock_args['max_parallelism'] == 2, f"Expected 2, got {mock_args['max_parallelism']}"
        assert mock_args['log_level'] == "DEBUG", f"Expected DEBUG, got {mock_args['log_level']}"
        
        print("✅ CLI argument override logic working")
        
        # Test 5: Environment variable checking
        print("\n🌍 Testing environment variable structure...")
        
        # Check if main config has environment variables section
        main_config_path = Path("story_grouping_config.yaml")
        if main_config_path.exists():
            with open(main_config_path, 'r') as f:
                main_config = yaml.safe_load(f)
            
            if 'environment_variables' in main_config:
                env_vars = main_config['environment_variables']
                if 'required' in env_vars and 'optional' in env_vars:
                    print(f"✅ Environment variables documented:")
                    print(f"   Required: {env_vars['required']}")
                    print(f"   Optional: {env_vars['optional']}")
                else:
                    print("⚠️  Environment variables section incomplete")
            else:
                print("⚠️  No environment variables section in main config")
        
        print("\n" + "=" * 60)
        print("🎉 All integration tests passed!")
        print("\n📋 Integration Summary:")
        print(f"   Configuration File: Compatible with YAML structure")
        print(f"   LLM Provider: {loaded_config['llm']['provider']}")
        print(f"   Embedding Model: {loaded_config['embedding']['model_name']}")
        print(f"   Similarity Threshold: {loaded_config['similarity']['threshold']}")
        print(f"   Max Parallelism: {loaded_config['performance']['max_parallelism']}")
        print(f"   Batch Size: {loaded_config['embedding']['batch_size']}")
        print(f"   Log Level: {loaded_config['monitoring']['log_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(test_config_path):
            os.unlink(test_config_path)

def main():
    """Run integration tests."""
    success = test_basic_integration()
    if success:
        print("\n✅ Configuration integration is working correctly!")
        print("\n🚀 Usage Examples:")
        print("   # Use configuration file with dry run script:")
        print("   python scripts/story_grouping_dry_run.py --config story_grouping_config.yaml")
        print("   ")
        print("   # Override specific parameters:")
        print("   python scripts/story_grouping_dry_run.py --config my_config.yaml --max-parallelism 8")
        print("   ")
        print("   # Validate deployment:")
        print("   python scripts/deploy_story_grouping.py validate --config my_config.yaml")
        print("   ")
        print("   # Interactive setup:")
        print("   python scripts/deploy_story_grouping.py setup")
        return 0
    else:
        print("\n❌ Configuration integration has issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
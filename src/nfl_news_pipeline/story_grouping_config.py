"""Configuration management for story grouping feature.

This module provides comprehensive configuration management for all story grouping 
parameters, extending the existing pipeline configuration patterns to support 
story similarity grouping, deployment, and LLM API integration.

Implements Task 10: Create configuration and deployment setup
- Configuration management for all story grouping parameters
- Environment variable setup for LLM API keys and database connections
- Deployment-ready configuration structure
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .models import DefaultsConfig


logger = logging.getLogger(__name__)


class StoryGroupingConfigError(Exception):
    """Raised when story grouping configuration is invalid or missing."""
    pass


@dataclass
class LLMConfig:
    """Configuration for LLM providers used in story grouping."""
    
    provider: str = "openai"  # openai, google, deepseek
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 500
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3

    def validate(self) -> None:
        """Validate LLM configuration parameters."""
        if self.provider not in ["openai", "google", "deepseek"]:
            raise StoryGroupingConfigError(f"Unsupported LLM provider: {self.provider}")
        if self.max_tokens <= 0:
            raise StoryGroupingConfigError("max_tokens must be positive")
        if not (0.0 <= self.temperature <= 2.0):
            raise StoryGroupingConfigError("temperature must be between 0.0 and 2.0")
        if self.timeout_seconds <= 0:
            raise StoryGroupingConfigError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise StoryGroupingConfigError("max_retries must be non-negative")

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable."""
        return os.getenv(self.api_key_env)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 32
    cache_ttl_hours: int = 24
    normalize_vectors: bool = True

    def validate(self) -> None:
        """Validate embedding configuration parameters."""
        if self.dimension <= 0:
            raise StoryGroupingConfigError("dimension must be positive")
        if self.batch_size <= 0:
            raise StoryGroupingConfigError("batch_size must be positive")
        if self.cache_ttl_hours < 0:
            raise StoryGroupingConfigError("cache_ttl_hours must be non-negative")


@dataclass
class SimilarityConfig:
    """Configuration for similarity calculations."""
    
    threshold: float = 0.8
    metric: str = "cosine"  # cosine, euclidean, dot_product
    max_candidates: int = 100
    search_limit: int = 1000
    candidate_similarity_floor: float = 0.35

    def validate(self) -> None:
        """Validate similarity configuration parameters."""
        if not (0.0 <= self.threshold <= 1.0):
            raise StoryGroupingConfigError("threshold must be between 0.0 and 1.0")
        if self.metric not in ["cosine", "euclidean", "dot_product"]:
            raise StoryGroupingConfigError(f"Unsupported similarity metric: {self.metric}")
        if self.max_candidates <= 0:
            raise StoryGroupingConfigError("max_candidates must be positive")
        if self.search_limit <= 0:
            raise StoryGroupingConfigError("search_limit must be positive")
        if not (0.0 <= self.candidate_similarity_floor <= 1.0):
            raise StoryGroupingConfigError("candidate_similarity_floor must be between 0.0 and 1.0")


@dataclass
class GroupingConfig:
    """Configuration for group management."""
    
    max_group_size: int = 50
    centroid_update_threshold: int = 5
    status_transition_hours: int = 24
    auto_tagging: bool = True
    max_stories_per_run: Optional[int] = None
    prioritize_recent: bool = True
    prioritize_high_relevance: bool = True
    reprocess_existing: bool = False

    def validate(self) -> None:
        """Validate grouping configuration parameters."""
        if self.max_group_size <= 0:
            raise StoryGroupingConfigError("max_group_size must be positive")
        if self.centroid_update_threshold <= 0:
            raise StoryGroupingConfigError("centroid_update_threshold must be positive")
        if self.status_transition_hours < 0:
            raise StoryGroupingConfigError("status_transition_hours must be non-negative")
        if self.max_stories_per_run is not None and self.max_stories_per_run <= 0:
            raise StoryGroupingConfigError("max_stories_per_run must be positive or None")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    parallel_processing: bool = True
    max_workers: int = 4
    max_parallelism: int = 4
    batch_size: int = 10
    cache_size: int = 1000
    max_total_processing_time: Optional[float] = None

    def validate(self) -> None:
        """Validate performance configuration parameters."""
        if self.max_workers <= 0:
            raise StoryGroupingConfigError("max_workers must be positive")
        if self.max_parallelism <= 0:
            raise StoryGroupingConfigError("max_parallelism must be positive")
        if self.batch_size <= 0:
            raise StoryGroupingConfigError("batch_size must be positive")
        if self.cache_size < 0:
            raise StoryGroupingConfigError("cache_size must be non-negative")
        if self.max_total_processing_time is not None and self.max_total_processing_time <= 0:
            raise StoryGroupingConfigError("max_total_processing_time must be positive or None")


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    
    connection_timeout_seconds: int = 30
    max_connections: int = 10
    batch_insert_size: int = 100
    vector_index_maintenance: bool = True

    def validate(self) -> None:
        """Validate database configuration parameters."""
        if self.connection_timeout_seconds <= 0:
            raise StoryGroupingConfigError("connection_timeout_seconds must be positive")
        if self.max_connections <= 0:
            raise StoryGroupingConfigError("max_connections must be positive")
        if self.batch_insert_size <= 0:
            raise StoryGroupingConfigError("batch_insert_size must be positive")


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    
    log_level: str = "INFO"
    metrics_enabled: bool = True
    cost_tracking_enabled: bool = True
    performance_alerts_enabled: bool = True
    group_quality_monitoring: bool = True

    def validate(self) -> None:
        """Validate monitoring configuration parameters."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise StoryGroupingConfigError(f"log_level must be one of: {valid_log_levels}")


@dataclass
class StoryGroupingConfig:
    """Complete configuration for story grouping feature."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def validate(self) -> None:
        """Validate all configuration sections."""
        self.llm.validate()
        self.embedding.validate()
        self.similarity.validate()
        self.grouping.validate()
        self.performance.validate()
        self.database.validate()
        self.monitoring.validate()

    def get_orchestrator_settings(self):
        """Convert to StoryGroupingSettings for orchestrator compatibility."""
        from .orchestrator.story_grouping import StoryGroupingSettings
        
        return StoryGroupingSettings(
            max_parallelism=self.performance.max_parallelism,
            max_candidates=self.similarity.max_candidates,
            candidate_similarity_floor=self.similarity.candidate_similarity_floor,
            max_total_processing_time=self.performance.max_total_processing_time,
            max_stories_per_run=self.grouping.max_stories_per_run,
            prioritize_recent=self.grouping.prioritize_recent,
            prioritize_high_relevance=self.grouping.prioritize_high_relevance,
            reprocess_existing=self.grouping.reprocess_existing,
        )


class StoryGroupingConfigManager:
    """Manager for story grouping configuration loading and validation."""
    
    def __init__(self, config_path: Union[str, Path] = "story_grouping_config.yaml") -> None:
        self.config_path = Path(config_path)
        self._config: Optional[StoryGroupingConfig] = None
        self._warnings: List[str] = []

    def load_config(self) -> StoryGroupingConfig:
        """Load and validate story grouping configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Story grouping config file not found: {self.config_path}, using defaults")
            self._config = StoryGroupingConfig()
            self._config.validate()
            return self._config

        try:
            raw = yaml.safe_load(self.config_path.read_text()) or {}
        except yaml.YAMLError as e:
            raise StoryGroupingConfigError(f"Invalid YAML in {self.config_path}: {e}") from e

        # Parse configuration sections
        llm_config = self._parse_llm_config(raw.get("llm", {}))
        embedding_config = self._parse_embedding_config(raw.get("embedding", {}))
        similarity_config = self._parse_similarity_config(raw.get("similarity", {}))
        grouping_config = self._parse_grouping_config(raw.get("grouping", {}))
        performance_config = self._parse_performance_config(raw.get("performance", {}))
        database_config = self._parse_database_config(raw.get("database", {}))
        monitoring_config = self._parse_monitoring_config(raw.get("monitoring", {}))

        self._config = StoryGroupingConfig(
            llm=llm_config,
            embedding=embedding_config,
            similarity=similarity_config,
            grouping=grouping_config,
            performance=performance_config,
            database=database_config,
            monitoring=monitoring_config,
        )

        # Validate complete configuration
        self._config.validate()
        
        # Apply logging configuration
        self._apply_logging_config()
        
        return self._config

    def _parse_llm_config(self, data: Dict[str, Any]) -> LLMConfig:
        """Parse LLM configuration section."""
        return LLMConfig(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            api_key_env=data.get("api_key_env", "OPENAI_API_KEY"),
            max_tokens=int(data.get("max_tokens", 500)),
            temperature=float(data.get("temperature", 0.1)),
            timeout_seconds=int(data.get("timeout_seconds", 30)),
            max_retries=int(data.get("max_retries", 3)),
        )

    def _parse_embedding_config(self, data: Dict[str, Any]) -> EmbeddingConfig:
        """Parse embedding configuration section."""
        return EmbeddingConfig(
            model_name=data.get("model_name", "text-embedding-3-small"),
            dimension=int(data.get("dimension", 1536)),
            batch_size=int(data.get("batch_size", 32)),
            cache_ttl_hours=int(data.get("cache_ttl_hours", 24)),
            normalize_vectors=bool(data.get("normalize_vectors", True)),
        )

    def _parse_similarity_config(self, data: Dict[str, Any]) -> SimilarityConfig:
        """Parse similarity configuration section."""
        return SimilarityConfig(
            threshold=float(data.get("threshold", 0.8)),
            metric=data.get("metric", "cosine"),
            max_candidates=int(data.get("max_candidates", 100)),
            search_limit=int(data.get("search_limit", 1000)),
            candidate_similarity_floor=float(data.get("candidate_similarity_floor", 0.35)),
        )

    def _parse_grouping_config(self, data: Dict[str, Any]) -> GroupingConfig:
        """Parse grouping configuration section."""
        max_stories_per_run = data.get("max_stories_per_run")
        if max_stories_per_run is not None:
            max_stories_per_run = int(max_stories_per_run)
            
        return GroupingConfig(
            max_group_size=int(data.get("max_group_size", 50)),
            centroid_update_threshold=int(data.get("centroid_update_threshold", 5)),
            status_transition_hours=int(data.get("status_transition_hours", 24)),
            auto_tagging=bool(data.get("auto_tagging", True)),
            max_stories_per_run=max_stories_per_run,
            prioritize_recent=bool(data.get("prioritize_recent", True)),
            prioritize_high_relevance=bool(data.get("prioritize_high_relevance", True)),
            reprocess_existing=bool(data.get("reprocess_existing", False)),
        )

    def _parse_performance_config(self, data: Dict[str, Any]) -> PerformanceConfig:
        """Parse performance configuration section."""
        max_total_processing_time = data.get("max_total_processing_time")
        if max_total_processing_time is not None:
            max_total_processing_time = float(max_total_processing_time)
            
        return PerformanceConfig(
            parallel_processing=bool(data.get("parallel_processing", True)),
            max_workers=int(data.get("max_workers", 4)),
            max_parallelism=int(data.get("max_parallelism", 4)),
            batch_size=int(data.get("batch_size", 10)),
            cache_size=int(data.get("cache_size", 1000)),
            max_total_processing_time=max_total_processing_time,
        )

    def _parse_database_config(self, data: Dict[str, Any]) -> DatabaseConfig:
        """Parse database configuration section."""
        return DatabaseConfig(
            connection_timeout_seconds=int(data.get("connection_timeout_seconds", 30)),
            max_connections=int(data.get("max_connections", 10)),
            batch_insert_size=int(data.get("batch_insert_size", 100)),
            vector_index_maintenance=bool(data.get("vector_index_maintenance", True)),
        )

    def _parse_monitoring_config(self, data: Dict[str, Any]) -> MonitoringConfig:
        """Parse monitoring configuration section."""
        return MonitoringConfig(
            log_level=data.get("log_level", "INFO"),
            metrics_enabled=bool(data.get("metrics_enabled", True)),
            cost_tracking_enabled=bool(data.get("cost_tracking_enabled", True)),
            performance_alerts_enabled=bool(data.get("performance_alerts_enabled", True)),
            group_quality_monitoring=bool(data.get("group_quality_monitoring", True)),
        )

    def _apply_logging_config(self) -> None:
        """Apply logging configuration from monitoring config."""
        if self._config and self._config.monitoring.log_level:
            log_level = getattr(logging, self._config.monitoring.log_level, logging.INFO)
            logging.getLogger("nfl_news_pipeline.story_grouping").setLevel(log_level)

    def get_config(self) -> StoryGroupingConfig:
        """Get loaded configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config

    def get_warnings(self) -> List[str]:
        """Get configuration loading warnings."""
        return self._warnings.copy()

    def check_environment_variables(self) -> Dict[str, bool]:
        """Check if required environment variables are set."""
        if self._config is None:
            self.load_config()
            
        checks = {}
        
        # Required variables
        checks["SUPABASE_URL"] = os.getenv("SUPABASE_URL") is not None
        checks["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY") is not None
        
        # LLM API key based on provider
        if self._config:
            api_key_env = self._config.llm.api_key_env
            checks[api_key_env] = os.getenv(api_key_env) is not None
        
        return checks

    def validate_environment(self) -> List[str]:
        """Validate environment setup and return list of issues."""
        issues = []
        env_checks = self.check_environment_variables()
        
        for var_name, is_set in env_checks.items():
            if not is_set:
                if var_name in ["SUPABASE_URL", "SUPABASE_KEY"]:
                    issues.append(f"Required environment variable {var_name} is not set")
                else:
                    issues.append(f"LLM API key {var_name} is not set (required for story grouping)")
        
        return issues


def get_story_grouping_config(config_path: Optional[Union[str, Path]] = None) -> StoryGroupingConfig:
    """Convenience function to get story grouping configuration."""
    manager = StoryGroupingConfigManager(config_path or "story_grouping_config.yaml")
    return manager.load_config()
"""NFL News Processing Pipeline core package."""

from .models import (
	NewsItem,
	ProcessedNewsItem,
	SourceWatermark,
	FilterResult,
	FeedConfig,
	DefaultsConfig,
	PipelineConfig,
)
from .filters import RuleBasedFilter, LLMFilter
from .storage import StorageManager, StorageResult
from .logging import AuditLogger
from .orchestrator import NFLNewsPipeline, PipelineSummary

__all__ = [
	"NewsItem",
	"ProcessedNewsItem",
	"SourceWatermark",
	"FilterResult",
	"FeedConfig",
	"DefaultsConfig",
	"PipelineConfig",
	"RuleBasedFilter",
	"LLMFilter",
	"StorageManager",
	"StorageResult",
	"AuditLogger",
	"NFLNewsPipeline",
	"PipelineSummary",
]

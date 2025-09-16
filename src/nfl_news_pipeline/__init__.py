"""NFL News Processing Pipeline core package."""

from .models import (
	NewsItem,
	ProcessedNewsItem,
	SourceWatermark,
	FilterResult,
	FeedConfig,
	DefaultsConfig,
	PipelineConfig,
	# Story grouping models
	ContextSummary,
	StoryEmbedding,
	StoryGroup,
	StoryGroupMember,
	GroupCentroid,
	GroupStatus,
	EMBEDDING_DIM,
)
from .filters import RuleBasedFilter, LLMFilter
from .storage import StorageManager, StorageResult
from .logging import AuditLogger
from .orchestrator import NFLNewsPipeline, PipelineSummary
from .similarity import SimilarityCalculator, SimilarityMetric, SimilarityResult
from .centroid_manager import GroupCentroidManager, CentroidUpdateResult

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
	# Story grouping exports
	"ContextSummary",
	"StoryEmbedding",
	"StoryGroup",
	"StoryGroupMember",
	"GroupCentroid",
	"GroupStatus",
	"EMBEDDING_DIM",
	"SimilarityCalculator",
	"SimilarityMetric",
	"SimilarityResult",
	"GroupCentroidManager",
	"CentroidUpdateResult",
]

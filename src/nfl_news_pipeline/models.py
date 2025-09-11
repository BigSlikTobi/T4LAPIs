from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class NewsItem:
    """Raw URL + metadata extracted from a feed/sitemap.

    Metadata-only per requirements; no full article content.
    """

    url: str
    title: str
    publication_date: datetime
    source_name: str
    publisher: str
    description: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedNewsItem(NewsItem):
    """NewsItem after filtering and enrichment."""

    relevance_score: float = 0.0
    filter_method: str = "rule_based"  # or 'llm'
    filter_reasoning: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SourceWatermark:
    source_name: str
    last_processed_date: datetime
    last_successful_run: datetime
    items_processed: int = 0
    errors_count: int = 0


@dataclass
class FilterResult:
    is_relevant: bool
    confidence_score: float
    reasoning: str
    method: str  # 'rule_based' or 'llm'


@dataclass
class FeedConfig:
    """Configuration for one source defined in feeds.yaml.

    Fields cover both RSS and Sitemap sources per design.
    """

    name: str
    type: str  # 'rss' | 'sitemap' | 'html'
    publisher: str
    enabled: bool = True
    nfl_only: bool = False
    url: Optional[str] = None
    url_template: Optional[str] = None
    max_articles: Optional[int] = None
    days_back: Optional[int] = None
    extract_content: bool = False  # must remain False for compliance


@dataclass
class DefaultsConfig:
    user_agent: str = "T4LAPIs-NFLNewsPipeline/1.0"
    timeout_seconds: int = 15
    max_parallel_fetches: int = 4


@dataclass
class PipelineConfig:
    defaults: DefaultsConfig
    sources: List[FeedConfig]


__all__ = [
    "NewsItem",
    "ProcessedNewsItem",
    "SourceWatermark",
    "FilterResult",
    "FeedConfig",
    "DefaultsConfig",
    "PipelineConfig",
]

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
    
    # Story grouping configuration
    enable_story_grouping: bool = False
    story_grouping_max_parallelism: int = 4
    story_grouping_max_stories_per_run: Optional[int] = None
    story_grouping_reprocess_existing: bool = False


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
    # Story grouping exports
    "EMBEDDING_DIM",
    "GroupStatus",
    "ContextSummary",
    "StoryEmbedding",
    "StoryGroup",
    "StoryGroupMember",
    "GroupCentroid",
]


# ==========================
# Story similarity grouping
# ==========================

# Keep in sync with DB schema (story_embeddings.vector and story_groups.centroid_embedding)
EMBEDDING_DIM: int = 1536


class GroupStatus(Enum):
    NEW = "new"
    UPDATED = "updated"
    STABLE = "stable"

    @classmethod
    def from_str(cls, value: str) -> "GroupStatus":
        v = (value or "").strip().lower()
        if v in (cls.NEW.value, cls.UPDATED.value, cls.STABLE.value):
            return cls(v)
        raise ValueError(f"Invalid GroupStatus: {value}")


@dataclass
class ContextSummary:
    """LLM-generated context summary for a news URL.

    Mirrors `context_summaries` table.
    """

    news_url_id: str
    summary_text: str
    llm_model: str
    confidence_score: float
    # Optional token usage details for cost estimation (not persisted unless wired)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    entities: Optional[Dict[str, Any]] = None
    key_topics: List[str] = field(default_factory=list)
    fallback_used: bool = False
    generated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[str] = None

    def validate(self) -> None:
        if not self.news_url_id:
            raise ValueError("news_url_id is required")
        if not self.summary_text or not self.summary_text.strip():
            raise ValueError("summary_text is required")
        if not self.llm_model or not self.llm_model.strip():
            raise ValueError("llm_model is required")
        if not (0.0 <= float(self.confidence_score) <= 1.0):
            raise ValueError("confidence_score must be in [0,1]")

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "ContextSummary":
        return cls(
            id=row.get("id"),
            news_url_id=row["news_url_id"],
            summary_text=row["summary_text"],
            entities=row.get("entities"),
            key_topics=row.get("key_topics") or [],
            llm_model=row["llm_model"],
            confidence_score=float(row.get("confidence_score", 0.0)),
            fallback_used=bool(row.get("fallback_used", False)),
            generated_at=_parse_dt(row.get("generated_at")),
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
        )

    def to_db(self) -> Dict[str, Any]:
        self.validate()
        return {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in {
                "id": self.id,
                "news_url_id": self.news_url_id,
                "summary_text": self.summary_text,
                "entities": self.entities or {},
                "key_topics": self.key_topics,
                "llm_model": self.llm_model,
                "confidence_score": float(self.confidence_score),
                "fallback_used": bool(self.fallback_used),
                "generated_at": self.generated_at,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }.items()
            if v is not None
        }


@dataclass
class StoryEmbedding:
    """Semantic embedding for a story summary.

    Mirrors `story_embeddings` table.
    """

    news_url_id: str
    embedding_vector: List[float]
    model_name: str
    model_version: str
    summary_text: str
    confidence_score: float
    generated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    id: Optional[str] = None

    def validate(self) -> None:
        if not self.news_url_id:
            raise ValueError("news_url_id is required")
        if not isinstance(self.embedding_vector, list):
            raise ValueError("embedding_vector must be a list of floats")
        if len(self.embedding_vector) != EMBEDDING_DIM:
            raise ValueError(f"embedding_vector must have length {EMBEDDING_DIM}")
        # Ensure they are floats
        try:
            _ = [float(x) for x in self.embedding_vector]
        except Exception as e:
            raise ValueError("embedding_vector contains non-float values") from e
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name is required")
        if not self.model_version or not self.model_version.strip():
            raise ValueError("model_version is required")
        if not self.summary_text or not self.summary_text.strip():
            raise ValueError("summary_text is required")
        if not (0.0 <= float(self.confidence_score) <= 1.0):
            raise ValueError("confidence_score must be in [0,1]")

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "StoryEmbedding":
        return cls(
            id=row.get("id"),
            news_url_id=row["news_url_id"],
            embedding_vector=list(row.get("embedding_vector") or []),
            model_name=row["model_name"],
            model_version=row["model_version"],
            summary_text=row["summary_text"],
            confidence_score=float(row.get("confidence_score", 0.0)),
            generated_at=_parse_dt(row.get("generated_at")),
            created_at=_parse_dt(row.get("created_at")),
        )

    def to_db(self) -> Dict[str, Any]:
        self.validate()
        return {
            k: v
            for k, v in {
                "id": self.id,
                "news_url_id": self.news_url_id,
                "embedding_vector": self.embedding_vector,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "summary_text": self.summary_text,
                "confidence_score": float(self.confidence_score),
                "generated_at": self.generated_at,
                "created_at": self.created_at,
            }.items()
            if v is not None
        }


@dataclass
class StoryGroup:
    """Story grouping metadata and centroid.

    Mirrors `story_groups` table.
    """

    member_count: int
    status: GroupStatus
    tags: List[str] = field(default_factory=list)
    centroid_embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    id: Optional[str] = None

    def validate(self) -> None:
        if self.member_count < 1:
            raise ValueError("member_count must be >= 1")
        if not isinstance(self.status, GroupStatus):
            raise ValueError("status must be a GroupStatus enum")
        if self.centroid_embedding is not None:
            if len(self.centroid_embedding) != EMBEDDING_DIM:
                raise ValueError(f"centroid_embedding must have length {EMBEDDING_DIM}")
            try:
                _ = [float(x) for x in self.centroid_embedding]
            except Exception as e:
                raise ValueError("centroid_embedding contains non-float values") from e

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "StoryGroup":
        centroid = row.get("centroid_embedding")
        status_value = row.get("status")
        return cls(
            id=row.get("id"),
            centroid_embedding=list(centroid) if centroid is not None else None,
            member_count=int(row.get("member_count", 0)),
            status=GroupStatus.from_str(status_value) if isinstance(status_value, str) else status_value,
            tags=list(row.get("tags") or []),
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
        )

    def to_db(self) -> Dict[str, Any]:
        self.validate()
        return {
            k: v
            for k, v in {
                "id": self.id,
                "centroid_embedding": [float(x) for x in self.centroid_embedding] if self.centroid_embedding is not None else None,
                "member_count": int(self.member_count),
                "status": self.status.value,
                "tags": self.tags,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }.items()
            if v is not None
        }


@dataclass
class StoryGroupMember:
    """Membership row linking a story to a group.

    Mirrors `story_group_members` table.
    """

    group_id: str
    news_url_id: str
    similarity_score: float
    added_at: Optional[datetime] = None
    id: Optional[str] = None

    def validate(self) -> None:
        if not self.group_id:
            raise ValueError("group_id is required")
        if not self.news_url_id:
            raise ValueError("news_url_id is required")
        if not (0.0 <= float(self.similarity_score) <= 1.0):
            raise ValueError("similarity_score must be in [0,1]")

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "StoryGroupMember":
        return cls(
            id=row.get("id"),
            group_id=row["group_id"],
            news_url_id=row["news_url_id"],
            similarity_score=float(row.get("similarity_score", 0.0)),
            added_at=_parse_dt(row.get("added_at")),
        )

    def to_db(self) -> Dict[str, Any]:
        self.validate()
        return {
            k: v
            for k, v in {
                "id": self.id,
                "group_id": self.group_id,
                "news_url_id": self.news_url_id,
                "similarity_score": float(self.similarity_score),
                "added_at": self.added_at,
            }.items()
            if v is not None
        }


@dataclass
class GroupCentroid:
    """Convenience model for centroid-based similarity comparisons."""

    group_id: str
    centroid_vector: List[float]
    member_count: int
    last_updated: Optional[datetime] = None

    def validate(self) -> None:
        if not self.group_id:
            raise ValueError("group_id is required")
        if len(self.centroid_vector) != EMBEDDING_DIM:
            raise ValueError(f"centroid_vector must have length {EMBEDDING_DIM}")
        try:
            _ = [float(x) for x in self.centroid_vector]
        except Exception as e:
            raise ValueError("centroid_vector contains non-float values") from e
        if self.member_count < 1:
            raise ValueError("member_count must be >= 1")

    @classmethod
    def from_group(cls, group: StoryGroup) -> "GroupCentroid":
        if group.centroid_embedding is None:
            raise ValueError("group has no centroid_embedding")
        return cls(
            group_id=group.id or "",
            centroid_vector=list(group.centroid_embedding),
            member_count=group.member_count,
            last_updated=group.updated_at,
        )


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse a datetime if value is a string; pass through datetime; else None.

    Supabase returns timestamps as ISO strings; tests may pass datetime objects.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Basic ISO-8601 handling
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None

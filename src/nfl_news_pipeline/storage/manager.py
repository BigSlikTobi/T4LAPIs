from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..models import ProcessedNewsItem
from .protocols import StoryGroupingCapable


@dataclass
class StorageResult:
    inserted_count: int
    updated_count: int
    errors_count: int
    ids_by_url: Dict[str, str]


class StorageManager(StoryGroupingCapable):
    """Storage layer for Supabase with URL dedup and watermarks.

    Notes:
    - This class depends on a Supabase client instance (supabase-py >= 2.x).
    - We avoid direct SQL; instead, use PostgREST ops exposed by the client.
    - For migrations, see the SQL files under db/migrations/.
    """

    BATCH_SIZE = 500

    def __init__(self, client: Any):
        self.client = client
        self.table_news = os.getenv("NEWS_URLS_TABLE", "news_urls")
        self.table_watermarks = os.getenv("SOURCE_WATERMARKS_TABLE", "source_watermarks")
        self.table_audit = os.getenv("PIPELINE_AUDIT_LOG_TABLE", "pipeline_audit_log")
        self.table_filter_decisions = os.getenv("FILTER_DECISIONS_TABLE", "filter_decisions")
        self.table_news_entities = os.getenv("NEWS_URL_ENTITIES_TABLE", "news_url_entities")
        # Players / teams source tables for contextual disambiguation
        self.table_players = os.getenv("PLAYERS_TABLE", "players")
        self.table_teams = os.getenv("TEAMS_TABLE", "teams")
        self.player_team_col = os.getenv("PLAYER_TEAM_COLUMN", "latest_team")  # fallback to 'team' if not present
        # Player name disambiguation behavior
        self.enable_player_team_filter = os.getenv("ENABLE_PLAYER_TEAM_FILTER", "1").lower() not in {"0", "false", "no"}
        # If a player name maps to multiple players across different teams within the filtered subset, skip by default
        self.keep_ambiguous_players = os.getenv("KEEP_AMBIGUOUS_PLAYERS", "0").lower() in {"1", "true", "yes"}
        # Back-compat flag: also write JSONB entities column (default off per normalization)
        self.write_entities_jsonb = os.getenv("WRITE_ENTITIES_JSONB", "0").lower() not in {"0", "false", "no"}

        # Lazy-populated caches for player/team lookups
        self._player_cache_loaded = False
        self._player_name_index: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        self._player_id_to_team: Dict[str, Optional[str]] = {}
        self._team_cache_loaded = False
        self._team_lookup: Dict[str, str] = {}
        self._team_abbr_set: Set[str] = set()

    def get_grouping_client(self) -> Any:
        """Get the Supabase client for story grouping operations.
        
        Returns:
            The Supabase client instance
        """
        return self.client

    # ------------- Dedup helpers -------------
    def check_duplicate_urls(self, urls: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Return mapping of existing URL -> row for given URLs."""
        url_list = list({u for u in urls if u})
        if not url_list:
            return {}
        resp = self.client.table(self.table_news).select("id,url,publication_date,updated_at").in_("url", url_list).execute()
        rows = getattr(resp, "data", []) or []
        return {r["url"]: r for r in rows if r.get("url")}

    # ------------- Core store -------------
    def store_news_items(self, items: List[ProcessedNewsItem]) -> StorageResult:
        """Insert/update news items by URL.

        - Inserts new URLs
        - Updates existing URLs with latest metadata
        - Returns counts and id mapping
        """
        ids_by_url: Dict[str, str] = {}
        inserted, updated, errors = 0, 0, 0
        if not items:
            return StorageResult(inserted, updated, errors, ids_by_url)

        # Lookup existing by URL
        urls = [i.url for i in items]
        existing_map = self.check_duplicate_urls(urls)

        # Partition new vs existing
        new_rows: List[Dict[str, Any]] = []
        updates: List[Tuple[str, Dict[str, Any]]] = []  # (url, payload)

        for it in items:
            payload = self._item_to_row(it)
            if it.url in existing_map:
                # Update existing
                updates.append((it.url, payload))
            else:
                new_rows.append(payload)

        # Batch insert new
        if new_rows:
            try:
                resp = self.client.table(self.table_news).insert(new_rows).execute()
                rows = getattr(resp, "data", []) or []
                for r in rows:
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
                inserted += len(rows)
            except Exception:
                errors += len(new_rows)

        # Update existing one-by-one to keep it simple and predictable
        for url, payload in updates:
            try:
                resp = self.client.table(self.table_news).update(payload).eq("url", url).execute()
                rows = getattr(resp, "data", []) or []
                if rows:
                    r = rows[0]
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
                updated += 1
            except Exception:
                errors += 1

        # For any URLs with no id from the responses, fetch ids to complete mapping
        missing = [u for u in urls if u not in ids_by_url]
        if missing:
            try:
                resp = self.client.table(self.table_news).select("id,url").in_("url", missing).execute()
                rows = getattr(resp, "data", []) or []
                for r in rows:
                    if r.get("url") and r.get("id"):
                        ids_by_url[r["url"]] = r["id"]
            except Exception:
                pass

        # Populate normalized entity table for all items
        try:
            self._store_entities_normalized(items, ids_by_url)
        except Exception:
            # Do not fail the main store on entity mapping issues
            pass

        return StorageResult(inserted, updated, errors, ids_by_url)

    def _item_to_row(self, it: ProcessedNewsItem) -> Dict[str, Any]:
        row = {
            "url": it.url,
            "title": it.title,
            "description": it.description,
            "publication_date": self._to_utc_iso(it.publication_date),
            "source_name": it.source_name,
            "publisher": it.publisher,
            "relevance_score": float(it.relevance_score),
            "filter_method": it.filter_method,
            "filter_reasoning": it.filter_reasoning,
            # Keep JSONB/write optional for back-compat
            "entities": (it.entities or []) if self.write_entities_jsonb else None,
            "categories": it.categories or [],
            "raw_metadata": it.raw_metadata or {},
        }
        if not self.write_entities_jsonb:
            row.pop("entities", None)
        return row

    def _store_entities_normalized(self, items: List[ProcessedNewsItem], ids_by_url: Dict[str, str]) -> None:
        if not items or not ids_by_url:
            return
        # Build rows for teams, players, and topics
        rows: List[Dict[str, Any]] = []
        url_ids_with_rows: Set[str] = set()
        for it in items:
            url_id = ids_by_url.get(it.url)
            if not url_id:
                continue
            seen_for_url: Set[Tuple[str, str]] = set()
            # Topics from categories
            for topic in (it.categories or []):
                if topic:
                    key = ("topic", str(topic))
                    if key in seen_for_url:
                        continue
                    rows.append({
                        "news_url_id": url_id,
                        "entity_type": "topic",
                        "entity_value": str(topic),
                    })
                    seen_for_url.add(key)
                    url_ids_with_rows.add(url_id)
            # Teams/players from raw_metadata.entity_tags (preferred) else flat entities
            players: List[str] = []
            teams: List[str] = []
            try:
                tags = (it.raw_metadata or {}).get("entity_tags") or {}
                players = list(tags.get("players", []) or [])
                teams = list(tags.get("teams", []) or [])
            except Exception:
                players, teams = [], []

            # Prefer categorization from entity_tags in raw_metadata if available.
            # Fallback: if flat entities present, best-effort partition by simple heuristic:
            # Entities that are all uppercase and 2-4 characters are considered teams, others as players.
            if not players and not teams and it.entities:
                for e in it.entities:
                    if isinstance(e, str) and e.isupper() and 2 <= len(e) <= 4:
                        teams.append(e)
                    else:
                        players.append(str(e))

            # Resolve teams to canonical team_abbr values
            normalized_teams = self._resolve_team_abbreviations(teams)
            for t in normalized_teams:
                key = ("team", t)
                if key in seen_for_url:
                    continue
                rows.append({
                    "news_url_id": url_id,
                    "entity_type": "team",
                    "entity_value": t,
                })
                seen_for_url.add(key)
                url_ids_with_rows.add(url_id)

            # Resolve player identifiers to player_id strings
            enforce_team = self.enable_player_team_filter and bool(normalized_teams)
            try:
                normalized_players = self._resolve_player_ids(players, normalized_teams if normalized_teams else teams, enforce_team=enforce_team)
            except Exception:
                normalized_players = []

            for p in normalized_players:
                key = ("player", p)
                if key in seen_for_url:
                    continue
                rows.append({
                    "news_url_id": url_id,
                    "entity_type": "player",
                    "entity_value": p,
                })
                seen_for_url.add(key)
                url_ids_with_rows.add(url_id)

        # Refresh existing mappings only for URLs where we have rows to write
        if url_ids_with_rows:
            id_list = list(url_ids_with_rows)
            try:
                for i in range(0, len(id_list), self.BATCH_SIZE):
                    chunk = id_list[i:i+self.BATCH_SIZE]
                    self.client.table(self.table_news_entities).delete().in_("news_url_id", chunk).execute()
            except Exception:
                # Best effort; continue with inserts
                pass
        if rows:
            # Insert in batches to avoid large payloads
            for i in range(0, len(rows), self.BATCH_SIZE):
                chunk = rows[i:i+self.BATCH_SIZE]
                try:
                    self.client.table(self.table_news_entities).insert(chunk).execute()
                except Exception:
                    # Best effort: continue with next chunk
                    continue

    # -------- Player disambiguation helpers --------
    def _filter_players_by_teams(self, extracted_players: List[str], extracted_teams: List[str]) -> List[str]:
        """Return player_ids constrained to the supplied team abbreviations."""
        return self._resolve_player_ids(extracted_players, extracted_teams, enforce_team=True)

    # -------- Player/team resolution helpers --------
    def _resolve_player_ids(
        self,
        extracted_players: List[str],
        extracted_teams: Iterable[str],
        *,
        enforce_team: bool,
    ) -> List[str]:
        players = [str(p).strip() for p in extracted_players if p]
        if not players:
            return []

        teams_upper = {str(t).strip().upper() for t in extracted_teams if t}
        if not teams_upper:
            enforce_team = False  # Nothing to enforce against

        try:
            self._ensure_player_cache()
        except Exception:
            return []

        resolved: List[str] = []
        seen: Set[str] = set()
        for original in players:
            if not original:
                continue
            if self._looks_like_player_id(original):
                player_id = original
                team = (self._player_id_to_team.get(player_id) or "").upper()
                if enforce_team and teams_upper and team and team not in teams_upper:
                    continue
                if player_id not in seen:
                    resolved.append(player_id)
                    seen.add(player_id)
                continue

            key = original.lower()
            matches = self._player_name_index.get(key, [])
            if not matches:
                continue

            filtered_matches = matches
            if teams_upper:
                filtered_matches = [m for m in matches if (m[1] or "").upper() in teams_upper]
                if not filtered_matches and not enforce_team:
                    filtered_matches = matches

            if not filtered_matches:
                continue

            if len(filtered_matches) == 1:
                candidate = filtered_matches[0][0]
                if candidate not in seen:
                    resolved.append(candidate)
                    seen.add(candidate)
                continue

            if self.keep_ambiguous_players or not enforce_team:
                candidate = filtered_matches[0][0]
                if candidate not in seen:
                    resolved.append(candidate)
                    seen.add(candidate)

        return resolved

    def _resolve_team_abbreviations(self, extracted_teams: List[str]) -> List[str]:
        if not extracted_teams:
            return []

        cache_loaded = True
        try:
            self._ensure_team_cache()
        except Exception:
            cache_loaded = False

        resolved: List[str] = []
        seen: Set[str] = set()
        for original in extracted_teams:
            value = str(original or "").strip()
            if not value:
                continue
            upper = value.upper()
            abbr: Optional[str] = None
            if upper in self._team_abbr_set:
                abbr = upper
            else:
                abbr = self._team_lookup.get(value.lower())
            if not abbr and value.isupper() and 2 <= len(value) <= 4:
                # Fallback: treat already-canonical abbreviations as valid even without cache
                abbr = upper
            if not abbr and not cache_loaded:
                # If we failed to load the cache entirely, fall back to uppercase heuristic
                if value.isalpha() and 2 <= len(value) <= 4:
                    abbr = upper
            if abbr and abbr not in seen:
                resolved.append(abbr)
                seen.add(abbr)
        return resolved

    def _ensure_player_cache(self) -> None:
        if self._player_cache_loaded:
            return

        name_index: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        id_to_team: Dict[str, Optional[str]] = {}

        offset = 0
        batch_size = 1000
        select_columns = "player_id,full_name,display_name,first_name,last_name,common_first_name,latest_team,team"

        while True:
            query = self.client.table(self.table_players).select(select_columns)
            try:
                resp = query.range(offset, offset + batch_size - 1).execute()
            except AttributeError:
                resp = query.execute()
            except Exception as exc:
                # Capture unexpected API errors so callers can decide on fallback behavior
                raise exc
            rows = getattr(resp, "data", []) or []
            if not rows:
                break

            for row in rows:
                player_id = row.get("player_id")
                if not player_id:
                    continue
                team_val = (
                    row.get(self.player_team_col)
                    or row.get("team")
                    or row.get("latest_team")
                    or row.get("team_abbr")
                    or row.get("team_code")
                )
                team_upper = str(team_val).strip().upper() if team_val else None
                id_to_team[player_id] = team_upper

                for name in self._player_name_variants(row):
                    key = name.lower()
                    name_index.setdefault(key, []).append((player_id, team_upper))

            if len(rows) < batch_size:
                break
            offset += batch_size

        self._player_name_index = name_index
        self._player_id_to_team = id_to_team
        self._player_cache_loaded = True

    def _ensure_team_cache(self) -> None:
        if self._team_cache_loaded:
            return

        lookup: Dict[str, str] = {}
        abbrs: Set[str] = set()
        resp = self.client.table(self.table_teams).select("team_abbr,team_name,team_nick,city").execute()
        rows = getattr(resp, "data", []) or []
        for row in rows:
            abbr = row.get("team_abbr")
            if not abbr:
                continue
            abbr_upper = str(abbr).strip().upper()
            if not abbr_upper:
                continue
            abbrs.add(abbr_upper)
            for name in self._team_name_variants(row, abbr_upper):
                lookup[name.lower()] = abbr_upper

        self._team_lookup = lookup
        self._team_abbr_set = abbrs
        self._team_cache_loaded = True

    def _player_name_variants(self, row: Dict[str, Any]) -> Set[str]:
        variants: Set[str] = set()

        def add(value: Any) -> None:
            if not value:
                return
            text = str(value).strip()
            if text:
                variants.add(text)

        add(row.get("full_name"))
        add(row.get("display_name"))
        first = row.get("first_name")
        last = row.get("last_name")
        common = row.get("common_first_name")
        if first and last:
            add(f"{first} {last}")
        if common and last:
            add(f"{common} {last}")
        add(last)

        return variants

    def _team_name_variants(self, row: Dict[str, Any], abbr_upper: str) -> Set[str]:
        variants: Set[str] = {abbr_upper}

        for key in ("team_name", "team_nick", "city"):
            value = row.get(key)
            if not value:
                continue
            text = str(value).strip()
            if text:
                variants.add(text)

        return variants

    def _looks_like_player_id(self, value: str) -> bool:
        v = value.strip()
        return bool(v) and "-" in v and any(ch.isdigit() for ch in v)

    # ------------- Filter decision logging -------------
    def log_filter_decision(
        self,
        news_url_id: str,
        *,
        method: str,
        stage: str,
        confidence: float,
        reasoning: str,
        model_id: Optional[str] = None,
    ) -> bool:
        payload = {
            "news_url_id": news_url_id,
            "method": method,
            "stage": stage,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "model_id": model_id,
        }
        try:
            self.client.table(self.table_filter_decisions).insert(payload).execute()
            return True
        except Exception:
            return False

    def add_audit_event(
        self,
        event_type: str,
        *,
        pipeline_run_id: Optional[str] = None,
        source_name: Optional[str] = None,
        message: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = {
            "pipeline_run_id": pipeline_run_id,
            "source_name": source_name,
            "event_type": event_type,
            "message": message,
            "event_data": event_data or {},
        }
        try:
            self.client.table(self.table_audit).insert(payload).execute()
            return True
        except Exception:
            return False

    # ------------- Watermarks -------------
    def get_watermark(self, source_name: str) -> Optional[datetime]:
        try:
            resp = self.client.table(self.table_watermarks).select("last_processed_date").eq("source_name", source_name).limit(1).execute()
            rows = getattr(resp, "data", []) or []
            if not rows:
                return None
            val = rows[0].get("last_processed_date")
            if not val:
                return None
            # Supabase returns ISO strings
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except Exception:
            return None

    def update_watermark(
        self,
        source_name: str,
        *,
        last_processed_date: datetime,
        last_successful_run: Optional[datetime] = None,
        items_processed: Optional[int] = None,
        errors_count: Optional[int] = None,
    ) -> bool:
        payload = {
            "source_name": source_name,
            "last_processed_date": self._to_utc_iso(last_processed_date),
        }
        if last_successful_run is not None:
            payload["last_successful_run"] = self._to_utc_iso(last_successful_run)
        if items_processed is not None:
            payload["items_processed"] = int(items_processed)
        if errors_count is not None:
            payload["errors_count"] = int(errors_count)

        try:
            # Upsert by primary key source_name
            self.client.table(self.table_watermarks).upsert(payload).execute()
            return True
        except Exception:
            return False

    def filter_new_items(self, items: List[ProcessedNewsItem], source_name: str) -> List[ProcessedNewsItem]:
        """Return only items with publication_date > stored watermark."""
        wm = self.get_watermark(source_name)
        if not wm:
            return items
        return [i for i in items if (i.publication_date or datetime.min.replace(tzinfo=timezone.utc)) > wm]

    # ------------- Utils -------------
    @staticmethod
    def _to_utc_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..models import ProcessedNewsItem


@dataclass
class StorageResult:
    inserted_count: int
    updated_count: int
    errors_count: int
    ids_by_url: Dict[str, str]


class StorageManager:
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
        self.player_team_col = os.getenv("PLAYER_TEAM_COLUMN", "latest_team")  # fallback to 'team' if not present
        # Player name disambiguation behavior
        self.enable_player_team_filter = os.getenv("ENABLE_PLAYER_TEAM_FILTER", "1").lower() not in {"0", "false", "no"}
        # If a player name maps to multiple players across different teams within the filtered subset, skip by default
        self.keep_ambiguous_players = os.getenv("KEEP_AMBIGUOUS_PLAYERS", "0").lower() in {"1", "true", "yes"}
        # Back-compat flag: also write JSONB entities column (default off per normalization)
        self.write_entities_jsonb = os.getenv("WRITE_ENTITIES_JSONB", "0").lower() not in {"0", "false", "no"}

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
        # Refresh existing mappings for these URLs to avoid duplicates on updates
        id_list = [v for k, v in ids_by_url.items() if k]
        try:
            for i in range(0, len(id_list), self.BATCH_SIZE):
                chunk = id_list[i:i+self.BATCH_SIZE]
                self.client.table(self.table_news_entities).delete().in_("news_url_id", chunk).execute()
        except Exception:
            # Best effort; continue with inserts
            pass
        # Build rows for teams, players, and topics
        rows: List[Dict[str, Any]] = []
        for it in items:
            url_id = ids_by_url.get(it.url)
            if not url_id:
                continue
            # Topics from categories
            for topic in (it.categories or []):
                if topic:
                    rows.append({
                        "news_url_id": url_id,
                        "entity_type": "topic",
                        "entity_value": str(topic),
                    })
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

            # --- Player disambiguation using team context ---
            if self.enable_player_team_filter and players and teams:
                try:
                    players = self._filter_players_by_teams(players, teams)
                except Exception:
                    # Fallback: retain original players list on any failure
                    pass

            for t in teams:
                if t:
                    rows.append({
                        "news_url_id": url_id,
                        "entity_type": "team",
                        "entity_value": str(t),
                    })
            for p in players:
                if p:
                    rows.append({
                        "news_url_id": url_id,
                        "entity_type": "player",
                        "entity_value": str(p),
                    })
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
        """Return a filtered list of player names constrained to the extracted teams.

        Strategy:
        1. Fetch the roster subset for the extracted team abbreviations.
        2. Build an index mapping lowercase full_name / first+last combos -> set of (team, canonical_full_name).
        3. For every extracted player name, keep it only if it appears in the index AND either:
           - It maps uniquely to a single team (unambiguous) OR
           - Ambiguous retention is enabled via KEEP_AMBIGUOUS_PLAYERS.
        4. Return canonical full_name (preferred) for consistency.

        Fallback: if any error occurs we return the original list.
        """
        players = [p for p in extracted_players if p]
        teams = [t for t in extracted_teams if t]
        if not players or not teams:
            return players

        # Deduplicate inputs
        teams_set = list({t.upper() for t in teams})
        name_index: Dict[str, List[Tuple[str, str]]] = {}

        # Attempt primary column; if fails, fall back to 'team'
        select_cols_primary = f"player_id,full_name,first_name,last_name,{self.player_team_col}"
        fallback_used = False
        try:
            resp = self.client.table(self.table_players).select(select_cols_primary).in_(self.player_team_col, teams_set).execute()
            roster_rows = getattr(resp, "data", []) or []
        except Exception:
            # Fallback attempt
            try:
                fallback_used = True
                team_col = "team"
                resp = self.client.table(self.table_players).select(f"player_id,full_name,first_name,last_name,{team_col}").in_(team_col, teams_set).execute()
                roster_rows = getattr(resp, "data", []) or []
                self.player_team_col = team_col  # Update for future calls
            except Exception:
                return players  # Give up silently

        for r in roster_rows:
            team_val = r.get(self.player_team_col) if not fallback_used else r.get("team")
            if not team_val:
                continue
            full_name = r.get("full_name") or None
            first_name = r.get("first_name") or None
            last_name = r.get("last_name") or None
            candidates = set()
            if full_name:
                candidates.add(str(full_name).strip())
            if first_name and last_name:
                candidates.add(f"{first_name} {last_name}".strip())
            for nm in candidates:
                key = nm.lower()
                name_index.setdefault(key, []).append((str(team_val).upper(), full_name or nm))

        filtered: List[str] = []
        for original in players:
            key = original.lower()
            matches = name_index.get(key)
            if not matches:
                # Player not in roster subset; drop to reduce false positives
                continue
            teams_for_name = {t for t, _ in matches}
            # Treat multiple roster rows as ambiguous even if same team appears duplicated
            if len(matches) == 1:
                canonical = matches[0][1]
                filtered.append(canonical)
                continue
            # Multiple teams -> ambiguous
            if self.keep_ambiguous_players:
                # Include once using first canonical reference
                canonical = matches[0][1]
                filtered.append(canonical)
            # else skip (dropped)

        # Do NOT fall back to original list if all were ambiguous; explicit empty conveys uncertainty.
        return filtered

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

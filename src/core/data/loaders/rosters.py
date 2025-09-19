"""Rosters data loader for populating the Supabase Rosters table."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import pandas as pd

from .base import BaseDataLoader
from ..fetch import fetch_seasonal_roster_data
from ..transform import BaseDataTransformer, RosterDataTransformer


class RostersDataLoader(BaseDataLoader):
    """Load seasonal roster data into the Supabase Rosters table."""

    def __init__(self, table_name: str = "Rosters") -> None:
        super().__init__(table_name)

    @property
    def transformer_class(self) -> Type[BaseDataTransformer]:
        return RosterDataTransformer

    def fetch_raw_data(self, season: int) -> pd.DataFrame:
        """Fetch raw roster data for the provided season."""
        self.logger.info(f"Fetching roster data for season {season}")
        return fetch_seasonal_roster_data([season])

    def load_data(
        self,
        season: int,
        dry_run: bool = False,
        clear_table: bool = False,
        include_records: bool = False,
    ) -> Dict[str, Any]:
        """Complete workflow to load roster data for a given season."""
        operation_name = f"Rosters data load ({season})"
        self.logger.info(f"Starting {operation_name}")

        try:
            raw_data = self.fetch_raw_data(season=season)
            if raw_data.empty:
                self.logger.warning("No roster data returned from source")
                return {"success": False, "message": "No roster data found"}

            team_mapping = self._fetch_team_mapping()
            version = self._get_next_version()
            transformer = self.transformer_class(team_mapping, version)  # type: ignore[call-arg]
            transformed_records = transformer.transform(raw_data)
            skipped_records = getattr(transformer, "skipped_records", [])

            if not transformed_records:
                self.logger.error("No valid roster records after transformation")
                return {
                    "success": False,
                    "message": "No valid roster records after transformation",
                    "skipped": skipped_records,
                }

            if dry_run:
                dry_run_result = self._handle_dry_run(
                    transformed_records,
                    clear_table,
                    include_records=include_records,
                )
                dry_run_result["version"] = version
                if skipped_records:
                    dry_run_result["skipped"] = skipped_records
                return dry_run_result

            if clear_table:
                self.logger.info("Clearing Rosters table before load")
                cleared = self.db_manager.clear_table()
                if not cleared:
                    return {"success": False, "message": "Failed to clear rosters table"}

            upsert_result = self.db_manager.upsert_records(transformed_records, on_conflict="slug")
            if not upsert_result.get("success"):
                error = upsert_result.get("error", "Unknown upsert error")
                self.logger.error(f"Failed to upsert roster records: {error}")
                return {"success": False, "error": error}

            result = {
                "success": True,
                "total_fetched": len(raw_data),
                "total_validated": len(transformed_records),
                "upsert_result": upsert_result,
                "cleared_table": clear_table,
                "version": version,
            }
            if skipped_records:
                result["skipped"] = skipped_records

            self.logger.info(
                "Rosters data load completed successfully", extra={"version": version, "records": len(transformed_records)}
            )
            return result

        except Exception as exc:
            self.logger.error(f"Error during {operation_name}: {exc}")
            return {"success": False, "error": str(exc)}

    # ----- Internal helpers -----

    def _fetch_team_mapping(self) -> Dict[str, int]:
        supabase = self.db_manager.supabase
        candidates = ("Teams", "teams")
        last_error: Optional[Exception] = None

        for table in candidates:
            try:
                response = supabase.table(table).select("id,teamId").execute()
                data = getattr(response, "data", None)
                if not data:
                    continue
                mapping: Dict[str, int] = {}
                for row in data:
                    team_key = row.get("teamId")
                    team_id = row.get("id")
                    if team_key is None or team_id is None:
                        continue
                    mapping[str(team_key).upper()] = int(team_id)
                if mapping:
                    self.logger.debug(f"Loaded {len(mapping)} team identifiers from '{table}'")
                    return mapping
            except Exception as exc:
                last_error = exc
                self.logger.debug(f"Attempt to load team mapping from '{table}' failed: {exc}")

        error_message = "Could not load team mapping from Supabase"
        if last_error:
            error_message = f"{error_message}: {last_error}"
        raise RuntimeError(error_message)

    def _get_next_version(self) -> int:
        supabase = self.db_manager.supabase
        try:
            response = (
                supabase.table(self.table_name)
                .select("version")
                .order("version", desc=True)
                .limit(1)
                .execute()
            )
            rows = getattr(response, "data", None)
            if rows:
                max_version = rows[0].get("version")
                if max_version is not None:
                    return int(max_version) + 1
            return 1
        except Exception as exc:
            self.logger.error(f"Failed to determine next roster version: {exc}")
            raise RuntimeError("Could not determine current roster version") from exc

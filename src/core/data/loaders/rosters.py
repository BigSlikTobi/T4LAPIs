"""Rosters data loader for populating the Supabase rosters table."""

from __future__ import annotations

from typing import Type

import pandas as pd

from .base import BaseDataLoader
from ..fetch import fetch_seasonal_roster_data
from ..transform import BaseDataTransformer, RosterDataTransformer


class RostersDataLoader(BaseDataLoader):
    """Load seasonal roster data into the Supabase rosters table."""

    def __init__(self, table_name: str = "rosters") -> None:
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

            transformer = self.transformer_class()  # type: ignore[call-arg]
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
                if skipped_records:
                    dry_run_result["skipped"] = skipped_records
                return dry_run_result

            if clear_table:
                self.logger.info("Clearing rosters table before load")
                cleared = self.db_manager.clear_table()
                if not cleared:
                    return {"success": False, "message": "Failed to clear rosters table"}

            upsert_result = self.db_manager.upsert_records(transformed_records, on_conflict="team,player")
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
            }
            if skipped_records:
                result["skipped"] = skipped_records

            self.logger.info(
                "Rosters data load completed successfully",
                extra={"records": len(transformed_records)},
            )
            return result

        except Exception as exc:
            self.logger.error(f"Error during {operation_name}: {exc}")
            return {"success": False, "error": str(exc)}

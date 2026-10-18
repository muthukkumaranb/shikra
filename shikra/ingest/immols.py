"""Adapter for IMMOLS inventory stock ingestion."""

from __future__ import annotations

import csv
import sqlite3
from datetime import date
from pathlib import Path

from shikra.ingest.emms import AdapterResult


class IMMOLSAdapter:
    """Ingests IMMOLS stock extracts with data-quality validation."""

    SOURCE_NAME = "immols_stock.csv"

    def __init__(self, conn: sqlite3.Connection, reference_date: date | None = None) -> None:
        self.conn = conn
        self.reference_date = reference_date

    def ingest_file(self, path: Path | str) -> AdapterResult:
        """Ingest CSV file into lru_stock or quarantine table."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            lines = [line for line in f if not line.startswith("#")]

        reader = csv.DictReader(lines)
        return self.ingest_rows(list(reader))

    def ingest_rows(self, rows: list[dict[str, str]]) -> AdapterResult:
        """Process list of row dictionaries."""
        loaded = 0
        quarantined = 0
        missing_key_cnt = 0
        stale_cnt = 0
        join_coverage_cnt = 0

        # Determine reference date for stale detection (max last_updated date if not explicitly set)
        ref_date = self.reference_date
        if ref_date is None:
            dates = []
            for r in rows:
                dt_str = (r.get("last_updated") or "").strip()
                if dt_str:
                    try:
                        dates.append(date.fromisoformat(dt_str))
                    except ValueError:
                        pass
            ref_date = max(dates) if dates else date.today()

        valid_rows: list[tuple[str, str, int, int, float, str]] = []
        quarantine_rows: list[tuple[str, int, str]] = []

        for row_id, r in enumerate(rows):
            lru_id = (r.get("lru_id") or "").strip()
            location = (r.get("location") or "").strip()
            on_hand_str = (r.get("on_hand") or "0").strip()
            on_order_str = (r.get("on_order") or "0").strip()
            lead_time_str = (r.get("lead_time_days") or "0").strip()
            last_updated_str = (r.get("last_updated") or "").strip()

            # Check missing primary keys
            if not lru_id:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: lru_id is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not location:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: location is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not last_updated_str:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: last_updated is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue

            # Check stale record (> 30 days old)
            try:
                row_date = date.fromisoformat(last_updated_str)
            except ValueError:
                quarantine_rows.append((self.SOURCE_NAME, row_id, f"missing_key: invalid date {last_updated_str}"))
                missing_key_cnt += 1
                quarantined += 1
                continue

            if (ref_date - row_date).days > 30:
                quarantine_rows.append(
                    (self.SOURCE_NAME, row_id, f"stale: last_updated {last_updated_str} is >30 days old")
                )
                stale_cnt += 1
                quarantined += 1
                continue

            # Valid row
            on_hand = int(on_hand_str)
            on_order = int(on_order_str)
            lead_time = float(lead_time_str)
            valid_rows.append((lru_id, location, on_hand, on_order, lead_time, last_updated_str))
            loaded += 1
            join_coverage_cnt += 1

        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO lru_stock (lru_id, location, on_hand, on_order, lead_time_days, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
            valid_rows,
        )
        cur.executemany(
            "INSERT INTO quarantine (source, row_id, reason) VALUES (?, ?, ?)",
            quarantine_rows,
        )
        self.conn.commit()

        return AdapterResult(
            source=self.SOURCE_NAME,
            rows=len(rows),
            loaded=loaded,
            quarantined=quarantined,
            missing_key_count=missing_key_cnt,
            mismatched_sn_count=0,
            stale_count=stale_cnt,
            join_coverage_count=join_coverage_cnt,
        )

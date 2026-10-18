"""Adapter for BRD repair orders ingestion."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from shikra.ingest.emms import AdapterResult


class BRDAdapter:
    """Ingests Base Repair Depot (BRD) repair orders with data-quality validation."""

    SOURCE_NAME = "brd_repairs.csv"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ingest_file(self, path: Path | str) -> AdapterResult:
        """Ingest CSV file into repair_order or quarantine table."""
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
        join_coverage_cnt = 0

        valid_rows: list[tuple[str, str, str, str, str, str]] = []
        quarantine_rows: list[tuple[str, int, str]] = []

        for row_id, r in enumerate(rows):
            ro_id = (r.get("ro_id") or "").strip()
            item_type = (r.get("item_type") or "").strip()
            sn = (r.get("sn") or "").strip()
            sent = (r.get("sent") or "").strip()
            returned = (r.get("returned") or "").strip()
            agency = (r.get("agency") or "").strip()

            # Check missing primary keys
            if not ro_id:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: ro_id is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not sn:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: sn is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not item_type:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: item_type is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue

            # Valid row
            valid_rows.append((ro_id, item_type, sn, sent, returned, agency))
            loaded += 1
            join_coverage_cnt += 1

        cur = self.conn.cursor()
        cur.executemany(
            """INSERT OR REPLACE INTO repair_order
               (ro_id, item_type, sn, sent, returned, agency)
               VALUES (?, ?, ?, ?, ?, ?)""",
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
            stale_count=0,
            join_coverage_count=join_coverage_cnt,
        )

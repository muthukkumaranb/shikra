"""Adapter for e-MMS work orders ingestion."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shikra.ingest.schema import get_fleet_mappings


@dataclass
class AdapterResult:
    source: str
    rows: int
    loaded: int
    quarantined: int
    missing_key_count: int = 0
    mismatched_sn_count: int = 0
    stale_count: int = 0
    join_coverage_count: int = 0


class EMMSAdapter:
    """Ingests e-MMS work orders with data-quality validation."""

    SOURCE_NAME = "emms_workorders.csv"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.valid_tails, self.engine_to_tail = get_fleet_mappings(conn)

    def ingest_file(self, path: Path | str) -> AdapterResult:
        """Ingest CSV file into work_order or quarantine table."""
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
        mismatched_sn_cnt = 0
        join_coverage_cnt = 0

        valid_rows: list[tuple[str, str, str, str, str, str]] = []
        quarantine_rows: list[tuple[str, int, str]] = []

        for row_id, r in enumerate(rows):
            wo_id = (r.get("wo_id") or "").strip()
            tail_id = (r.get("tail_id") or "").strip()
            dt = (r.get("date") or "").strip()
            system_text = (r.get("system_text") or "").strip()
            action = (r.get("action") or "").strip()
            engine_sn = (r.get("engine_sn") or "").strip()

            # Check missing primary keys
            if not wo_id:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: wo_id is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not tail_id:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: tail_id is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not dt:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: date is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue

            # Check tail exists
            if self.valid_tails and tail_id not in self.valid_tails:
                quarantine_rows.append(
                    (self.SOURCE_NAME, row_id, f"missing_key: tail {tail_id} not in fleet")
                )
                missing_key_cnt += 1
                quarantined += 1
                continue

            # Check engine serial mismatch if engine_sn is populated
            if engine_sn:
                if self.engine_to_tail and engine_sn not in self.engine_to_tail:
                    quarantine_rows.append(
                        (self.SOURCE_NAME, row_id, f"mismatched_sn: engine {engine_sn} not in fleet")
                    )
                    mismatched_sn_cnt += 1
                    quarantined += 1
                    continue
                if self.engine_to_tail and self.engine_to_tail[engine_sn] != tail_id:
                    actual_tail = self.engine_to_tail[engine_sn]
                    quarantine_rows.append((
                        self.SOURCE_NAME,
                        row_id,
                        f"mismatched_sn: engine {engine_sn} belongs to {actual_tail}, not {tail_id}",
                    ))
                    mismatched_sn_cnt += 1
                    quarantined += 1
                    continue

            # Valid row
            valid_rows.append((wo_id, tail_id, dt, system_text, action, engine_sn))
            loaded += 1
            join_coverage_cnt += 1

        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO work_order (wo_id, tail_id, date, system_text, action, engine_sn) VALUES (?, ?, ?, ?, ?, ?)",
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
            mismatched_sn_count=mismatched_sn_cnt,
            stale_count=0,
            join_coverage_count=join_coverage_cnt,
        )

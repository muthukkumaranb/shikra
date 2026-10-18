"""Adapter for HUMS engine telemetry ingestion."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from shikra.ingest.emms import AdapterResult
from shikra.ingest.schema import get_fleet_mappings


class HUMSAdapter:
    """Ingests HUMS sensor telemetry with data-quality validation."""

    SOURCE_NAME = "hums_engine.csv"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.valid_tails, self.engine_to_tail = get_fleet_mappings(conn)

    def ingest_file(self, path: Path | str) -> AdapterResult:
        """Ingest CSV file into hums_reading or quarantine table."""
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

        valid_rows: list[tuple] = []
        quarantine_rows: list[tuple[str, int, str]] = []

        for row_id, r in enumerate(rows):
            engine_sn = (r.get("engine_sn") or "").strip()
            tail_id = (r.get("tail_id") or "").strip()
            dt = (r.get("date") or "").strip()
            cycle_str = (r.get("cycle") or "").strip()

            # Check missing primary keys
            if not engine_sn:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: engine_sn is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not tail_id:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: tail_id is blank"))
                missing_key_cnt += 1
                quarantined += 1
                continue
            if not cycle_str:
                quarantine_rows.append((self.SOURCE_NAME, row_id, "missing_key: cycle is blank"))
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

            # Check engine serial mismatch
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
            cycle = int(cycle_str)
            s2 = float(r.get("s2", 0.0) or 0.0)
            s3 = float(r.get("s3", 0.0) or 0.0)
            s4 = float(r.get("s4", 0.0) or 0.0)
            s7 = float(r.get("s7", 0.0) or 0.0)
            s11 = float(r.get("s11", 0.0) or 0.0)
            s12 = float(r.get("s12", 0.0) or 0.0)
            s15 = float(r.get("s15", 0.0) or 0.0)

            valid_rows.append((engine_sn, tail_id, dt, cycle, s2, s3, s4, s7, s11, s12, s15))
            loaded += 1
            join_coverage_cnt += 1

        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO hums_reading
               (engine_sn, tail_id, date, cycle, s2, s3, s4, s7, s11, s12, s15)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

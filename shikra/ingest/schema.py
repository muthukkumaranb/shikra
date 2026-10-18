"""Database schema and fleet master data loader for unified ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import yaml

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tail (
    tail_id TEXT PRIMARY KEY,
    squadron TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS engine (
    engine_sn TEXT PRIMARY KEY,
    tail_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY(tail_id) REFERENCES tail(tail_id)
);

CREATE TABLE IF NOT EXISTS lru_stock (
    lru_id TEXT NOT NULL,
    location TEXT NOT NULL,
    on_hand INTEGER NOT NULL,
    on_order INTEGER NOT NULL,
    lead_time_days REAL NOT NULL,
    last_updated TEXT NOT NULL,
    PRIMARY KEY(lru_id, location)
);

CREATE TABLE IF NOT EXISTS work_order (
    wo_id TEXT PRIMARY KEY,
    tail_id TEXT NOT NULL,
    date TEXT NOT NULL,
    system_text TEXT NOT NULL,
    action TEXT NOT NULL,
    engine_sn TEXT,
    FOREIGN KEY(tail_id) REFERENCES tail(tail_id)
);

CREATE TABLE IF NOT EXISTS repair_order (
    ro_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    sn TEXT NOT NULL,
    sent TEXT NOT NULL,
    returned TEXT NOT NULL,
    agency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hums_reading (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    engine_sn TEXT NOT NULL,
    tail_id TEXT NOT NULL,
    date TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    s2 REAL,
    s3 REAL,
    s4 REAL,
    s7 REAL,
    s11 REAL,
    s12 REAL,
    s15 REAL,
    FOREIGN KEY(engine_sn) REFERENCES engine(engine_sn),
    FOREIGN KEY(tail_id) REFERENCES tail(tail_id)
);

CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    reason TEXT NOT NULL
);
"""


def init_db(conn: sqlite3.Connection, config_path: Path | None = None) -> None:
    """Initialize SQLite database tables and seed fleet master tables."""
    conn.executescript(SCHEMA_SQL)
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "wing.yaml"

    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seed_fleet(conn, cfg)


def seed_fleet(conn: sqlite3.Connection, cfg: dict[str, Any]) -> None:
    """Seed master tail and engine tables from wing configuration."""
    n_sqn = int(cfg["fleet"]["squadrons"])
    ac_per_sqn = int(cfg["fleet"]["aircraft_per_sqn"])
    engines_per_ac = int(cfg["fleet"]["engines_per_ac"])
    n_ac = n_sqn * ac_per_sqn

    tails: list[tuple[str, str]] = []
    engines: list[tuple[str, str, int]] = []

    for idx in range(n_ac):
        tid = f"T{idx + 1:02d}"
        sqn_idx = (idx // ac_per_sqn) + 1
        sqn = f"S{sqn_idx}"
        tails.append((tid, sqn))

        for e in range(engines_per_ac):
            pos = idx * engines_per_ac + e
            esn = f"ENG-{pos + 1001:04d}"
            engines.append((esn, tid, e))

    conn.executemany("INSERT OR REPLACE INTO tail (tail_id, squadron) VALUES (?, ?)", tails)
    conn.executemany(
        "INSERT OR REPLACE INTO engine (engine_sn, tail_id, position) VALUES (?, ?, ?)",
        engines,
    )
    conn.commit()


def get_fleet_mappings(conn: sqlite3.Connection) -> tuple[set[str], dict[str, str]]:
    """Return set of valid tail_ids and dict of engine_sn -> tail_id."""
    cur = conn.cursor()
    cur.execute("SELECT tail_id FROM tail")
    valid_tails = {row[0] for row in cur.fetchall()}
    cur.execute("SELECT engine_sn, tail_id FROM engine")
    engine_to_tail = {row[0]: row[1] for row in cur.fetchall()}
    return valid_tails, engine_to_tail

"""Append-only SQLite audit log for SHIKRA sign-off actions.

Writes to app_data/audit.db (gitignored). The schema is fixed and
append-only: no update, no delete.

Tamper evidence: every row carries prev_hash and row_hash, where
row_hash = sha256(prev_hash + canonical(row)). Editing, deleting or
reordering rows breaks the chain.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

APP_DATA_DIR = Path(__file__).parent.parent.parent / "app_data"
AUDIT_DB_PATH = APP_DATA_DIR / "audit.db"

GENESIS_HASH = "0" * 64
_FIELDS = ("ts_utc", "role", "action", "target", "recommendation_id", "decision", "note")


def canonical(row: dict) -> str:
    """Deterministic JSON encoding of a row's content fields."""
    return json.dumps(
        {k: row[k] for k in _FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _row_hash(prev_hash: str, row: dict) -> str:
    return hashlib.sha256((prev_hash + canonical(row)).encode()).hexdigest()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables; add hash columns to an old DB and backfill the chain once."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL,
            recommendation_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            note TEXT DEFAULT '',
            prev_hash TEXT,
            row_hash TEXT
        )
    """)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(audit_log)")}
    if {"prev_hash", "row_hash"} <= cols:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-check under the write lock in case another process migrated first.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(audit_log)")}
        if not {"prev_hash", "row_hash"} <= cols:
            conn.execute("ALTER TABLE audit_log ADD COLUMN prev_hash TEXT")
            conn.execute("ALTER TABLE audit_log ADD COLUMN row_hash TEXT")
            prev = GENESIS_HASH
            rows = conn.execute(
                f"SELECT id, {', '.join(_FIELDS)} FROM audit_log ORDER BY id"
            ).fetchall()
            for r in rows:
                row = dict(zip(_FIELDS, r[1:]))
                h = _row_hash(prev, row)
                conn.execute(
                    "UPDATE audit_log SET prev_hash = ?, row_hash = ? WHERE id = ?",
                    (prev, h, r[0]),
                )
                prev = h
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the audit database, creating tables if needed."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_DB_PATH), isolation_level=None)
    _ensure_schema(conn)
    return conn


def append(
    role: str,
    action: str,
    target: str,
    recommendation_id: str,
    decision: str,
    note: str = "",
) -> None:
    """Append a row to the audit log. No update, no delete."""
    conn = _get_connection()
    row = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "role": role,
        "action": action,
        "target": target,
        "recommendation_id": recommendation_id,
        "decision": decision,
        "note": note,
    }
    try:
        # One write transaction: read the tip and insert atomically so
        # concurrent appends cannot fork the chain.
        conn.execute("BEGIN IMMEDIATE")
        try:
            tip = conn.execute(
                "SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev = tip[0] if tip else GENESIS_HASH
            conn.execute(
                """
                INSERT INTO audit_log (ts_utc, role, action, target, recommendation_id,
                                       decision, note, prev_hash, row_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*(row[k] for k in _FIELDS), prev, _row_hash(prev, row)),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def get_recent(n: int = 20) -> list[dict]:
    """Return the last n audit log entries, newest first."""
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT id, ts_utc, role, action, target, recommendation_id, decision, note "
        "FROM audit_log ORDER BY id DESC LIMIT ?",
        (n,),
    )
    columns = ["id", "ts_utc", "role", "action", "target", "recommendation_id", "decision", "note"]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return rows


def count() -> int:
    """Return the total number of audit log entries."""
    conn = _get_connection()
    cursor = conn.execute("SELECT COUNT(*) FROM audit_log")
    result = cursor.fetchone()[0]
    conn.close()
    return result

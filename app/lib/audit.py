"""Append-only SQLite audit log for SHIKRA sign-off actions.

Writes to app_data/audit.db (gitignored). The schema is fixed and
append-only: no update, no delete.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

APP_DATA_DIR = Path(__file__).parent.parent.parent / "app_data"
AUDIT_DB_PATH = APP_DATA_DIR / "audit.db"


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the audit database, creating tables if needed."""
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL,
            recommendation_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            note TEXT DEFAULT ''
        )
    """)
    conn.commit()
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
    conn.execute(
        """
        INSERT INTO audit_log (ts_utc, role, action, target, recommendation_id, decision, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            role,
            action,
            target,
            recommendation_id,
            decision,
            note,
        ),
    )
    conn.commit()
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

"""Append-only SQLite audit log for SHIKRA sign-off actions.

Writes to app_data/audit.db (gitignored). The schema is fixed and
append-only: no update, no delete.

Tamper evidence: every row carries prev_hash and row_hash, where
row_hash = sha256(prev_hash + canonical(row)). Editing, deleting or
reordering rows breaks the chain. Each UTC day is also sealed with a
Merkle root over its row hashes, authenticated by an HMAC keyed from
app_data/audit.key, so truncating the tail of the log is detectable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
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


def merkle_root(hashes: list[str]) -> str:
    """Merkle root over hex row hashes; odd levels duplicate the last node."""
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


def _load_key() -> bytes:
    """Read the HMAC key, creating it (mode 0o600) on first use."""
    path = Path(APP_DATA_DIR) / "audit.key"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as f:
        f.write(secrets.token_bytes(32))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path.read_bytes()


def _root_hmac(key: bytes, date_utc: str, n_rows: int, root: str) -> str:
    return hmac.new(key, f"{date_utc}|{n_rows}|{root}".encode(), hashlib.sha256).hexdigest()


def _today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def _day_hashes(conn: sqlite3.Connection, date_utc: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT row_hash FROM audit_log WHERE substr(ts_utc, 1, 10) = ? ORDER BY id",
            (date_utc,),
        )
    ]


def _seal(conn: sqlite3.Connection, date_utc: str, replace: bool) -> None:
    """Write the day's root inside the caller's transaction."""
    hashes = _day_hashes(conn, date_utc)
    root = merkle_root(hashes)
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    conn.execute(
        f"{verb} INTO daily_roots (date_utc, n_rows, merkle_root, hmac) VALUES (?, ?, ?, ?)",
        (date_utc, len(hashes), root, _root_hmac(_load_key(), date_utc, len(hashes), root)),
    )


def seal_day(date_utc: str | None = None) -> None:
    """Seal (or re-seal) the current UTC day's Merkle root. Past days are final."""
    today = _today_utc()
    date_utc = date_utc or today
    if date_utc != today:
        raise ValueError(f"refusing to reseal past day {date_utc}; only {today} may be sealed")
    conn = _get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            _seal(conn, date_utc, replace=True)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables; add hash columns to an old DB and backfill the chain once."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_roots (
            date_utc TEXT PRIMARY KEY,
            n_rows INT,
            merkle_root TEXT,
            hmac TEXT
        )
    """)
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
            # First-time seal of every legacy day (not a reseal: the table is new).
            today = _today_utc()
            days = conn.execute(
                "SELECT DISTINCT substr(ts_utc, 1, 10) FROM audit_log ORDER BY 1"
            ).fetchall()
            for (day,) in days:
                _seal(conn, day, replace=(day == today))
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
            # seal_day() for the row's own (current) UTC day, in the same transaction.
            _seal(conn, row["ts_utc"][:10], replace=True)
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


def verify() -> dict:
    """Recompute the hash chain and every daily root; report the first break.

    Returns {ok, n_rows, n_days, first_bad_id, reason, latest_date, latest_root}.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            f"SELECT id, {', '.join(_FIELDS)}, prev_hash, row_hash FROM audit_log ORDER BY id"
        ).fetchall()
        roots = conn.execute(
            "SELECT date_utc, n_rows, merkle_root, hmac FROM daily_roots ORDER BY date_utc"
        ).fetchall()
    finally:
        conn.close()

    result = {
        "ok": False,
        "n_rows": len(rows),
        "n_days": len(roots),
        "first_bad_id": None,
        "reason": "",
        "latest_date": roots[-1][0] if roots else None,
        "latest_root": roots[-1][2] if roots else None,
    }

    def broken(row_id: int | None, reason: str) -> dict:
        result.update(first_bad_id=row_id, reason=reason)
        return result

    prev = GENESIS_HASH
    by_day: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        row_id, row, stored_prev, stored_hash = r[0], dict(zip(_FIELDS, r[1:8])), r[8], r[9]
        if stored_prev != prev:
            return broken(row_id, "prev_hash does not link to previous row (deleted or reordered)")
        if _row_hash(prev, row) != stored_hash:
            return broken(row_id, "row_hash mismatch (row content edited)")
        prev = stored_hash
        by_day.setdefault(str(row["ts_utc"])[:10], []).append((row_id, stored_hash))

    key = _load_key()
    unsealed = sorted(set(by_day) - {d for d, *_ in roots})
    if unsealed:
        day = unsealed[0]
        return broken(by_day[day][0][0], f"day {day} has rows but no sealed root")
    for date_utc, n_rows, root, mac in roots:
        day_rows = by_day.get(date_utc, [])
        first_id = day_rows[0][0] if day_rows else None
        expected_mac = _root_hmac(key, date_utc, n_rows, root)
        if not hmac.compare_digest(expected_mac, mac or ""):
            return broken(first_id, f"HMAC mismatch on daily root {date_utc}")
        if n_rows != len(day_rows):
            return broken(
                first_id, f"day {date_utc} sealed {n_rows} rows but log has {len(day_rows)}"
            )
        if merkle_root([h for _, h in day_rows]) != root:
            return broken(first_id, f"Merkle root mismatch for day {date_utc}")

    result.update(ok=True, reason="chain intact")
    return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m app.lib.audit")
    parser.add_argument("command", choices=["verify"])
    parser.parse_args()
    res = verify()
    if res["ok"]:
        latest = (res["latest_root"] or "none")[:12]
        print(f"chain intact: {res['n_rows']} rows, {res['n_days']} days, latest root {latest}")
        sys.exit(0)
    print(f"BROKEN at row id={res['first_bad_id']}: {res['reason']}")
    sys.exit(1)

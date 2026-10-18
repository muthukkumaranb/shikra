"""Tamper-evidence tests for the audit log hash chain and daily roots."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

import app.lib.audit as audit


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    test_db = tmp_path / "test_audit.db"
    monkeypatch.setattr(audit, "APP_DATA_DIR", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", test_db)
    return test_db


def _append_n(n: int) -> None:
    for i in range(1, n + 1):
        audit.append(
            role="Engineering Officer",
            action="engine_removal",
            target=f"T0{i}/eng0",
            recommendation_id=f"REC-{i:03d}",
            decision="approved",
            note=f"row {i}",
        )


def _raw(db: Path, sql: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(sql)
    conn.commit()
    conn.close()


def test_five_appends_verify_ok(db: Path):
    _append_n(5)
    res = audit.verify()
    assert res["ok"], res
    assert res["n_rows"] == 5
    assert res["n_days"] >= 1
    assert res["first_bad_id"] is None
    assert (db.parent / "audit.key").stat().st_size == 32


def test_edit_detected_at_row(db: Path):
    _append_n(5)
    _raw(db, "UPDATE audit_log SET decision = 'Defer' WHERE id = 3")
    res = audit.verify()
    assert not res["ok"]
    assert res["first_bad_id"] == 3


def test_delete_detected(db: Path):
    _append_n(5)
    _raw(db, "DELETE FROM audit_log WHERE id = 2")
    res = audit.verify()
    assert not res["ok"]
    assert res["first_bad_id"] == 3


def test_tail_truncation_detected(db: Path):
    _append_n(5)
    _raw(db, "DELETE FROM audit_log WHERE id = 5")
    res = audit.verify()
    assert not res["ok"]


def test_root_tamper_hmac_mismatch(db: Path):
    _append_n(5)
    _raw(db, "UPDATE daily_roots SET merkle_root = '" + "ab" * 32 + "'")
    res = audit.verify()
    assert not res["ok"]
    assert "HMAC mismatch" in res["reason"]


def test_merkle_root_known_leaves():
    leaves = ["00" * 32, "11" * 32, "22" * 32]
    # Hand-computed: sha256(sha256(L0 || L1) || sha256(L2 || L2)), last leaf duplicated.
    expected = "f59242779d0bc9cfb1f3decf078f0dd2e0a5797bad13a4451ed3a96ad0e4182f"
    assert audit.merkle_root(leaves) == expected
    assert audit.merkle_root([]) == hashlib.sha256(b"").hexdigest()


def test_reseal_past_day_refused(db: Path):
    with pytest.raises(ValueError):
        audit.seal_day("2000-01-01")


def test_old_schema_migration(db: Path):
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE audit_log (
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
    conn.executemany(
        "INSERT INTO audit_log (ts_utc, role, action, target, recommendation_id, decision, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("2026-01-01T10:00:00+00:00", "Commander", "spares_dispatch", "S1", "R-1", "approved", ""),
            ("2026-01-02T10:00:00+00:00", "Commander", "spares_dispatch", "S2", "R-2", "deferred", "x"),
        ],
    )
    conn.commit()
    conn.close()

    _append_n(1)
    res = audit.verify()
    assert res["ok"], res
    assert res["n_rows"] == 3


def test_no_mutating_attributes():
    for name in ("update", "delete", "edit", "remove"):
        assert not hasattr(audit, name)

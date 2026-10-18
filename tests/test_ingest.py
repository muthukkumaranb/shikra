"""Tests for data ingestion and quality checks (contract §7)."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from shikra.ingest import (
    BRDAdapter,
    EMMSAdapter,
    HUMSAdapter,
    IMMOLSAdapter,
    init_db,
    load_all,
)


def _create_mem_db() -> sqlite3.Connection:
    """Create in-memory SQLite database initialized with fleet schema."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


def test_emms_roundtrip() -> None:
    """EMMSAdapter loads a valid work order and round-trips correctly."""
    conn = _create_mem_db()
    adapter = EMMSAdapter(conn)

    row = {
        "wo_id": "WO-00001",
        "tail_id": "T01",
        "date": "2026-01-01",
        "system_text": "No.1 eng vibration trend rising, investigate",
        "action": "planned removal",
        "engine_sn": "ENG-1001",
    }
    res = adapter.ingest_rows([row])
    assert res.loaded == 1
    assert res.quarantined == 0

    cur = conn.cursor()
    cur.execute("SELECT wo_id, tail_id, date, system_text, action, engine_sn FROM work_order")
    fetched = cur.fetchone()
    assert fetched == (
        "WO-00001",
        "T01",
        "2026-01-01",
        "No.1 eng vibration trend rising, investigate",
        "planned removal",
        "ENG-1001",
    )


def test_immols_roundtrip() -> None:
    """IMMOLSAdapter loads a valid stock row and round-trips correctly."""
    conn = _create_mem_db()
    adapter = IMMOLSAdapter(conn, reference_date=date(2027, 1, 1))

    row = {
        "lru_id": "fuel_pump",
        "location": "base_S1",
        "on_hand": "4",
        "on_order": "1",
        "lead_time_days": "15.0",
        "last_updated": "2026-12-25",
    }
    res = adapter.ingest_rows([row])
    assert res.loaded == 1
    assert res.quarantined == 0

    cur = conn.cursor()
    cur.execute("SELECT lru_id, location, on_hand, on_order, lead_time_days, last_updated FROM lru_stock")
    fetched = cur.fetchone()
    assert fetched == ("fuel_pump", "base_S1", 4, 1, 15.0, "2026-12-25")


def test_hums_roundtrip() -> None:
    """HUMSAdapter loads a valid telemetry row and round-trips correctly."""
    conn = _create_mem_db()
    adapter = HUMSAdapter(conn)

    row = {
        "engine_sn": "ENG-1001",
        "tail_id": "T01",
        "date": "2026-01-05",
        "cycle": "10",
        "s2": "642.5",
        "s3": "1580.1",
        "s4": "1400.2",
        "s7": "553.8",
        "s11": "47.2",
        "s12": "521.9",
        "s15": "8.4",
    }
    res = adapter.ingest_rows([row])
    assert res.loaded == 1
    assert res.quarantined == 0

    cur = conn.cursor()
    cur.execute("SELECT engine_sn, tail_id, date, cycle, s2, s3, s4, s7, s11, s12, s15 FROM hums_reading")
    fetched = cur.fetchone()
    assert fetched == ("ENG-1001", "T01", "2026-01-05", 10, 642.5, 1580.1, 1400.2, 553.8, 47.2, 521.9, 8.4)


def test_brd_roundtrip() -> None:
    """BRDAdapter loads a valid repair order and round-trips correctly."""
    conn = _create_mem_db()
    adapter = BRDAdapter(conn)

    row = {
        "ro_id": "RO-00001",
        "item_type": "engine",
        "sn": "ENG-1001",
        "sent": "2026-02-01",
        "returned": "2026-04-01",
        "agency": "BRD-A",
    }
    res = adapter.ingest_rows([row])
    assert res.loaded == 1
    assert res.quarantined == 0

    cur = conn.cursor()
    cur.execute("SELECT ro_id, item_type, sn, sent, returned, agency FROM repair_order")
    fetched = cur.fetchone()
    assert fetched == ("RO-00001", "item_type" if False else "engine", "ENG-1001", "2026-02-01", "2026-04-01", "BRD-A")


def test_quarantine_catches_missing_key() -> None:
    """Quarantine table captures missing keys across adapters."""
    conn = _create_mem_db()
    cur = conn.cursor()

    # EMMS missing wo_id
    emms = EMMSAdapter(conn)
    emms.ingest_rows([{"wo_id": "", "tail_id": "T01", "date": "2026-01-01"}])
    cur.execute("SELECT source, reason FROM quarantine WHERE source = ?", (emms.SOURCE_NAME,))
    q_emms = cur.fetchone()
    assert q_emms[1].startswith("missing_key")

    # IMMOLS missing lru_id
    immols = IMMOLSAdapter(conn)
    immols.ingest_rows([{"lru_id": "", "location": "depot", "last_updated": "2026-01-01"}])
    cur.execute("SELECT source, reason FROM quarantine WHERE source = ?", (immols.SOURCE_NAME,))
    q_imm = cur.fetchone()
    assert q_imm[1].startswith("missing_key")

    # HUMS missing cycle
    hums = HUMSAdapter(conn)
    hums.ingest_rows([{"engine_sn": "ENG-1001", "tail_id": "T01", "cycle": ""}])
    cur.execute("SELECT source, reason FROM quarantine WHERE source = ?", (hums.SOURCE_NAME,))
    q_hums = cur.fetchone()
    assert q_hums[1].startswith("missing_key")

    # BRD missing ro_id
    brd = BRDAdapter(conn)
    brd.ingest_rows([{"ro_id": "", "item_type": "engine", "sn": "ENG-1001"}])
    cur.execute("SELECT source, reason FROM quarantine WHERE source = ?", (brd.SOURCE_NAME,))
    q_brd = cur.fetchone()
    assert q_brd[1].startswith("missing_key")


def test_quarantine_catches_mismatched_sn() -> None:
    """Quarantine catches engine serial numbers on the wrong tail."""
    conn = _create_mem_db()
    cur = conn.cursor()

    # ENG-1001 belongs to T01, not T02
    emms = EMMSAdapter(conn)
    res = emms.ingest_rows([{
        "wo_id": "WO-99999",
        "tail_id": "T02",
        "date": "2026-01-01",
        "engine_sn": "ENG-1001",
    }])
    assert res.quarantined == 1
    cur.execute("SELECT reason FROM quarantine WHERE source = ?", (emms.SOURCE_NAME,))
    reason = cur.fetchone()[0]
    assert reason.startswith("mismatched_sn")

    # In HUMS
    hums = HUMSAdapter(conn)
    res_hums = hums.ingest_rows([{
        "engine_sn": "ENG-1001",
        "tail_id": "T02",
        "date": "2026-01-01",
        "cycle": "1",
    }])
    assert res_hums.quarantined == 1
    cur.execute("SELECT reason FROM quarantine WHERE source = ?", (hums.SOURCE_NAME,))
    reason_hums = cur.fetchone()[0]
    assert reason_hums.startswith("mismatched_sn")


def test_quarantine_catches_stale_stock() -> None:
    """Quarantine catches stock records older than 30 days."""
    conn = _create_mem_db()
    cur = conn.cursor()
    ref_date = date(2027, 1, 1)

    immols = IMMOLSAdapter(conn, reference_date=ref_date)
    res = immols.ingest_rows([{
        "lru_id": "fuel_pump",
        "location": "depot",
        "on_hand": "5",
        "last_updated": "2026-11-01",  # 61 days before ref_date
    }])
    assert res.quarantined == 1
    cur.execute("SELECT reason FROM quarantine WHERE source = ?", (immols.SOURCE_NAME,))
    reason = cur.fetchone()[0]
    assert reason.startswith("stale")


def test_row_conservation_on_mock_exports(tmp_path: Path) -> None:
    """Verify loaded + quarantined = rows for every source on full mock exports."""
    mock_dir = Path("data/mock_exports")
    if not (mock_dir / "emms_workorders.csv").exists():
        return

    db_path = tmp_path / "test_unified.db"
    load_all(mock_dir, db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    sources = [
        ("emms_workorders.csv", "work_order"),
        ("immols_stock.csv", "lru_stock"),
        ("hums_engine.csv", "hums_reading"),
        ("brd_repairs.csv", "repair_order"),
    ]

    for src_file, tbl in sources:
        with (mock_dir / src_file).open("r", encoding="utf-8") as f:
            lines = [line for line in f if not line.startswith("#") and line.strip()]
        total_rows = len(lines) - 1  # subtract header

        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        loaded = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM quarantine WHERE source = ?", (src_file,))
        quarantined = cur.fetchone()[0]

        assert loaded + quarantined == total_rows, f"Row conservation violated for {src_file}"

    conn.close()


def test_deterministic_seed(tmp_path: Path) -> None:
    """Same random seed produces identical mock exports."""
    import pandas as pd

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    # Save original OUT_DIR
    import scripts.make_mock_exports as mme
    orig_out = mme.OUT_DIR

    try:
        mme.OUT_DIR = out1
        mme.generate(seed=2026)
        mme.OUT_DIR = out2
        mme.generate(seed=2026)

        for fname in ["emms_workorders.csv", "immols_stock.csv", "hums_engine.csv", "brd_repairs.csv", "faults_injected.csv"]:
            df1 = pd.read_csv(out1 / fname, comment="#")
            df2 = pd.read_csv(out2 / fname, comment="#")
            pd.testing.assert_frame_equal(df1, df2)
    finally:
        mme.OUT_DIR = orig_out

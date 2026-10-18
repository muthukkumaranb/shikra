"""Pipeline orchestration for loading mock exports into SQLite unified model."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from shikra.ingest.brd import BRDAdapter
from shikra.ingest.emms import EMMSAdapter
from shikra.ingest.hums import HUMSAdapter
from shikra.ingest.immols import IMMOLSAdapter
from shikra.ingest.schema import init_db


def load_all(
    export_dir: Path | str = "data/mock_exports",
    db_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> Path:
    """Load all mock exports into a unified SQLite database.

    Returns the Path to the generated sqlite3 database.
    """
    export_dir = Path(export_dir)
    if db_path is None:
        db_path = export_dir.parent / "unified.db"
    else:
        db_path = Path(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        init_db(conn, Path(config_path) if config_path else None)

        emms_path = export_dir / "emms_workorders.csv"
        if emms_path.exists():
            EMMSAdapter(conn).ingest_file(emms_path)

        immols_path = export_dir / "immols_stock.csv"
        if immols_path.exists():
            IMMOLSAdapter(conn).ingest_file(immols_path)

        hums_path = export_dir / "hums_engine.csv"
        if hums_path.exists():
            HUMSAdapter(conn).ingest_file(hums_path)

        brd_path = export_dir / "brd_repairs.csv"
        if brd_path.exists():
            BRDAdapter(conn).ingest_file(brd_path)

    finally:
        conn.close()

    return db_path


def generate_reports(
    db_path: Path | str,
    faults_csv: Path | str = "data/mock_exports/faults_injected.csv",
    out_dir: Path | str = "results",
) -> tuple[Path, Path]:
    """Generate dq_report.csv and dq_detection.csv from the SQLite database."""
    db_path = Path(db_path)
    faults_csv = Path(faults_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # ── 1. Data-quality report (dq_report.csv) ──────────────────────────
        # Sources in fixed order
        sources = [
            ("emms_workorders.csv", "work_order"),
            ("immols_stock.csv", "lru_stock"),
            ("hums_engine.csv", "hums_reading"),
            ("brd_repairs.csv", "repair_order"),
        ]

        report_rows: list[dict[str, Any]] = []
        for src_name, table_name in sources:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            loaded = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM quarantine WHERE source = ?", (src_name,))
            quarantined = int(cur.fetchone()[0])

            total_rows = loaded + quarantined

            cur.execute(
                "SELECT COUNT(*) FROM quarantine WHERE source = ? AND reason LIKE 'missing_key%'",
                (src_name,),
            )
            missing_keys = int(cur.fetchone()[0])

            cur.execute(
                "SELECT COUNT(*) FROM quarantine WHERE source = ? AND reason LIKE 'mismatched_sn%'",
                (src_name,),
            )
            mismatched_sns = int(cur.fetchone()[0])

            cur.execute(
                "SELECT COUNT(*) FROM quarantine WHERE source = ? AND reason LIKE 'stale%'",
                (src_name,),
            )
            stale = int(cur.fetchone()[0])

            load_rate = round(100.0 * loaded / total_rows, 2) if total_rows > 0 else 0.0
            mk_pct = round(100.0 * missing_keys / total_rows, 2) if total_rows > 0 else 0.0
            msn_pct = round(100.0 * mismatched_sns / total_rows, 2) if total_rows > 0 else 0.0
            stale_pct = round(100.0 * stale / total_rows, 2) if total_rows > 0 else 0.0

            if table_name == "work_order":
                cur.execute(
                    """SELECT COUNT(*) FROM work_order w
                       WHERE w.tail_id IN (SELECT tail_id FROM tail)
                         AND (w.engine_sn = '' OR EXISTS (
                             SELECT 1 FROM engine e WHERE e.engine_sn = w.engine_sn AND e.tail_id = w.tail_id
                         ))"""
                )
                resolved = int(cur.fetchone()[0])
                key_res = round(100.0 * resolved / loaded, 2) if loaded > 0 else 100.0
            elif table_name == "hums_reading":
                cur.execute(
                    """SELECT COUNT(*) FROM hums_reading h
                       WHERE h.tail_id IN (SELECT tail_id FROM tail)
                         AND EXISTS (
                             SELECT 1 FROM engine e WHERE e.engine_sn = h.engine_sn AND e.tail_id = h.tail_id
                         )"""
                )
                resolved = int(cur.fetchone()[0])
                key_res = round(100.0 * resolved / loaded, 2) if loaded > 0 else 100.0
            else:
                key_res = 100.0

            report_rows.append({
                "source": src_name,
                "rows": total_rows,
                "loaded": loaded,
                "quarantined": quarantined,
                "load_rate_pct": load_rate,
                "key_resolution_pct": key_res,
                "missing_key_pct": mk_pct,
                "mismatched_sn_pct": msn_pct,
                "stale_pct": stale_pct,
            })

        df_report = pd.DataFrame(report_rows)
        dq_report_path = out_dir / "dq_report.csv"
        df_report.to_csv(dq_report_path, index=False)

        # ── 2. Detection precision and recall (dq_detection.csv) ───────────
        # Load ground truth faults
        faults_df = pd.read_csv(faults_csv, comment="#")

        # Load quarantined items
        cur.execute("SELECT source, row_id, reason FROM quarantine")
        q_records = cur.fetchall()
        q_rows = []
        for src, rid, reason in q_records:
            ftype = reason.split(":")[0].strip()
            q_rows.append({"source": src, "row_id": int(rid), "detected_type": ftype})
        q_df = pd.DataFrame(q_rows) if q_rows else pd.DataFrame(columns=["source", "row_id", "detected_type"])

        detection_rows: list[dict[str, Any]] = []

        # Analyze by source and fault_type
        all_sources = sorted(list(set(sources_tuple[0] for sources_tuple in sources)))
        for src in all_sources:
            src_injected = faults_df[faults_df["source"] == src]
            src_detected = q_df[q_df["source"] == src]

            fault_types = sorted(list(set(src_injected["fault_type"].unique())))
            if not fault_types and not src_detected.empty:
                fault_types = sorted(list(set(src_detected["detected_type"].unique())))

            for ftype in fault_types:
                inj_set = set(src_injected[src_injected["fault_type"] == ftype]["row_id"])
                det_set = set(src_detected[src_detected["detected_type"] == ftype]["row_id"])

                tp = len(inj_set & det_set)
                fp = len(det_set - inj_set)
                fn = len(inj_set - det_set)

                prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
                rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0

                detection_rows.append({
                    "source": src,
                    "fault_type": ftype,
                    "injected": len(inj_set),
                    "detected": len(det_set),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": prec,
                    "recall": rec,
                })

        # Overall summary
        all_inj = set(zip(faults_df["source"], faults_df["row_id"]))
        all_det = set(zip(q_df["source"], q_df["row_id"])) if not q_df.empty else set()
        tp_all = len(all_inj & all_det)
        fp_all = len(all_det - all_inj)
        fn_all = len(all_inj - all_det)
        prec_all = round(tp_all / (tp_all + fp_all), 4) if (tp_all + fp_all) > 0 else 1.0
        rec_all = round(tp_all / (tp_all + fn_all), 4) if (tp_all + fn_all) > 0 else 1.0

        detection_rows.append({
            "source": "ALL",
            "fault_type": "ALL",
            "injected": len(all_inj),
            "detected": len(all_det),
            "tp": tp_all,
            "fp": fp_all,
            "fn": fn_all,
            "precision": prec_all,
            "recall": rec_all,
        })

        df_detection = pd.DataFrame(detection_rows)
        dq_detection_path = out_dir / "dq_detection.csv"
        df_detection.to_csv(dq_detection_path, index=False)

    finally:
        conn.close()

    return dq_report_path, dq_detection_path

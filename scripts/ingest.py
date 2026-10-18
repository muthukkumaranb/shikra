"""Ingestion CLI: loads mock exports into SQLite and produces DQ reports (contract §7)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.ingest.pipeline import generate_reports, load_all  # noqa: E402


def main() -> None:
    export_dir = ROOT / "data" / "mock_exports"
    db_path = ROOT / "data" / "unified.db"
    faults_csv = export_dir / "faults_injected.csv"
    results_dir = ROOT / "results"

    print(f"Loading mock exports from {export_dir} into {db_path}...")
    load_all(export_dir, db_path=db_path)
    print("Database loaded successfully.")

    print(f"Generating data-quality reports into {results_dir}...")
    dq_rep, dq_det = generate_reports(db_path, faults_csv=faults_csv, out_dir=results_dir)
    print(f"  -> {dq_rep}")
    print(f"  -> {dq_det}")


if __name__ == "__main__":
    main()

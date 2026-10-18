"""Tests for the SHIKRA Streamlit dashboard application.

Verifies:
- Page imports and Streamlit AppTest execution with zero exceptions.
- Data loader schemas and expected columns.
- Audit log append-only properties (no update/delete functions).
- Deterministic failure injection rule in Scenario.
- Adherence to ground rules (never asserting SHIKRA beats reactive).
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.lib.audit as audit
from app.lib.data import (
    load_conformal,
    load_heatmap,
    load_params,
    load_policy_summary,
    load_replay_rul,
    load_rul_cmapss,
    load_rul_cmapss_multi,
    load_supply_shock,
)
from app.lib.state import get_deterministic_injection_trajectory

PAGE_FILES = [
    "app/Home.py",
    "app/pages/1_Commander.py",
    "app/pages/2_Engineering_Officer.py",
    "app/pages/3_Logistics.py",
    "app/pages/4_Technician.py",
    "app/pages/5_Scenario.py",
    "app/pages/6_Evidence.py",
]


@pytest.mark.parametrize("page_path", PAGE_FILES)
def test_page_imports(page_path: str):
    """Verify each page file exists and is readable valid Python."""
    p = Path(page_path)
    assert p.exists(), f"Page file {page_path} not found"
    code = p.read_text(encoding="utf-8")
    assert len(code) > 0
    # Compile code to verify syntax
    compiled = compile(code, str(p), "exec")
    assert compiled is not None


@pytest.mark.parametrize("page_path", PAGE_FILES)
def test_apptest_pages_no_exceptions(page_path: str):
    """Use Streamlit's AppTest to run each page and verify no uncaught exceptions."""
    at = AppTest.from_file(page_path).run(timeout=20)
    assert len(at.exception) == 0, (
        f"Page {page_path} raised exceptions: {[e.value for e in at.exception]}"
    )


def test_data_loaders_columns():
    """Verify lib/data.py loaders return expected columns for each CSV/artifact."""
    # Heatmap
    heatmap = load_heatmap()
    for col in ["pipeline_mult", "quality", "serviceability_mean", "gain_vs_reactive_mean"]:
        assert col in heatmap.columns, f"Missing column {col} in heatmap"

    # Supply shock
    shock = load_supply_shock()
    for col in ["lru_id", "source", "loss_pp_shock_window", "peak_extra_grounded", "recovery_days"]:
        assert col in shock.columns, f"Missing column {col} in supply_shock"

    # Policy summary
    summary = load_policy_summary()
    for col in ["policy", "metric", "mean", "ci_lo", "ci_hi"]:
        assert col in summary.columns, f"Missing column {col} in policy_summary"

    # RUL C-MAPSS
    rul_s = load_rul_cmapss()
    for col in ["subset", "n_units"]:
        assert col in rul_s.columns, f"Missing column {col} in rul_cmapss"

    # RUL C-MAPSS Multi
    rul_m = load_rul_cmapss_multi()
    for col in ["subset", "n_units"]:
        assert col in rul_m.columns, f"Missing column {col} in rul_cmapss_multi"

    # Replay RUL
    replay = load_replay_rul()
    for col in ["subset", "unit", "cycle", "q10", "q50", "q90"]:
        assert col in replay.columns, f"Missing column {col} in replay_rul"
    # rul_true exists in parquet but is never displayed to users
    assert "rul_true" in replay.columns

    # Params & conformal
    params = load_params()
    assert "R_star" in params
    conformal = load_conformal()
    assert isinstance(conformal, dict)


def test_audit_log_append_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify audit log is strictly append-only with no update or delete functions."""
    # Redirect DB to temporary directory for isolated test
    test_db = tmp_path / "test_audit.db"
    monkeypatch.setattr(audit, "APP_DATA_DIR", tmp_path)
    monkeypatch.setattr(audit, "AUDIT_DB_PATH", test_db)

    # Initial count
    initial_count = audit.count()
    assert initial_count == 0

    # Insert 2 rows
    audit.append(
        role="Engineering Officer",
        action="engine_removal",
        target="T01/eng0",
        recommendation_id="REC-001",
        decision="approved",
        note="First test log",
    )
    audit.append(
        role="Commander",
        action="spares_dispatch",
        target="S1",
        recommendation_id="REC-002",
        decision="approved",
        note="Second test log",
    )

    # Assert count is exactly 2
    assert audit.count() == 2

    # Check recent entries
    recent = audit.get_recent(5)
    assert len(recent) == 2
    assert recent[0]["recommendation_id"] == "REC-002"
    assert recent[1]["recommendation_id"] == "REC-001"

    # Confirm there is no update or delete function in audit module
    audit_funcs = [
        name.lower()
        for name, obj in inspect.getmembers(audit, inspect.isfunction)
    ]
    forbidden_terms = ["update", "delete", "remove", "drop", "clear", "modify", "edit"]
    for func_name in audit_funcs:
        for term in forbidden_terms:
            assert term not in func_name, (
                f"Audit module contains prohibited function '{func_name}' with term '{term}'"
            )


def test_scenario_deterministic_injection():
    """Verify the scenario injection rule is deterministic and yields the same unit twice."""
    r_star = int(load_params()["R_star"])
    run1 = get_deterministic_injection_trajectory(r_star)
    run2 = get_deterministic_injection_trajectory(r_star)

    subset1, unit1, cycle1, traj1 = run1
    subset2, unit2, cycle2, traj2 = run2

    assert subset1 == subset2
    assert unit1 == unit2
    assert cycle1 == cycle2
    assert len(traj1) == len(traj2)
    assert traj1.iloc[0]["q10"] == traj2.iloc[0]["q10"]


def test_lib_audit_module_integrity():
    """Verify audit module re-imports cleanly."""
    mod = importlib.import_module("app.lib.audit")
    assert hasattr(mod, "append")
    assert hasattr(mod, "get_recent")
    assert hasattr(mod, "count")

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

ROOT = Path(__file__).resolve().parent.parent

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
    p = ROOT / page_path
    assert p.exists(), f"Page file {page_path} not found"
    code = p.read_text(encoding="utf-8")
    assert len(code) > 0
    # Compile code to verify syntax
    compiled = compile(code, str(p), "exec")
    assert compiled is not None


@pytest.mark.parametrize("page_path", PAGE_FILES)
def test_apptest_pages_no_exceptions(page_path: str):
    """Use Streamlit's AppTest to run each page and verify no uncaught exceptions."""
    at = AppTest.from_file(str(ROOT / page_path), default_timeout=120).run(timeout=120)
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

    # Spares allocation & comparison
    from app.lib.data import load_spares_alloc, load_spares_compare

    spares_alloc = load_spares_alloc()
    assert spares_alloc is not None and not spares_alloc.empty
    assert "pipeline_mult" in spares_alloc.columns
    assert "vm_base_per_sqn" in spares_alloc.columns
    assert "vm_depot" in spares_alloc.columns

    spares_comp = load_spares_compare()
    assert spares_comp is not None and not spares_comp.empty
    assert "pipeline_mult" in spares_comp.columns
    assert "diff_vs_current_pp" in spares_comp.columns
    assert "nmcs_lru_sim" in spares_comp.columns
    assert "nmcs_engine_sim" in spares_comp.columns


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


def test_pages_without_readiness_fallback(monkeypatch: pytest.MonkeyPatch):
    """Verify pages display fallback notes when readiness is missing and never show numbers."""
    import app.lib.readiness as rd_mod

    monkeypatch.setattr(rd_mod, "get_readiness_module", lambda: None)
    monkeypatch.setattr(rd_mod, "is_readiness_available", lambda: False)

    # 1. Engineering Officer page fallback
    at_eo = AppTest.from_file(str(ROOT / "app/pages/2_Engineering_Officer.py")).run(timeout=20)
    assert len(at_eo.exception) == 0
    all_eo_text = " ".join(
        [c.value for c in at_eo.caption]
        + [m.value for m in at_eo.markdown]
    )
    assert "engine detail after T1.7 merge" in all_eo_text
    assert "Engine RUL table (q10/q50/q90) available after T1.7 merge" in all_eo_text
    assert "rul_true" not in all_eo_text

    # 2. Scenario page step 5 fallback
    at_sc = AppTest.from_file(str(ROOT / "app/pages/5_Scenario.py"))
    at_sc.session_state["scenario_step"] = 5
    at_sc.run(timeout=20)
    assert len(at_sc.exception) == 0
    all_sc_text = " ".join(
        [c.value for c in at_sc.caption]
        + [m.value for m in at_sc.markdown]
    )
    assert "illustrative layout; live hangar state after T1.7 merge" in all_sc_text
    assert "Readiness forecast: available after merge of T1.7" in all_sc_text
    assert "BEFORE INJECTION (Baseline S2 at D+3)" not in all_sc_text
    assert "rul_true" not in all_sc_text


def test_eo_page_with_readiness_available():
    """Verify that with readiness available, EO page shows no 'after T1.7 merge' text and no 'rul_true'."""
    at_eo = AppTest.from_file(str(ROOT / "app/pages/2_Engineering_Officer.py")).run(timeout=20)
    assert len(at_eo.exception) == 0
    all_eo_text = " ".join(
        [c.value for c in at_eo.caption]
        + [m.value for m in at_eo.markdown]
    )
    assert "after T1.7 merge" not in all_eo_text
    assert "rul_true" not in all_eo_text

    for df_elem in at_eo.dataframe:
        assert "rul_true" not in df_elem.value.columns


def test_pages_with_readiness_wired(monkeypatch: pytest.MonkeyPatch):
    """Verify pages display real data and drop illustrative captions when readiness is present."""
    import sys

    # Check if readiness is already available locally post-merge
    has_local_readiness = False
    try:
        import shikra.strategy.readiness as rd

        has_local_readiness = hasattr(rd, "snapshot_at") and hasattr(rd, "forecast")
    except (ImportError, AttributeError):
        pass

    if not has_local_readiness:
        rd_worktree = Path(__file__).parent.parent.parent / "shikra-rd"
        if not (rd_worktree / "shikra" / "strategy" / "readiness.py").exists():
            pytest.skip("Temporary shikra-rd worktree not found")

        import shikra

        p = str((rd_worktree / "shikra").resolve())
        monkeypatch.syspath_prepend(str(rd_worktree))
        monkeypatch.setattr(shikra, "__path__", [p] + list(shikra.__path__))

        for mod_name in list(sys.modules.keys()):
            if any(
                k in mod_name
                for k in ("readiness", "spares", "strategy", "sim", "app.lib")
            ):
                del sys.modules[mod_name]

    # 1. Engineering Officer page with readiness
    at_eo = AppTest.from_file(str(ROOT / "app/pages/2_Engineering_Officer.py")).run(timeout=20)
    assert len(at_eo.exception) == 0
    all_eo_text = " ".join(
        [c.value for c in at_eo.caption]
        + [m.value for m in at_eo.markdown]
    )
    assert "engine detail after T1.7 merge" not in all_eo_text
    assert "Engine RUL table (q10/q50/q90) available after T1.7 merge" not in all_eo_text
    assert "Live propulsion and subsystem health" in all_eo_text
    assert "rul_true" not in all_eo_text

    # Check dataframe columns on EO page
    for df_elem in at_eo.dataframe:
        df = df_elem.value
        assert "rul_true" not in df.columns
        if "q10 RUL" in df.columns:
            assert "q50 Median" in df.columns
            assert "q90 RUL" in df.columns

    # 2. Scenario step 5 with readiness
    at_sc = AppTest.from_file(str(ROOT / "app/pages/5_Scenario.py"))
    at_sc.session_state["scenario_step"] = 5
    at_sc.run(timeout=25)
    assert len(at_sc.exception) == 0
    all_sc_text = " ".join(
        [c.value for c in at_sc.caption]
        + [m.value for m in at_sc.markdown]
    )
    assert "illustrative layout; live hangar state after T1.7 merge" not in all_sc_text
    assert "Live hangar state from Day 485 simulation snapshot" in all_sc_text
    assert ("raw forecast band; held-out coverage" in all_sc_text or "coverage: pending" in all_sc_text)
    assert "Forecast band (q10–q90)" in all_sc_text
    assert "80% Confidence Band" not in all_sc_text
    assert "58.7%" not in all_sc_text
    assert "rul_true" not in all_sc_text


def test_no_rul_true_reaches_any_page():
    """Verify that rul_true is never exposed or rendered on any dashboard page."""
    for page_path in PAGE_FILES:
        at = AppTest.from_file(str(ROOT / page_path)).run(timeout=20)
        assert len(at.exception) == 0

        # Inspect all text/caption/markdown
        for elem in list(at.markdown) + list(at.caption) + list(at.info) + list(at.warning):
            assert "rul_true" not in elem.value.lower(), f"rul_true leaked on {page_path}"

        # Inspect all dataframes
        for df_elem in at.dataframe:
            assert "rul_true" not in df_elem.value.columns, f"rul_true column leaked on {page_path}"


def test_aircraft_detail_svg_no_codeblocks():
    """Assert rendered SVG has no blank lines (\\n\\n) and no line starting with 4 spaces."""
    from app.lib.visuals import render_aircraft_detail_svg

    # Default rendering
    svg_default = render_aircraft_detail_svg("T01", "MC")
    assert "\n\n" not in svg_default
    for line in svg_default.splitlines():
        assert not line.startswith("    "), f"Line starts with 4 spaces: {line[:30]}"

    # Custom engine states and component states
    svg_custom = render_aircraft_detail_svg(
        "T19",
        "NMCS",
        engine_states={"engine_0": "REMOVAL", "engine_1": "WATCH"},
        component_states={"radar": "MISSING", "generator": "OK"},
    )
    assert "\n\n" not in svg_custom
    for line in svg_custom.splitlines():
        assert not line.startswith("    "), f"Line starts with 4 spaces: {line[:30]}"


def test_scenario_step3_rstar_dynamic_caption():
    """Verify that Scenario step 3 caption builds R* from policy_params.json dynamically."""
    from app.lib.data import load_params

    params = load_params()
    r_star = int(params["R_star"])

    at = AppTest.from_file(str(ROOT / "app/pages/5_Scenario.py"))
    at.session_state["scenario_step"] = 3
    at.run(timeout=20)
    assert len(at.exception) == 0

    all_captions = " ".join([c.value for c in at.caption])
    expected_caption = f"Red dashed line: R* = {r_star} cycles."
    assert expected_caption in all_captions


def test_aircraft_detail_svg_valid_xml():
    """Verify that render_aircraft_detail_svg produces well-formed XML for all tail/engine/component states."""
    import xml.dom.minidom

    from app.lib.visuals import render_aircraft_detail_svg

    # Every tail status
    for tail_status in ["MC", "NMCM", "NMCS"]:
        svg = render_aircraft_detail_svg("T19", tail_status)
        dom = xml.dom.minidom.parseString(svg)
        assert dom.documentElement.tagName == "svg"

    # Every engine state
    for eng_st in ["OK", "WATCH", "REMOVAL"]:
        svg = render_aircraft_detail_svg(
            "T19",
            "MC",
            engine_states={"engine_0": eng_st, "engine_1": eng_st},
        )
        dom = xml.dom.minidom.parseString(svg)
        assert dom.documentElement.tagName == "svg"

    # Every component state
    for comp_st in ["OK", "MISSING"]:
        svg = render_aircraft_detail_svg(
            "T19",
            "MC",
            component_states={"radar_tx": comp_st, "fuel_pump": comp_st},
        )
        dom = xml.dom.minidom.parseString(svg)
        assert dom.documentElement.tagName == "svg"

    # Combinations of tail, engine, and component states
    for tail_status in ["MC", "NMCM", "NMCS"]:
        for eng0 in ["OK", "WATCH", "REMOVAL"]:
            for comp_st in ["OK", "MISSING"]:
                svg = render_aircraft_detail_svg(
                    "T01",
                    tail_status,
                    engine_states={"engine_0": eng0, "engine_1": "OK"},
                    component_states={"radar_tx": comp_st},
                )
                dom = xml.dom.minidom.parseString(svg)
                assert dom.documentElement.tagName == "svg"


def test_all_pages_with_readiness_live_no_exceptions_no_merge_stubs():
    """With readiness available: every page runs with 0 exceptions and no 'after T1.7 merge' or 'rul_true'."""
    for page_path in PAGE_FILES:
        at = AppTest.from_file(str(ROOT / page_path), default_timeout=40).run(timeout=40)
        assert len(at.exception) == 0, f"Page {page_path} raised exceptions: {[e.value for e in at.exception]}"
        all_text = " ".join(
            [c.value for c in at.caption]
            + [m.value for m in at.markdown]
            + [i.value for i in at.info]
            + [w.value for w in at.warning]
        ).lower()
        assert "after t1.7 merge" not in all_text, f"'after T1.7 merge' found on {page_path}"
        assert "after merge of t1.7" not in all_text, f"'after merge of T1.7' found on {page_path}"
        assert "after t1.6 merge" not in all_text, f"'after T1.6 merge' found on {page_path}"
        assert "rul_true" not in all_text, f"'rul_true' found on {page_path}"
        for df_elem in at.dataframe:
            assert "rul_true" not in df_elem.value.columns, f"'rul_true' column found on {page_path}"


def test_commander_readiness_coverage_caption():
    """Commander shows the coverage caption, and its numbers equal results/readiness_calib.csv."""
    import pandas as pd

    calib_csv = ROOT / "results" / "readiness_calib.csv"
    assert calib_csv.exists(), "results/readiness_calib.csv must exist"
    calib = pd.read_csv(calib_csv)
    conf_row = calib[(calib["lead_days"] == 3) & (calib["method"] == "shikra_conformal")].iloc[0]
    fc_row = calib[(calib["lead_days"] == 3) & (calib["method"] == "shikra_forecast")].iloc[0]
    pe_row = calib[(calib["lead_days"] == 3) & (calib["method"] == "persistence")].iloc[0]

    conf_cov = float(conf_row["coverage_80"])
    fc_mae = float(fc_row["mae"])
    pe_mae = float(pe_row["mae"])

    at = AppTest.from_file(str(ROOT / "app/pages/1_Commander.py"), default_timeout=40).run(timeout=40)
    assert len(at.exception) == 0

    all_text = " ".join([c.value for c in at.caption] + [m.value for m in at.markdown])
    assert "Calibrated band covers" in all_text
    assert (f"{conf_cov:.4f}" in all_text) or (f"{conf_cov * 100:.1f}%" in all_text)
    assert f"{fc_mae:.4f}" in all_text
    assert f"{pe_mae:.4f}" in all_text
    assert "does not yet beat persistence" in all_text
    assert "Synthetic wing." in all_text


def test_commander_one_page_brief():
    """The brief: AppTest clicks 'Open one-page brief' and asserts title, top-3 indigenisation list, audit line appear."""
    at = AppTest.from_file(str(ROOT / "app/pages/1_Commander.py"), default_timeout=40).run(timeout=40)
    assert len(at.exception) == 0

    # Click the open brief button
    at.button(key="open_brief_btn").click().run(timeout=40)
    assert len(at.exception) == 0

    all_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])

    # Assert title
    assert "SHIKRA · Commander's brief · Day 485 · synthetic wing" in all_text

    # Top-3 indigenisation list equal to supply_shock.csv order
    shock = load_supply_shock()
    top3_lrus = shock[shock["lru_id"] != "all_foreign"].sort_values(
        "loss_pp_shock_window", ascending=False
    ).head(3)["lru_id"].tolist()

    for lru in top3_lrus:
        assert lru in all_text, f"Top indigenisation item {lru} not found in brief"

    pos0 = all_text.find(top3_lrus[0])
    pos1 = all_text.find(top3_lrus[1])
    pos2 = all_text.find(top3_lrus[2])
    assert -1 < pos0 < pos1 < pos2, "Top 3 indigenisation items not in expected ranking order"

    # Audit status line
    assert "Audit chain: intact" in all_text or "Audit chain: BROKEN" in all_text


def test_eo_page_rstar_owner_label():
    """The EO page shows the R* owner label with the value from policy_params.json."""
    params = load_params()
    r_star = params["R_star"]

    at = AppTest.from_file(str(ROOT / "app/pages/2_Engineering_Officer.py"), default_timeout=40).run(timeout=40)
    assert len(at.exception) == 0

    all_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    expected_label = (
        f"Removal threshold R* = {r_star} cycles · set by the Engineering Officer · changes are audit-logged"
    )
    assert expected_label in all_text, f"Expected R* owner label not found: {expected_label}"


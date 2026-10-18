"""Tests for simulation snapshot hook and readiness forecasting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shikra.rul.variants import apply
from shikra.sim.engine import run
from shikra.spares.demand import inverse_removal_prob, removal_prob

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"


def test_inverse_removal_prob():
    """Verify that inverse_removal_prob correctly inverts removal_prob piecewise linear CDF."""
    q10, q50, q90 = 20.0, 50.0, 90.0

    # Knot checks
    assert pytest.approx(inverse_removal_prob(q10, q50, q90, 0.10)) == 20.0
    assert pytest.approx(inverse_removal_prob(q10, q50, q90, 0.50)) == 50.0
    assert pytest.approx(inverse_removal_prob(q10, q50, q90, 0.90)) == 90.0
    assert inverse_removal_prob(q10, q50, q90, 0.0) == 0.0

    # Round trip checks across u in [0.01, 0.99]
    for u in [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]:
        h = inverse_removal_prob(q10, q50, q90, u)
        p = removal_prob(q10, q50, q90, h)
        assert pytest.approx(p, abs=1e-6) == u

    # Degenerate case
    assert inverse_removal_prob(50.0, 50.0, 50.0, 0.4) == 50.0
    assert inverse_removal_prob(60.0, 50.0, 40.0, 0.8) == 50.0


def test_snapshots_identical_results():
    """Verify that running with snapshot_days produces byte-identical simulation KPIs and histories.

    The snapshot hook must not draw random numbers or alter any simulation state.
    """
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    seed = 1
    days = 60
    snapshot_days = [15, 30, 45]

    # Run without snapshots
    res_no_snap = run("shikra", cfg, rul_shikra, params, seed=seed, days=days, snapshot_days=None)
    assert res_no_snap.snapshots is None

    # Run with snapshots
    res_with_snap = run("shikra", cfg, rul_shikra, params, seed=seed, days=days, snapshot_days=snapshot_days)
    assert res_with_snap.snapshots is not None
    assert set(res_with_snap.snapshots.keys()) == set(snapshot_days)

    # 1. KPIs must match identically
    for key, val in res_no_snap.kpis.items():
        assert res_with_snap.kpis[key] == val, f"KPI {key} drifted"

    # 2. Daily records must match identically
    pd.testing.assert_frame_equal(res_no_snap.daily, res_with_snap.daily)

    # 3. Events must match identically
    pd.testing.assert_frame_equal(res_no_snap.events, res_with_snap.events)

    # 4. Tail status must match identically
    assert res_no_snap.tail_status == res_with_snap.tail_status


def test_snapshot_structure_and_fields():
    """Verify snapshot dictionary schema contains all required wing and logistics fields."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    res = run("shikra", cfg, rul_shikra, params, seed=1, days=35, snapshot_days=[30])
    snap = res.snapshots[30]

    assert snap["day"] == 30
    assert "tails" in snap
    assert len(snap["tails"]) == 54
    assert all("status" in t and "sqn" in t and "return_day" in t for t in snap["tails"])

    assert "engines" in snap
    assert len(snap["engines"]) == 108
    assert all(
        "q10" in e and "q50" in e and "q90" in e and "rul_true" in e and "cycle" in e
        for e in snap["engines"]
    )

    assert "lrus" in snap
    assert len(snap["lrus"]) == 54  # per tail
    assert "stock" in snap
    assert "engines" in snap["stock"] and "lrus" in snap["stock"]
    assert "queues" in snap
    assert "hangar_busy" in snap


def test_forecast_never_reads_rul_true():
    """Verify that forecast NEVER reads rul_true.

    Setting rul_true to invalid garbage values in the snapshot must leave the
    forecast outputs 100% byte-identical.
    """
    import copy

    from shikra.strategy.readiness import forecast

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    res = run("shikra", cfg, rul_shikra, params, seed=1, days=35, snapshot_days=[30])
    snap_clean = res.snapshots[30]

    # Clean forecast
    fc_clean = forecast(snap_clean, cfg, params, policy="shikra", horizon_days=7, n=50, seed=42)

    # Corrupt rul_true with negative, infinite, and nonsense values
    snap_corrupt = copy.deepcopy(snap_clean)
    for eng in snap_corrupt["engines"]:
        eng["rul_true"] = -99999.0

    fc_corrupt = forecast(snap_corrupt, cfg, params, policy="shikra", horizon_days=7, n=50, seed=42)

    pd.testing.assert_frame_equal(fc_clean, fc_corrupt)


def test_forecast_columns_and_determinism():
    """Verify forecast DataFrame structure and determinism."""
    from shikra.strategy.readiness import forecast

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    res = run("shikra", cfg, rul_shikra, params, seed=1, days=35, snapshot_days=[30])
    snap = res.snapshots[30]

    df1 = forecast(snap, cfg, params, policy="shikra", horizon_days=7, n=50, seed=123)
    df2 = forecast(snap, cfg, params, policy="shikra", horizon_days=7, n=50, seed=123)
    pd.testing.assert_frame_equal(df1, df2)

    # Check required columns
    expected_cols = ["lead_days", "sqn", "expected_mc", "mc_q10", "mc_q90"] + [
        f"p_ge_{i}" for i in range(6, 19)
    ]
    for col in expected_cols:
        assert col in df1.columns

    assert len(df1) == 7 * 3  # 7 lead days * 3 squadrons


def test_rank_actions():
    """Verify rank_actions returns top candidate actions with correct metadata."""
    from shikra.strategy.readiness import rank_actions

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    res = run("shikra", cfg, rul_shikra, params, seed=1, days=35, snapshot_days=[30])
    snap = res.snapshots[30]

    actions_df = rank_actions(snap, cfg, params, sqn=1, lead_days=3, n=50, seed=99)
    assert len(actions_df) <= 5
    assert "rank" in actions_df.columns
    assert "action" in actions_df.columns
    assert "target" in actions_df.columns
    assert "delta_expected_mc" in actions_df.columns
    assert actions_df.attrs["note"] == "forecast-estimated gain (not validated in simulation)"

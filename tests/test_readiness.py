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

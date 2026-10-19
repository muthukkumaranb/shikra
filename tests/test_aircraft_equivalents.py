"""Tests for scripts/aircraft_equivalents.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "aircraft_equivalents", ROOT / "scripts" / "aircraft_equivalents.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_lever_row_recomputed_by_hand(tmp_path: Path):
    _load().main(tmp_path / "eq.csv")
    out = pd.read_csv(tmp_path / "eq.csv")

    wing = yaml.safe_load((ROOT / "config" / "wing.yaml").read_text(encoding="utf-8"))
    n = wing["fleet"]["squadrons"] * wing["fleet"]["aircraft_per_sqn"]
    hm = pd.read_csv(ROOT / "results" / "heatmap.csv")
    fast = hm["pipeline_mult"].min()
    r_fast = hm.loc[hm["pipeline_mult"] == fast, "reactive_serviceability_mean"].iloc[0]
    r_today = hm.loc[
        hm["pipeline_mult"] == wing["pipeline_mult"], "reactive_serviceability_mean"
    ].iloc[0]
    expected_pp = (r_fast - r_today) * 100
    expected_ac = expected_pp / 100 * n

    row = out[out["item"].str.startswith("pipeline lever")].iloc[0]
    assert abs(row["pp_or_value"] - expected_pp) < 1e-3
    assert abs(row["aircraft_equivalent"] - expected_ac) < 1e-2
    assert out["note"].str.contains("synthetic wing").all()


def test_committed_csv_matches_regeneration(tmp_path: Path):
    _load().main(tmp_path / "eq.csv")
    committed = ROOT / "results" / "aircraft_equivalents.csv"
    assert committed.read_text(encoding="utf-8") == (tmp_path / "eq.csv").read_text(
        encoding="utf-8"
    )

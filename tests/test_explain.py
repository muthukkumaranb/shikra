"""Tests for scripts/explain_contrib.py: explanations must not change predictions.

The data-marked tests rebuild replay features from data/raw (``make data``) and
must run in the pinned requirements.txt environment: other LightGBM/pandas
versions flip a few tree splits and the parquet check would fail by design.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "results" / "explain_contrib.csv"
HAS_DATA = (ROOT / "data" / "raw" / "cmapss" / "train_FD001.txt").exists()


def _load():
    spec = importlib.util.spec_from_file_location(
        "explain_contrib", ROOT / "scripts" / "explain_contrib.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def res():
    return _load().compute()


@pytest.fixture(scope="module")
def sample(res) -> np.ndarray:
    rng = np.random.default_rng(2026)
    return np.sort(rng.choice(len(res["meta"]), size=200, replace=False))


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="needs data/raw (make data)")
def test_parts_sum_to_raw_q10(res, sample):
    parts = res["base"][sample] + res["feat"][sample].sum(axis=1)
    err = np.abs(parts - res["raw_q10"][sample]).max()
    print(f"\n200 rows: max |base + contributions - raw seed-averaged q10| = {err:.3e}")
    assert err < 1e-6


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="needs data/raw (make data)")
def test_offset_reproduces_replay_q10(res, sample):
    replay = pd.read_parquet(
        ROOT / "artifacts" / "replay_rul.parquet", columns=["subset", "unit", "cycle", "q10"]
    )
    rows = res["meta"].iloc[sample]
    stored = rows.merge(replay, on=["subset", "unit", "cycle"], how="left", validate="1:1")
    shown = res["raw_q10"][sample] + res["offset"][sample]
    err = np.abs(shown - stored["q10"].to_numpy()).max()
    print(f"\n200 rows: max |raw q10 + calibration offset - replay_rul.parquet q10| = {err:.3e}")
    assert err < 1e-6


@pytest.mark.data
@pytest.mark.skipif(not HAS_DATA, reason="needs data/raw (make data)")
def test_byte_identical_and_frozen_files_untouched(tmp_path: Path):
    mod = _load()
    mod.main(tmp_path / "run1.csv")
    mod.main()  # default path: regenerates results/explain_contrib.csv in place
    assert (tmp_path / "run1.csv").read_bytes() == CSV_PATH.read_bytes()
    diff = subprocess.run(
        ["git", "diff", "--stat", "--", "artifacts", "results/*.csv"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    print(f"\ngit diff --stat -- artifacts results/*.csv: {diff.stdout!r}")
    assert diff.stdout == ""


def _panel_script(subset: str, unit: int, cycle: int, r_star: float, missing: bool) -> None:
    from pathlib import Path as _Path

    import app.lib.explain_panel as ep

    if missing:
        ep.CONTRIB_PATH = _Path("does/not/exist.csv")
        ep.GLOBAL_PATH = _Path("does/not/exist_either.csv")
    ep.render_why_flagged(subset, unit, cycle, r_star)


def _pick_engine(flagged: bool) -> tuple[str, int, int, float]:
    import json

    r_star = float(json.loads((ROOT / "artifacts" / "policy_params.json").read_text())["R_star"])
    rp = pd.read_parquet(
        ROOT / "artifacts" / "replay_rul.parquet", columns=["subset", "unit", "cycle", "q10"]
    ).sort_values(["subset", "unit", "cycle"])
    row = rp[rp["q10"] <= r_star].iloc[0] if flagged else rp[rp["q10"] > r_star].iloc[0]
    return str(row["subset"]), int(row["unit"]), int(row["cycle"]), r_star


@pytest.mark.parametrize("flagged", [True, False])
def test_panel_renders(flagged: bool):
    from streamlit.testing.v1 import AppTest

    if not CSV_PATH.exists():
        pytest.skip("results/explain_contrib.csv not generated")
    args = (*_pick_engine(flagged), False)
    at = AppTest.from_function(_panel_script, args=args).run(timeout=60)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Why is this engine flagged?" in text
    assert ("REMOVAL DUE" in text) is flagged
    assert ("→ OK" in text) is not flagged
    assert "rul_true" not in text


def test_panel_without_csv_does_not_crash():
    from streamlit.testing.v1 import AppTest

    args = (*_pick_engine(True), True)
    at = AppTest.from_function(_panel_script, args=args).run(timeout=60)
    assert not at.exception
    assert any("make explain" in i.value for i in at.info)


def test_csv_has_no_ground_truth():
    if not CSV_PATH.exists():
        pytest.skip("results/explain_contrib.csv not generated")
    header = CSV_PATH.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [
        "subset", "unit", "cycle", "rank", "feature", "sensor_label", "contribution"
    ]
    assert "rul_true" not in CSV_PATH.read_text(encoding="utf-8")
    assert CSV_PATH.stat().st_size < 5_000_000

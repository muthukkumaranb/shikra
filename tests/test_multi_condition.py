"""Tests for multi-condition RUL prediction (T1.5)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shikra.rul.multi_condition import (
    compute_condition_history,
    generate_multi_splits,
)

ROOT = Path(__file__).parent.parent
SPLITS_MULTI_PATH = ROOT / "artifacts" / "splits_multi.json"


def test_condition_history_causality():
    """Verify that cycles_in_cond_k at cycle c depends strictly on cycles <= c."""
    n_rows = 50
    clusters = np.random.default_rng(42).integers(0, 6, size=n_rows)
    df = pd.DataFrame({
        "unit": 1,
        "cycle": np.arange(1, n_rows + 1),
        "cluster": clusters,
    })

    full_hist = compute_condition_history(df, n_clusters=6)

    # For any prefix length k < n_rows, computing on truncated history must yield
    # identical values for all cycles 1..k
    for k in [5, 15, 30, 45]:
        truncated_df = df.iloc[:k].copy()
        trunc_hist = compute_condition_history(truncated_df, n_clusters=6)
        pd.testing.assert_frame_equal(
            full_hist.iloc[:k].reset_index(drop=True),
            trunc_hist.reset_index(drop=True),
            check_dtype=False,
        )

    # Future modification test: modifying future clusters cannot alter past counts
    df_modified = df.copy()
    df_modified.loc[30:, "cluster"] = 0
    mod_hist = compute_condition_history(df_modified, n_clusters=6)
    pd.testing.assert_frame_equal(
        full_hist.iloc[:30].reset_index(drop=True),
        mod_hist.iloc[:30].reset_index(drop=True),
        check_dtype=False,
    )


def test_condition_history_sums_to_cycle():
    """Cumulative counts across all conditions must equal the cycle count."""
    clusters = np.array([0, 1, 1, 2, 5, 3, 4, 0, 1, 2])
    df = pd.DataFrame({"cycle": np.arange(1, len(clusters) + 1), "cluster": clusters})
    hist = compute_condition_history(df, n_clusters=6)
    row_sums = hist.sum(axis=1)
    np.testing.assert_array_equal(row_sums.values, df["cycle"].values)


@pytest.mark.data
def test_splits_multi_engine_disjoint():
    """splits_multi.json must be engine-disjoint per subset."""
    if not SPLITS_MULTI_PATH.exists():
        splits = generate_multi_splits(seed=2026)
    else:
        with SPLITS_MULTI_PATH.open("r", encoding="utf-8") as f:
            splits = json.load(f)

    for subset in ["FD002", "FD004"]:
        assert subset in splits, f"Missing {subset} in splits_multi"
        tr = set(splits[subset]["train"])
        val = set(splits[subset]["val"])

        # Engine disjointness
        overlap = tr.intersection(val)
        assert len(overlap) == 0, f"{subset} train and val overlap: {overlap}"

        # 80/20 proportion
        total = len(tr) + len(val)
        if subset == "FD002":
            assert total == 260
            assert len(tr) == 208
            assert len(val) == 52
        elif subset == "FD004":
            assert total == 249
            assert len(tr) == 199
            assert len(val) == 50

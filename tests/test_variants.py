"""Tests for RUL quality variants (shikra/rul/variants.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shikra.rul.variants import apply


def _make_df() -> pd.DataFrame:
    """Create a tiny replay-table-shaped DataFrame for testing."""
    return pd.DataFrame(
        {
            "subset": ["FD001"] * 5,
            "unit": [1] * 5,
            "cycle": [1, 2, 3, 4, 5],
            "life": [10] * 5,
            "rul_true": [9, 8, 7, 6, 5],
            "q10": [6, 5, 4, 3, 2],
            "q50": [9, 8, 7, 6, 5],
            "q90": [12, 11, 10, 9, 8],
            "model_id": ["abc"] * 5,
            "calib_id": ["def"] * 5,
        }
    )


def test_oracle_equals_truth() -> None:
    """quality=0 → q10 = q50 = q90 = rul_true (capped at 125)."""
    df = _make_df()
    out = apply(df, 0)
    np.testing.assert_array_equal(out["q50"].values, np.minimum(df["rul_true"].values, 125))
    np.testing.assert_array_equal(out["q10"].values, out["q50"].values)
    np.testing.assert_array_equal(out["q90"].values, out["q50"].values)


def test_identity() -> None:
    """quality=1 → quantiles unchanged."""
    df = _make_df()
    out = apply(df, 1)
    np.testing.assert_array_equal(out["q10"].values, df["q10"].values)
    np.testing.assert_array_equal(out["q50"].values, df["q50"].values)
    np.testing.assert_array_equal(out["q90"].values, df["q90"].values)


def test_k2_doubles_error() -> None:
    """quality=2 → each quantile becomes rul_true + 2*(q - rul_true), i.e. doubles abs error."""
    df = _make_df()
    out = apply(df, 2)
    expected_q50 = np.maximum(
        df["rul_true"].values + 2 * (df["q50"].values - df["rul_true"].values), 0
    )
    np.testing.assert_allclose(out["q50"].values, expected_q50, atol=1e-10)


def test_nonnegative_output() -> None:
    """Output quantiles must always be ≥ 0."""
    df = _make_df()
    for k in (0, 1, 2, 3):
        out = apply(df, k)
        assert (out["q10"] >= 0).all(), f"q10 < 0 for quality={k}"
        assert (out["q50"] >= 0).all()
        assert (out["q90"] >= 0).all()

"""Tests for causal feature builder (shikra/rul/features.py).

The key test: features at cycle c depend only on rows ≤ c.
Uses a synthetic fixture — no raw data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shikra.rul.features import build_features, fit_scaler, get_informative_sensors


def _make_synthetic_df(n_cycles: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sensors = [f"s{i}" for i in range(1, 22)]
    rows = []
    for cycle in range(1, n_cycles + 1):
        row: dict = {
            "subset": "FD001",
            "unit": 1,
            "cycle": cycle,
            "op1": 0.0,
            "op2": 0.0,
            "op3": 100.0,
            "life": n_cycles,
            "rul_true": n_cycles - cycle,
        }
        for s in sensors:
            row[s] = rng.normal(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_informative_sensors_count() -> None:
    """Should return 14 sensors (21 total minus 7 near-constant)."""
    sensors = get_informative_sensors()
    assert len(sensors) == 14


def test_feature_causality() -> None:
    """Features at cycle c must be identical regardless of whether later rows exist."""
    df_full = _make_synthetic_df(n_cycles=30)
    df_prefix = df_full[df_full["cycle"] <= 20].copy()

    scaler = fit_scaler(df_full)

    X_full, _, meta_full = build_features(df_full, scaler)
    X_prefix, _, meta_prefix = build_features(df_prefix, scaler)

    # Find rows at cycle 15 in both
    idx_full = meta_full[meta_full["cycle"] == 15].index[0]
    idx_prefix = meta_prefix[meta_prefix["cycle"] == 15].index[0]

    feats_full = X_full[idx_full]
    feats_prefix = X_prefix[idx_prefix]

    np.testing.assert_allclose(
        feats_full, feats_prefix, atol=1e-10,
        err_msg="Features at cycle 15 differ between full and prefix trajectory!"
    )


def test_feature_matrix_shape() -> None:
    """X must have shape (n_rows, n_features) with n_rows matching the DataFrame."""
    df = _make_synthetic_df(n_cycles=20)
    scaler = fit_scaler(df)
    X, y, meta = build_features(df, scaler)
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0
    assert y.shape == (len(df),)


def test_rul_cap() -> None:
    """y values must all be ≤ RUL_CAP."""
    from shikra.rul.features import RUL_CAP

    df = _make_synthetic_df(n_cycles=200)  # long trajectory exceeds cap
    scaler = fit_scaler(df)
    _, y, _ = build_features(df, scaler)
    assert np.all(y <= RUL_CAP), f"Some y values exceed RUL_CAP={RUL_CAP}"

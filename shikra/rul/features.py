"""Causal rolling features for the RUL model in SHIKRA.

Computes per-unit, causal rolling statistics (mean, std, slope) over multiple
windows for the informative C-MAPSS sensors.  The scaler is fit **only** on
train-split units.

All features are causal: features at cycle ``c`` depend only on rows ≤ ``c``.

Public API
----------
get_informative_sensors() -> list[str]
fit_scaler(df_train) -> StandardScaler
build_features(df, scaler, rul_cap) -> (X: np.ndarray, y: np.ndarray, meta: pd.DataFrame)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------

# Near-constant sensors to drop for FD001/FD003 (single operating condition).
# Verified by near-zero std on training data; these are hard-coded to ensure
# reproducibility without needing to re-check every run.
NEAR_CONSTANT_SENSORS: frozenset[str] = frozenset(
    ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
)

ALL_SENSORS: list[str] = [f"s{i}" for i in range(1, 22)]

RUL_CAP = 125  # piecewise-linear cap


def get_informative_sensors() -> list[str]:
    """Return the list of informative sensor columns (op3 and near-constants excluded)."""
    return [s for s in ALL_SENSORS if s not in NEAR_CONSTANT_SENSORS]


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


def fit_scaler(df_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on train-split sensor columns only.

    Parameters
    ----------
    df_train:
        DataFrame of train-split rows (must contain all informative sensors).

    Returns
    -------
    Fitted ``StandardScaler``.
    """
    sensors = get_informative_sensors()
    scaler = StandardScaler()
    scaler.fit(df_train[sensors].values)
    return scaler


# ---------------------------------------------------------------------------
# Causal rolling helpers
# ---------------------------------------------------------------------------

WINDOWS = [5, 10, 20, 30]


def _causal_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute least-squares slope of a rolling window (causal, min_periods=1)."""
    result = np.empty(len(series), dtype=float)
    arr = series.to_numpy(dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        chunk = arr[lo : i + 1]
        n = len(chunk)
        if n < 2:
            result[i] = 0.0
        else:
            x = np.arange(n, dtype=float)
            xm = x.mean()
            ym = chunk.mean()
            denom = ((x - xm) ** 2).sum()
            result[i] = ((x - xm) * (chunk - ym)).sum() / denom if denom > 0 else 0.0
    return pd.Series(result, index=series.index)


def _unit_features(unit_df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Build causal rolling features for a single unit (already sorted by cycle)."""
    frames: list[pd.Series] = []

    frames.append(unit_df["cycle"].rename("cycle"))

    for s in sensors:
        col = unit_df[s]
        # Last value (current reading)
        frames.append(col.rename(f"{s}_last"))
        # Rolling mean and std over each window
        for w in WINDOWS:
            frames.append(
                col.rolling(w, min_periods=1).mean().rename(f"{s}_mean{w}")
            )
            frames.append(
                col.rolling(w, min_periods=1).std(ddof=0).fillna(0).rename(f"{s}_std{w}")
            )
            # Causal slope
            frames.append(_causal_slope(col, w).rename(f"{s}_slope{w}"))

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# Public feature builder
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    rul_cap: int = RUL_CAP,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix and target vector from a C-MAPSS DataFrame.

    Parameters
    ----------
    df:
        DataFrame (train or replay split) with columns including
        ``unit``, ``cycle``, informative sensors, and ``rul_true``.
    scaler:
        Fitted ``StandardScaler`` (fit on train units only).
    rul_cap:
        Piecewise-linear RUL cap.  Default is 125.

    Returns
    -------
    X : np.ndarray, shape (n_rows, n_features)
    y : np.ndarray, shape (n_rows,)   — ``min(rul_true, rul_cap)``
    meta : pd.DataFrame with ``subset``, ``unit``, ``cycle``, ``rul_true``
    """
    sensors = get_informative_sensors()

    # Scale sensor columns
    df = df.copy()
    df[sensors] = scaler.transform(df[sensors].values)

    # Build features per unit (maintains causal ordering)
    df = df.sort_values(["subset", "unit", "cycle"]).reset_index(drop=True)

    feat_frames: list[pd.DataFrame] = []
    for (subset, unit), grp in df.groupby(["subset", "unit"], sort=False):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        feats = _unit_features(grp, sensors)
        feats["subset"] = subset
        feats["unit"] = unit
        feats["rul_true"] = grp["rul_true"].values
        feat_frames.append(feats)

    feat_df = pd.concat(feat_frames, ignore_index=True)

    feature_cols = [c for c in feat_df.columns if c not in ("subset", "unit", "rul_true")]
    X = feat_df[feature_cols].values.astype(float)
    y = np.minimum(feat_df["rul_true"].values, rul_cap).astype(float)
    meta = feat_df[["subset", "unit", "cycle", "rul_true"]].copy()

    return X, y, meta

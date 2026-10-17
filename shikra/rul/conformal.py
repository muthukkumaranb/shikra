"""Conformal Quantile Regression (CQR) calibration for SHIKRA RUL bands.

Uses the calibration split (never the replay or test splits) to compute
nonconformity scores and the finite-sample-corrected quantile ``Q`` for
80% and 90% prediction bands.

The calibration set is constructed by sampling 10 truncation points per
calib unit (uniformly over its life), simulating the test condition where
we observe a prefix of the trajectory.

Public API
----------
build_calib_set(df_calib, scaler, models, n_trunc, seed) -> (X, y, preds)
calibrate(y, preds) -> dict   (containing Q values for both bands)
adjust(preds, Q_dict) -> dict (conformal-adjusted quantiles)
save_conformal(Q_dict, path)
load_conformal(path) -> dict
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from shikra.rul.features import RUL_CAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRUNC = 10          # truncation points per calib unit
BANDS = {
    0.80: (0.10, 0.90),   # (alpha_lo, alpha_hi)
    0.90: (0.05, 0.95),
}

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
CONFORMAL_PATH = ARTIFACTS_DIR / "conformal.json"


# ---------------------------------------------------------------------------
# Calibration set construction
# ---------------------------------------------------------------------------


def build_calib_rows(
    df_calib: pd.DataFrame,  # noqa: F821
    n_trunc: int = N_TRUNC,
    seed: int = 2026,
) -> pd.DataFrame:  # noqa: F821
    """Sample truncation points for each calib unit.

    For each calib unit, draws ``n_trunc`` cycle indices uniformly from
    [1, life] and returns those rows.  ``df_calib`` must be train-format
    (has ``rul_true`` and ``life`` columns).
    """
    import pandas as pd  # local to avoid circular import at module level

    rng = np.random.default_rng(seed)
    selected_rows: list[pd.DataFrame] = []

    for (subset, unit), grp in df_calib.groupby(["subset", "unit"], sort=False):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        life = grp["cycle"].max()
        # Sample n_trunc cycle positions uniformly
        chosen_cycles = rng.integers(1, life + 1, size=n_trunc)
        chosen_cycles = np.unique(chosen_cycles)  # deduplicate
        for c in chosen_cycles:
            # Use the prefix up to cycle c — take the row at exactly cycle c
            # (features are causal so the row at c already encodes history)
            row_idx = grp[grp["cycle"] == c]
            if len(row_idx) == 0:
                # Pick nearest available cycle
                diffs = (grp["cycle"] - c).abs()
                row_idx = grp.iloc[[diffs.argmin()]]
            selected_rows.append(row_idx)

    return pd.concat(selected_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# CQR calibration
# ---------------------------------------------------------------------------


def _nonconformity(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> np.ndarray:
    """Compute CQR nonconformity scores: E = max(q_lo - y, y - q_hi)."""
    return np.maximum(q_lo - y, y - q_hi)


def _finite_sample_quantile(E: np.ndarray, nominal_coverage: float) -> float:
    """Return the finite-sample corrected quantile of nonconformity scores.

    Q = empirical quantile at level ceil((1-alpha)*(1+1/n)) / n,
    capped at 1.0.
    """
    n = len(E)
    level = min((1 - (1 - nominal_coverage)) * (1 + 1 / n), 1.0)
    # Use ceiling as per standard CQR derivation
    idx = int(np.ceil(level * n)) - 1
    idx = np.clip(idx, 0, n - 1)
    E_sorted = np.sort(E)
    return float(E_sorted[idx])


def calibrate(
    y: np.ndarray,
    preds: dict[float, np.ndarray],
) -> dict:
    """Compute Q values for 80% and 90% bands.

    Parameters
    ----------
    y     : true RUL values (capped at RUL_CAP), shape (n,)
    preds : dict mapping alpha → predicted quantile array, shape (n,)

    Returns
    -------
    dict with keys ``Q_80``, ``Q_90`` (the additive correction values).
    """
    results: dict = {}

    for nominal_cov, (alpha_lo, alpha_hi) in BANDS.items():
        q_lo = preds[alpha_lo]
        q_hi = preds[alpha_hi]
        E = _nonconformity(y, q_lo, q_hi)
        Q = _finite_sample_quantile(E, nominal_cov)
        key = f"Q_{int(nominal_cov * 100)}"
        results[key] = Q
        print(f"  Band {int(nominal_cov*100)}%: Q = {Q:.3f}  (n={len(E)})")

    return results


def adjust(
    preds: dict[float, np.ndarray],
    Q_dict: dict,
) -> dict[float, np.ndarray]:
    """Return conformal-adjusted quantile predictions.

    Each band is widened by Q: [q_lo - Q, q_hi + Q], clipped to [0, RUL_CAP].

    Parameters
    ----------
    preds  : raw model predictions (dict alpha → array)
    Q_dict : output of ``calibrate()``

    Returns
    -------
    Adjusted predictions dict with same keys.
    """
    adjusted = {a: arr.copy() for a, arr in preds.items()}

    for nominal_cov, (alpha_lo, alpha_hi) in BANDS.items():
        Q = Q_dict[f"Q_{int(nominal_cov * 100)}"]
        adjusted[alpha_lo] = np.clip(preds[alpha_lo] - Q, 0, RUL_CAP)
        adjusted[alpha_hi] = np.clip(preds[alpha_hi] + Q, 0, RUL_CAP)

    # Re-enforce monotonicity after adjustment
    alphas_sorted = sorted(adjusted.keys())
    mat = np.stack([adjusted[a] for a in alphas_sorted], axis=1)
    mat = np.sort(mat, axis=1)
    mat = np.clip(mat, 0, RUL_CAP)

    return {a: mat[:, i] for i, a in enumerate(alphas_sorted)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_conformal(Q_dict: dict, path: Path = CONFORMAL_PATH) -> str:
    """Save Q_dict to JSON and return sha256[:12] of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(Q_dict, indent=2))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"Conformal params saved -> {path}  (sha256[:12]={sha})")
    return sha


def load_conformal(path: Path = CONFORMAL_PATH) -> dict:
    """Load conformal Q_dict from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Conformal calibration file not found: {path}\nRun `make rul` first."
        )
    data = json.loads(path.read_text())
    return data


def conformal_hash(path: Path = CONFORMAL_PATH) -> str:
    """Return sha256[:12] of the conformal JSON file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

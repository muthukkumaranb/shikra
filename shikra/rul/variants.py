"""RUL quality variants for oracle/degraded prediction scenarios.

Applies a quality multiplier to replay RUL quantiles for sensitivity analysis
and for the oracle policy (quality=0 → perfect predictions).

Public API
----------
apply(df, quality) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd

from shikra.rul.features import RUL_CAP


def apply(df: pd.DataFrame, quality: float) -> pd.DataFrame:
    """Apply a quality multiplier to the replay RUL quantiles.

    Parameters
    ----------
    df      : replay RUL DataFrame with columns ``rul_true, q10, q50, q90``.
    quality : float
        - ``0``   → oracle: ``q10 = q50 = q90 = rul_true``
        - ``1``   → identity: quantiles unchanged
        - ``k>1`` → each quantile becomes ``rul_true + k*(q - rul_true)``,
                    clipped to ≥ 0 (inflates errors)

    Returns
    -------
    Copy of ``df`` with ``q10``, ``q50``, ``q90`` updated.
    """
    out = df.copy()
    if quality == 0:
        # Oracle: perfect predictions
        out["q10"] = out["rul_true"].clip(0, RUL_CAP)
        out["q50"] = out["rul_true"].clip(0, RUL_CAP)
        out["q90"] = out["rul_true"].clip(0, RUL_CAP)
    elif quality == 1:
        # Identity — no change
        pass
    else:
        # Inflate errors by factor k
        k = float(quality)
        for col in ("q10", "q50", "q90"):
            out[col] = (
                out["rul_true"] + k * (out[col] - out["rul_true"])
            ).clip(lower=0)
    return out

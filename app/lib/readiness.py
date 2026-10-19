"""Safe interface for optional T1.7 readiness strategy integration.

Guards all calls with feature detection so the app functions gracefully
both before and after T1.7 is merged into main.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"


def get_readiness_module() -> Any | None:
    """Check if shikra.strategy.readiness has snapshot_at and forecast.

    Returns the module if both functions are present, else None.
    """
    try:
        import shikra.strategy.readiness as rd

        if hasattr(rd, "snapshot_at") and hasattr(rd, "forecast"):
            return rd
    except (ImportError, AttributeError):
        pass
    return None


def is_readiness_available() -> bool:
    """Return True if readiness module is available with snapshot_at and forecast."""
    return get_readiness_module() is not None


@st.cache_data
def get_snapshot(policy: str = "shikra_v2", seed: int = 1, day: int = 485) -> dict | None:
    """Return a deepcopied simulation snapshot for the specified day if available."""
    mod = get_readiness_module()
    if mod is None:
        return None
    try:
        return copy.deepcopy(mod.snapshot_at(policy, seed, day))
    except Exception:
        return None


@st.cache_data
def get_forecast(
    snapshot: dict,
    config: dict,
    params: dict,
    policy: str = "shikra",
    horizon_days: int = 3,
    n: int = 150,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Run Monte Carlo readiness forecast if module is available."""
    mod = get_readiness_module()
    if mod is None:
        return None
    try:
        return mod.forecast(
            snapshot=snapshot,
            config=config,
            params=params,
            policy=policy,
            horizon_days=horizon_days,
            n=n,
            seed=seed,
        )
    except Exception:
        return None


def load_readiness_calib_coverage() -> str | None:
    """Read honest 80% raw band coverage from results/readiness_calib.csv for D+3.

    Returns the value from the shikra_forecast row, or None if the CSV is missing.
    """
    calib_csv = RESULTS_DIR / "readiness_calib.csv"
    if calib_csv.exists():
        try:
            df = pd.read_csv(calib_csv)
            sub = df[(df["lead_days"] == 3) & (df["method"] == "shikra_forecast")]
            if not sub.empty:
                val = sub.iloc[0]["coverage_80"]
                try:
                    num = float(val)
                    return f"{num:.2f}" if abs(num - round(num, 2)) < 1e-4 else f"{num}"
                except (ValueError, TypeError):
                    return str(val)
        except Exception:
            pass
    return None

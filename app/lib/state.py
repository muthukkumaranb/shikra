"""Cached simulation state for the SHIKRA dashboard.

Runs the simulation once (in-memory only) and provides snapshot access
to the wing state on any given day.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"


@st.cache_resource
def get_sim_result(policy: str = "shikra_v2", seed: int = 1):
    """Run the simulation once (cached) and return the SimResult.

    Uses in-memory computation only, never writes to disk.
    """
    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul = pd.read_parquet(REPLAY_PATH)

    # Oracle policies use quality=0 (perfect prediction)
    if policy in ("oracle", "oracle_v2"):
        rul_use = apply(rul, 0)
    else:
        rul_use = apply(rul, 1)

    days = config["ops"]["days"]
    result = run(policy, config, rul_use, params, seed, days)
    return result


def get_daily(policy: str = "shikra_v2", seed: int = 1) -> pd.DataFrame:
    """Return the daily DataFrame from the cached sim result."""
    return get_sim_result(policy, seed).daily


def get_events(policy: str = "shikra_v2", seed: int = 1) -> pd.DataFrame:
    """Return the events DataFrame from the cached sim result."""
    return get_sim_result(policy, seed).events


def get_tail_mc(policy: str = "shikra_v2", seed: int = 1):
    """Return the tail MC array (days × tails) from the cached sim result."""
    return get_sim_result(policy, seed).tail_mc


def get_tail_status(policy: str = "shikra_v2", seed: int = 1):
    """Return the tail status list from the cached sim result."""
    return get_sim_result(policy, seed).tail_status


def get_kpis(policy: str = "shikra_v2", seed: int = 1) -> dict:
    """Return the KPIs dict from the cached sim result."""
    return get_sim_result(policy, seed).kpis


def get_wing_state_on_day(day: int, policy: str = "shikra_v2", seed: int = 1) -> dict:
    """Derive approximate wing state on a given day from daily/events DataFrames.

    Returns a dict with:
      - mc_count: number of MC aircraft
      - sqn_mc: list of MC counts per squadron (S1, S2, S3)
      - tail_statuses: list of (tail_id, squadron, status) tuples
    """
    result = get_sim_result(policy, seed)
    daily = result.daily
    tail_mc = result.tail_mc
    tail_status = result.tail_status

    if day < 0 or day >= len(daily):
        day = min(max(0, day), len(daily) - 1)

    mc_arr = tail_mc[day]  # bool array, len 54
    status_arr = tail_status[day]  # list of "MC"/"NMCM"/"NMCS"

    sqn_mc = [0, 0, 0]
    tail_statuses = []
    for tail in range(54):
        sqn = tail // 18
        sqn_label = f"S{sqn + 1}"
        tail_label = f"T{tail + 1:02d}"
        status = status_arr[tail]
        if mc_arr[tail]:
            sqn_mc[sqn] += 1
        tail_statuses.append({
            "tail": tail_label,
            "squadron": sqn_label,
            "status": status,
        })

    return {
        "day": day,
        "mc_count": int(mc_arr.sum()),
        "sqn_mc": sqn_mc,
        "tail_statuses": tail_statuses,
    }


def get_deterministic_injection_trajectory(r_star: int) -> tuple[str, int, int, pd.DataFrame]:
    """Deterministic injection rule for the Scenario walkthrough.

    Filter replay units from replay_rul.parquet where predictive q10 is between
    (R_star + 15) and (R_star + 30). Sort candidates by (subset, unit, cycle).
    Select the first matching unit and starting cycle deterministically.
    """
    from app.lib.data import load_replay_rul

    rul_df = load_replay_rul()
    candidates = rul_df[(rul_df["q10"] >= r_star + 15) & (rul_df["q10"] <= r_star + 30)]
    candidates_sorted = candidates.sort_values(["subset", "unit", "cycle"])
    first_row = candidates_sorted.iloc[0]
    inj_subset = str(first_row["subset"])
    inj_unit = int(first_row["unit"])
    inj_cycle = int(first_row["cycle"])

    unit_traj = (
        rul_df[
            (rul_df["subset"] == inj_subset)
            & (rul_df["unit"] == inj_unit)
            & (rul_df["cycle"] >= inj_cycle)
        ]
        .sort_values("cycle")
        .copy()
    )

    return inj_subset, inj_unit, inj_cycle, unit_traj


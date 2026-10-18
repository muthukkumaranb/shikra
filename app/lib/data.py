"""Cached data loaders for the SHIKRA dashboard.

Reads from results/*.csv, artifacts/, and config/wing.yaml.
All loaders are cached with @st.cache_data or @st.cache_resource.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ARTIFACTS_DIR / "policy_params.json"
REPLAY_PATH = ARTIFACTS_DIR / "replay_rul.parquet"
CONFORMAL_PATH = ARTIFACTS_DIR / "conformal.json"


@st.cache_data
def load_config() -> dict:
    """Load wing.yaml configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_params() -> dict:
    """Load policy_params.json (T_prev, R_star, miss_rate_at_R_star)."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_conformal() -> dict:
    """Load conformal.json calibration parameters."""
    with CONFORMAL_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_replay_rul() -> pd.DataFrame:
    """Load the replay RUL parquet file."""
    return pd.read_parquet(REPLAY_PATH)


@st.cache_data
def load_heatmap() -> pd.DataFrame:
    """Load results/heatmap.csv."""
    return pd.read_csv(RESULTS_DIR / "heatmap.csv")


@st.cache_data
def load_supply_shock() -> pd.DataFrame:
    """Load results/supply_shock.csv."""
    return pd.read_csv(RESULTS_DIR / "supply_shock.csv")


@st.cache_data
def load_policy_summary() -> pd.DataFrame:
    """Load results/policy_summary.csv."""
    return pd.read_csv(RESULTS_DIR / "policy_summary.csv")


@st.cache_data
def load_rul_cmapss() -> pd.DataFrame:
    """Load results/rul_cmapss.csv."""
    return pd.read_csv(RESULTS_DIR / "rul_cmapss.csv")


@st.cache_data
def load_rul_cmapss_multi() -> pd.DataFrame:
    """Load results/rul_cmapss_multi.csv."""
    return pd.read_csv(RESULTS_DIR / "rul_cmapss_multi.csv")


@st.cache_data
def load_coverage() -> pd.DataFrame:
    """Load results/coverage.csv."""
    return pd.read_csv(RESULTS_DIR / "coverage.csv")


@st.cache_data
def load_bias() -> pd.DataFrame:
    """Load results/bias.csv."""
    return pd.read_csv(RESULTS_DIR / "bias.csv")


@st.cache_data
def load_policies() -> pd.DataFrame:
    """Load results/policies.csv (raw per-seed simulation results)."""
    path = RESULTS_DIR / "policies.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_gates_md() -> str:
    """Load results/GATES.md as raw text."""
    path = RESULTS_DIR / "GATES.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "*GATES.md not found.*"


@st.cache_data
def load_spares_alloc() -> pd.DataFrame | None:
    """Load results/spares_alloc.csv if it exists (after T1.6 merge)."""
    path = RESULTS_DIR / "spares_alloc.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_spares_compare() -> pd.DataFrame | None:
    """Load results/spares_compare.csv if it exists (after T1.6 merge)."""
    path = RESULTS_DIR / "spares_compare.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def get_result_figures() -> list[Path]:
    """Return list of PNG figure paths in results/."""
    return sorted(RESULTS_DIR.glob("*.png"))

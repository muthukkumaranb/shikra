"""Multiprocessing runner for SHIKRA fleet simulation.

Runs 4 policies × 30 seeds in parallel and writes results to
``results/policies.csv``.

Run as a module:  ``python -m shikra.sim.runner``
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ARTIFACTS_DIR / "policy_params.json"
REPLAY_PATH = ARTIFACTS_DIR / "replay_rul.parquet"
POLICIES = ("reactive", "preventive", "shikra", "oracle", "shikra_v2", "oracle_v2")
N_SEEDS = 30


def _run_one(args: tuple) -> dict:
    """Worker function for one (policy, seed) replication."""
    policy, seed, config, params, days = args

    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    rul = pd.read_parquet(REPLAY_PATH)

    # Oracle uses perfect RUL predictions
    if policy in ("oracle", "oracle_v2"):
        rul_use = apply(rul, 0)
    else:
        rul_use = apply(rul, 1)

    result = run(policy, config, rul_use, params, seed, days)
    print(f"  [{policy}/{seed:02d}] serviceability={result.kpis['serviceability']:.3f}")
    return result.kpis


def run_all(n_seeds: int = N_SEEDS, n_jobs: int | None = None) -> pd.DataFrame:
    """Run all (policy, seed) combinations and return the KPI DataFrame."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)

    with PARAMS_PATH.open() as f:
        params = json.load(f)

    days = config["ops"]["days"]

    tasks = [
        (policy, seed, config, params, days)
        for policy in POLICIES
        for seed in range(1, n_seeds + 1)
    ]

    n_jobs = n_jobs or max(1, mp.cpu_count() - 1)
    print(f"Running {len(tasks)} simulations on {n_jobs} workers …")

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(_run_one, tasks)

    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "policies.csv"
    df.to_csv(out_path, index=False)
    print(f"Results written -> {out_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    run_all()

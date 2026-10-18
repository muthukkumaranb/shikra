"""Benchmark CP-SAT vs EDF hangar scheduling on operational wing simulation snapshots.

Evaluates days 485, 545, 605, 665 across seeds 1-5 for policy shikra_v2.
Writes paired results to results/cpsat_wing.csv and reports paired mean differences
with bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shikra.rul.variants import apply  # noqa: E402
from shikra.schedule.cpsat import edf_baseline, evaluate, solve  # noqa: E402
from shikra.schedule.from_snapshot import problem_from_snapshot  # noqa: E402

SNAPSHOT_DAYS = [485, 545, 605, 665]
SEEDS = [1, 2, 3, 4, 5]
POLICY = "shikra_v2"
OUTPUT_PATH = Path("results/cpsat_wing.csv")


def _get_snapshot_sim():
    """Import FleetSim supporting snapshot_days, dynamically loading if needed."""
    try:
        from shikra.strategy.readiness import snapshot_at  # noqa: F401

        return None
    except ImportError:
        pass

    # Dynamically load FleetSim from origin/t1/readiness-v2
    code = subprocess.check_output(
        ["git", "show", "origin/t1/readiness-v2:shikra/sim/engine.py"],
        text=True,
    )
    mod = types.ModuleType("shikra.sim.engine_snap")
    mod.__file__ = "engine_snap.py"
    sys.modules["shikra.sim.engine_snap"] = mod
    exec(code, mod.__dict__)
    return mod


def load_snapshots(
    policy: str,
    seeds: list[int],
    days: list[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Generate or retrieve fleet state snapshots for each (seed, day)."""
    sim_mod = _get_snapshot_sim()

    snapshots: dict[tuple[int, int], dict[str, Any]] = {}

    if sim_mod is None:
        from shikra.strategy.readiness import snapshot_at

        for s in seeds:
            for d in days:
                snapshots[(s, d)] = snapshot_at(policy, s, d)
        return snapshots

    # Load shared simulation configs
    root = Path(".")
    with open(root / "config" / "wing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(root / "artifacts" / "policy_params.json", encoding="utf-8") as f:
        params = json.load(f)
    rul = pd.read_parquet(root / "artifacts" / "replay_rul.parquet")

    max_day = max(days)
    for s in seeds:
        print(f"Simulating fleet for seed={s} through day {max_day}...")
        rul_variant = apply(rul, 1)
        sim_result = sim_mod.run(
            policy,
            cfg,
            rul_variant,
            params,
            seed=s,
            days=max_day + 1,
            snapshot_days=days,
        )
        for d in days:
            if sim_result.snapshots and d in sim_result.snapshots:
                snapshots[(s, d)] = sim_result.snapshots[d]
            else:
                raise RuntimeError(f"Snapshot missing for seed {s}, day {d}")

    return snapshots


def bootstrap_ci(
    data: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute bootstrap percentile confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(data))
    if len(data) == 0:
        return mean_val, mean_val, mean_val
    indices = rng.integers(0, len(data), size=(n_resamples, len(data)))
    sample_means = np.mean(data[indices], axis=1)
    low = float(np.percentile(sample_means, 100 * (alpha / 2)))
    high = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    return mean_val, low, high


def run_wing_benchmark() -> pd.DataFrame:
    """Run CP-SAT vs EDF benchmark across operational wing snapshots."""
    snapshots = load_snapshots(POLICY, SEEDS, SNAPSHOT_DAYS)

    records: list[dict[str, Any]] = []

    for d in SNAPSHOT_DAYS:
        for s in SEEDS:
            snap = snapshots[(s, d)]
            for sqn in (0, 1, 2):
                prob = problem_from_snapshot(snap, sqn=sqn)

                res_edf = edf_baseline(prob)
                eval_edf = evaluate(prob, res_edf)

                res_cpsat = solve(prob)
                eval_cpsat = evaluate(prob, res_cpsat)

                records.append(
                    {
                        "day": d,
                        "seed": s,
                        "sqn": sqn,
                        "n_jobs": len(prob.jobs),
                        "edf_wt": eval_edf["total_weighted_tardiness"],
                        "cpsat_wt": eval_cpsat["total_weighted_tardiness"],
                        "edf_makespan": eval_edf["makespan"],
                        "cpsat_makespan": eval_cpsat["makespan"],
                    }
                )

    df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved real wing benchmark results to {OUTPUT_PATH.resolve()}")

    # Print summary & statistical report
    delta_wt = (df["edf_wt"] - df["cpsat_wt"]).to_numpy()
    delta_ms = (df["edf_makespan"] - df["cpsat_makespan"]).to_numpy()

    mean_wt_diff, low_wt, high_wt = bootstrap_ci(delta_wt, n_resamples=2000, seed=0)
    mean_ms_diff, low_ms, high_ms = bootstrap_ci(delta_ms, n_resamples=2000, seed=0)

    cpsat_wt_wins = int(np.sum(delta_wt > 0))
    ties = int(np.sum((delta_wt == 0) & (delta_ms == 0)))
    cpsat_ms_wins = int(np.sum((delta_wt == 0) & (delta_ms > 0)))
    edf_wins = int(np.sum(delta_wt < 0))

    print("\n" + "=" * 70)
    print("CP-SAT vs EDF REAL WING OPERATIONAL BENCHMARK (shikra_v2)")
    print("=" * 70)
    print(f"Snapshots evaluated : {len(df)} (4 days x 5 seeds x 3 sqns)")
    print(f"Total bay jobs      : {df['n_jobs'].sum()} (mean {df['n_jobs'].mean():.2f}/sqn)")
    print("-" * 70)
    print("Outcome Distribution:")
    print(f"  CP-SAT Wins (Weighted Tardiness) : {cpsat_wt_wins:2d} / {len(df)} ({cpsat_wt_wins / len(df):.1%})")
    print(f"  CP-SAT Wins (Makespan Tiebreak)  : {cpsat_ms_wins:2d} / {len(df)} ({cpsat_ms_wins / len(df):.1%})")
    print(f"  Ties (Exact match)               : {ties:2d} / {len(df)} ({ties / len(df):.1%})")
    print(f"  EDF Wins                         : {edf_wins:2d} / {len(df)} ({edf_wins / len(df):.1%})")
    print("-" * 70)
    print("Paired Bootstrap 95% Confidence Intervals (2,000 resamples, seed 0):")
    print(f"  Delta Weighted Tardiness (EDF - CP-SAT): {mean_wt_diff:+.4f} [95% CI: {low_wt:+.4f}, {high_wt:+.4f}]")
    print(f"  Delta Makespan           (EDF - CP-SAT): {mean_ms_diff:+.4f} [95% CI: {low_ms:+.4f}, {high_ms:+.4f}]")
    print("-" * 70)
    print("Mean Performance Metrics:")
    print(f"  Mean Weighted Tardiness: EDF = {df['edf_wt'].mean():.4f} | CP-SAT = {df['cpsat_wt'].mean():.4f}")
    print(f"  Mean Makespan (days)   : EDF = {df['edf_makespan'].mean():.4f} | CP-SAT = {df['cpsat_makespan'].mean():.4f}")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    run_wing_benchmark()

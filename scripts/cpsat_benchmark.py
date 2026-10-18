"""Benchmark CP-SAT vs Earliest-Due-First (EDF) hangar bay scheduling.

Generates 200 synthetic scheduling instances with fixed seed 2026, executes both
CP-SAT (exact/optimizer) and EDF greedy list baseline, validates schedules, computes
paired differences, and calculates 95% bootstrap confidence intervals.

Outputs results to results/cpsat_benchmark.csv and prints a clear statistical summary.
Reports win/tie/loss distribution honestly.

All scheduling models and synthetic wing assumptions represent a synthetic wing.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.schedule.cpsat import (  # noqa: E402
    Job,
    Problem,
    edf_baseline,
    evaluate,
    solve,
)

RESULTS_DIR = ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "cpsat_benchmark.csv"

N_INSTANCES = 200
SEED = 2026
N_BOOT = 2000
BOOT_SEED = 0


def bootstrap_ci(
    data: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using percentile bootstrap."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def generate_benchmark_instance(instance_id: int, rng: random.Random) -> Problem:
    """Generate a diverse synthetic hangar scheduling problem instance."""
    # Varying problem sizes: 4 to 16 jobs
    n_jobs = rng.randint(4, 16)

    # Capacity: 1, 2, or 3 bays (typical squadron/wing hangar bay allocations)
    capacity = rng.choice([1, 2, 3])

    weights_pool = [1.0, 1.0, 2.0, 5.0, 10.0]
    jobs: list[Job] = []

    for j_idx in range(n_jobs):
        job_id = f"inst{instance_id:03d}_j{j_idx:02d}"
        tail = (instance_id * 100 + j_idx) % 54

        # Durations: 1 to 6 days
        duration = rng.randint(1, 6)

        # Release dates: staggered arrival (0 to 10 days)
        release_date = rng.randint(0, 10)

        # Due date slack: tight (0-3 days), moderate (4-8 days), loose (9-15 days)
        slack = rng.choice([rng.randint(0, 2), rng.randint(2, 6), rng.randint(6, 12)])
        due_date = release_date + duration + slack

        weight = rng.choice(weights_pool)
        sqn = rng.choice([1, 2, 3])

        jobs.append(
            Job(
                job_id=job_id,
                tail=tail,
                duration=duration,
                release_date=release_date,
                due_date=due_date,
                weight=weight,
                sqn=sqn,
                job_type="engine_change",
            )
        )

    # Ample horizon to allow all feasible schedules
    max_span = max(j.release_date + j.duration for j in jobs)
    total_dur = sum(j.duration for j in jobs)
    horizon = max_span + total_dur + 20

    return Problem(
        name=f"benchmark_inst_{instance_id:03d}",
        jobs=jobs,
        capacity=capacity,
        horizon=horizon,
    )


def run_benchmark(
    n_instances: int = N_INSTANCES, seed: int = SEED
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run CP-SAT vs EDF benchmark across generated instances."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    print(f"Running benchmark across {n_instances} instances (seed={seed})...")
    t0_all = time.perf_counter()
    total_edf_time = 0.0
    total_cpsat_time = 0.0

    for i in range(n_instances):
        problem = generate_benchmark_instance(i, rng)

        # Solve with EDF baseline
        t0_edf = time.perf_counter()
        edf_res = edf_baseline(problem)
        total_edf_time += time.perf_counter() - t0_edf

        # Validate EDF schedule
        eval_edf = evaluate(problem, edf_res)
        assert eval_edf["is_valid"], f"Instance {i}: EDF produced invalid schedule: {eval_edf['errors']}"

        # Solve with CP-SAT / optimizer
        t0_cpsat = time.perf_counter()
        cpsat_res = solve(problem, max_deterministic_time=5.0, deterministic=True, fallback_to_baseline=True)
        total_cpsat_time += time.perf_counter() - t0_cpsat

        # Validate CP-SAT schedule
        eval_cpsat = evaluate(problem, cpsat_res)
        assert eval_cpsat["is_valid"], f"Instance {i}: CP-SAT produced invalid schedule: {eval_cpsat['errors']}"

        # Compare metrics
        # Positive delta means CP-SAT achieved lower (better) objective
        wt_diff = edf_res.total_weighted_tardiness - cpsat_res.total_weighted_tardiness
        ms_diff = edf_res.makespan - cpsat_res.makespan
        tt_diff = edf_res.total_tardiness - cpsat_res.total_tardiness

        # Categorize winner
        # Primary objective: weighted tardiness. Secondary: makespan.
        if wt_diff > 1e-6:
            outcome = "CPSAT_WIN"
        elif wt_diff < -1e-6:
            outcome = "EDF_WIN"
        else:
            # Weighted tardiness tied
            if ms_diff > 0:
                outcome = "CPSAT_WIN_MAKESPAN"
            elif ms_diff < 0:
                outcome = "EDF_WIN_MAKESPAN"
            else:
                outcome = "TIE"

        rows.append({
            "instance_id": i,
            "n_jobs": len(problem.jobs),
            "capacity": problem.capacity,
            "horizon": problem.horizon,
            "cpsat_status": cpsat_res.status,
            "edf_weighted_tardiness": round(edf_res.total_weighted_tardiness, 4),
            "cpsat_weighted_tardiness": round(cpsat_res.total_weighted_tardiness, 4),
            "delta_weighted_tardiness": round(wt_diff, 4),
            "edf_makespan": edf_res.makespan,
            "cpsat_makespan": cpsat_res.makespan,
            "delta_makespan": ms_diff,
            "edf_total_tardiness": edf_res.total_tardiness,
            "cpsat_total_tardiness": cpsat_res.total_tardiness,
            "delta_total_tardiness": tt_diff,
            "edf_num_tardy": edf_res.num_tardy_jobs,
            "cpsat_num_tardy": cpsat_res.num_tardy_jobs,
            "outcome": outcome,
        })

    elapsed_all = time.perf_counter() - t0_all
    df = pd.DataFrame(rows)

    # Paired bootstrap calculations
    wt_diffs = df["delta_weighted_tardiness"].values
    ms_diffs = df["delta_makespan"].values

    wt_mean, wt_lo, wt_hi = bootstrap_ci(wt_diffs, n_boot=N_BOOT, seed=BOOT_SEED)
    ms_mean, ms_lo, ms_hi = bootstrap_ci(ms_diffs, n_boot=N_BOOT, seed=BOOT_SEED)

    cpsat_wins = int((df["outcome"].str.startswith("CPSAT")).sum())
    edf_wins = int((df["outcome"].str.startswith("EDF")).sum())
    ties = int((df["outcome"] == "TIE").sum())

    summary = {
        "n_instances": n_instances,
        "seed": seed,
        "elapsed_total_s": round(elapsed_all, 2),
        "cpsat_wins": cpsat_wins,
        "cpsat_win_rate_pct": round(100.0 * cpsat_wins / n_instances, 1),
        "ties": ties,
        "tie_rate_pct": round(100.0 * ties / n_instances, 1),
        "edf_wins": edf_wins,
        "edf_win_rate_pct": round(100.0 * edf_wins / n_instances, 1),
        "delta_weighted_tardiness_mean": round(wt_mean, 4),
        "delta_weighted_tardiness_ci95": [round(wt_lo, 4), round(wt_hi, 4)],
        "delta_makespan_mean": round(ms_mean, 4),
        "delta_makespan_ci95": [round(ms_lo, 4), round(ms_hi, 4)],
        "edf_weighted_tardiness_mean": round(float(df["edf_weighted_tardiness"].mean()), 4),
        "cpsat_weighted_tardiness_mean": round(float(df["cpsat_weighted_tardiness"].mean()), 4),
        "edf_makespan_mean": round(float(df["edf_makespan"].mean()), 4),
        "cpsat_makespan_mean": round(float(df["cpsat_makespan"].mean()), 4),
        "mean_edf_time_s": round(total_edf_time / n_instances, 6),
        "mean_cpsat_time_s": round(total_cpsat_time / n_instances, 6),
    }

    return df, summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, summary = run_benchmark(N_INSTANCES, SEED)

    # Save benchmark CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved benchmark results to {OUTPUT_CSV}")

    # Honest statistical report
    print("\n" + "=" * 70)
    print("CP-SAT vs EARLIEST-DUE-FIRST (EDF) HANGAR SCHEDULING BENCHMARK")
    print("=" * 70)
    print(f"Instances evaluated : {summary['n_instances']} (seed={summary['seed']})")
    print(f"Total benchmark time: {summary['elapsed_total_s']}s")
    print("-" * 70)
    print("Outcome Distribution:")
    print(f"  CP-SAT Wins       : {summary['cpsat_wins']} / {summary['n_instances']} ({summary['cpsat_win_rate_pct']}%)")
    print(f"  Ties              : {summary['ties']} / {summary['n_instances']} ({summary['tie_rate_pct']}%)")
    print(f"  EDF Wins          : {summary['edf_wins']} / {summary['n_instances']} ({summary['edf_win_rate_pct']}%)")
    print("-" * 70)
    print("Paired Bootstrap 95% Confidence Intervals (2,000 resamples, seed 0):")
    print(
        f"  Delta Weighted Tardiness (EDF - CP-SAT): {summary['delta_weighted_tardiness_mean']:.4f} "
        f"[95% CI: {summary['delta_weighted_tardiness_ci95'][0]:.4f}, {summary['delta_weighted_tardiness_ci95'][1]:.4f}]"
    )
    print(
        f"  Delta Makespan           (EDF - CP-SAT): {summary['delta_makespan_mean']:.4f} "
        f"[95% CI: {summary['delta_makespan_ci95'][0]:.4f}, {summary['delta_makespan_ci95'][1]:.4f}]"
    )
    print("-" * 70)
    print("Mean Performance Metrics:")
    print(f"  Mean Weighted Tardiness: EDF = {summary['edf_weighted_tardiness_mean']:.4f} | CP-SAT = {summary['cpsat_weighted_tardiness_mean']:.4f}")
    print(f"  Mean Makespan (days)   : EDF = {summary['edf_makespan_mean']:.4f} | CP-SAT = {summary['cpsat_makespan_mean']:.4f}")
    print(f"  Mean Solve Time (s)    : EDF = {summary['mean_edf_time_s']:.6f}s | CP-SAT = {summary['mean_cpsat_time_s']:.6f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""A10 Readiness Forecast Calibration & Reliability Evaluation.

Contract §6, T1.7 & Step B4:
- Runs policy shikra_v2 at pipeline_mult = 1.7, 30 seeds (1..30).
- Captures snapshots at D in {395, 425, ..., 695} (11 days * 30 seeds = 330 snapshots).
- Evaluates 7-day Monte Carlo forecast (n=200, seed=run_seed) against realised MC counts
  at lead days 3 and 7.
- Evaluates against two naive baselines:
    1. Persistence: degenerate point mass at current MC(D).
    2. Climatology: leave-one-seed-out pooled historical distribution.
- Metrics:
    - 80% coverage (target ~0.80)
    - MAE of expected MC
    - Brier score of P(>=N MC) averaged over N in {6..18}
- Outputs:
    - results/readiness_calib.csv
    - results/fig_readiness_reliability.png
    - results/readiness_example.csv (seed 0, D = 180)
    - results/readiness_actions_example.csv (seed 0, D = 180, S2, lead 3)

All parameters and wings are synthetic.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.rul.variants import apply  # noqa: E402
from shikra.sim.engine import AC_PER_SQN, N_SQN, run  # noqa: E402
from shikra.strategy.readiness import forecast, rank_actions  # noqa: E402

CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"

SNAPSHOT_DAYS = list(range(395, 696, 30))
LEAD_DAYS = [3, 7]
MC_THRESHOLDS = list(range(6, 19))  # N in 6..18


def _run_seed_simulation(args: tuple) -> tuple[int, dict, list[list[int]]]:
    """Run one simulation replication and return (seed, snapshots_dict, daily_sqn_mc)."""
    seed, cfg_dict, params_dict, days = args
    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    res = run("shikra_v2", cfg_dict, rul_shikra, params_dict, seed=seed, days=days, snapshot_days=SNAPSHOT_DAYS)

    # Precompute daily realised MC per squadron: shape (days, N_SQN)
    daily_sqn_mc = []
    for day_statuses in res.tail_status:
        sqn_counts = []
        for s in range(N_SQN):
            s_start = s * AC_PER_SQN
            s_end = s_start + AC_PER_SQN
            cnt = sum(1 for t in range(s_start, s_end) if day_statuses[t] == "MC")
            sqn_counts.append(cnt)
        daily_sqn_mc.append(sqn_counts)

    return seed, res.snapshots, daily_sqn_mc


def _forecast_worker(args: tuple) -> tuple[int, int, pd.DataFrame]:
    """Compute forecast for one snapshot."""
    seed, snap_day, snap_dict, cfg_dict, params_dict, n_mc = args
    fc_df = forecast(snap_dict, cfg_dict, params_dict, policy="shikra_v2", horizon_days=7, n=n_mc, seed=seed)
    return seed, snap_day, fc_df


def main() -> None:
    parser = argparse.ArgumentParser(description="A10 readiness forecast calibration")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds (default 30, range(0..seeds-1))")
    parser.add_argument("--n-mc", type=int, default=200, help="Monte Carlo samples per forecast (default 200)")
    parser.add_argument("--jobs", type=int, default=None, help="Parallel worker processes")
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline_mult"] = 1.7
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    days = cfg["ops"]["days"]
    seeds = list(range(1, args.seeds + 1))
    n_workers = args.jobs or max(1, mp.cpu_count() - 1)

    print(f"Step 1: Running {len(seeds)} fleet simulations (policy=shikra_v2, pipeline_mult=1.7, seeds={seeds[0]}..{seeds[-1]})...")
    sim_tasks = [(s, cfg, params, days) for s in seeds]
    with mp.Pool(processes=n_workers) as pool:
        sim_outputs = pool.map(_run_seed_simulation, sim_tasks)

    # Index simulation outputs by seed
    all_snapshots: dict[tuple[int, int], dict] = {}
    all_realised_mc: dict[int, list[list[int]]] = {}
    for s, snaps, daily_mc in sim_outputs:
        all_realised_mc[s] = daily_mc
        for d in SNAPSHOT_DAYS:
            all_snapshots[(s, d)] = snaps[d]

    print(f"Captured {len(all_snapshots)} snapshots across {len(seeds)} seeds.")

    # Step 2: Forecast for each snapshot
    print(f"Step 2: Computing Monte Carlo forecasts (n={args.n_mc}) on {n_workers} workers...")
    fc_tasks = [
        (s, d, all_snapshots[(s, d)], cfg, params, args.n_mc)
        for s in seeds
        for d in SNAPSHOT_DAYS
    ]
    with mp.Pool(processes=n_workers) as pool:
        fc_results = pool.map(_forecast_worker, fc_tasks)

    forecast_lookup: dict[tuple[int, int], pd.DataFrame] = {}
    for s, d, df_fc in fc_results:
        forecast_lookup[(s, d)] = df_fc

    # Step 3: Compute leave-one-seed-out climatology pool per seed and squadron
    climatology_pools: dict[tuple[int, int], np.ndarray] = {}
    for s_eval in seeds:
        other_seeds = [s for s in seeds if s != s_eval]
        for sqn_idx in range(N_SQN):
            pooled = []
            for os in other_seeds:
                for day_idx in range(365, days):
                    pooled.append(all_realised_mc[os][day_idx][sqn_idx])
            climatology_pools[(s_eval, sqn_idx)] = np.array(pooled)

    # Step 4: Evaluate metrics for each lead day and method
    print("Step 3: Scoring forecasts and baselines (coverage, MAE, Brier)...")

    # Collectors for reliability diagram (lead 3)
    reliability_pairs: list[tuple[float, int]] = []

    # Storage for scoring rows: method -> lead -> list of metric dicts
    eval_records: list[dict[str, Any]] = []

    for lead in LEAD_DAYS:
        for s in seeds:
            for d in SNAPSHOT_DAYS:
                target_day = d + lead
                if target_day >= days:
                    continue

                fc_df = forecast_lookup[(s, d)]
                fc_lead = fc_df[fc_df["lead_days"] == lead]

                for sqn_idx in range(N_SQN):
                    sqn_num = sqn_idx + 1
                    y_realised = all_realised_mc[s][target_day][sqn_idx]
                    y_curr = all_realised_mc[s][d][sqn_idx]

                    # 1. SHIKRA Forecast
                    row_fc = fc_lead[fc_lead["sqn"] == sqn_num].iloc[0]
                    exp_mc = float(row_fc["expected_mc"])
                    q10 = float(row_fc["mc_q10"])
                    q90 = float(row_fc["mc_q90"])

                    cov_fc = 1.0 if (q10 <= y_realised <= q90) else 0.0
                    mae_fc = abs(y_realised - exp_mc)
                    brier_fc = np.mean([
                        (float(row_fc[f"p_ge_{n}"]) - (1.0 if y_realised >= n else 0.0)) ** 2
                        for n in MC_THRESHOLDS
                    ])

                    if lead == 3:
                        for n in MC_THRESHOLDS:
                            reliability_pairs.append((
                                float(row_fc[f"p_ge_{n}"]),
                                1 if y_realised >= n else 0,
                            ))

                    eval_records.append({
                        "lead_days": lead,
                        "method": "shikra_forecast",
                        "coverage_80": cov_fc,
                        "mae": mae_fc,
                        "brier": float(brier_fc),
                    })

                    # 2. Persistence Baseline (MC(D+k) = MC(D))
                    cov_pers = 1.0 if y_realised == y_curr else 0.0
                    mae_pers = abs(y_realised - y_curr)
                    brier_pers = np.mean([
                        ((1.0 if y_curr >= n else 0.0) - (1.0 if y_realised >= n else 0.0)) ** 2
                        for n in MC_THRESHOLDS
                    ])

                    eval_records.append({
                        "lead_days": lead,
                        "method": "persistence",
                        "coverage_80": cov_pers,
                        "mae": mae_pers,
                        "brier": float(brier_pers),
                    })

                    # 3. Climatology Baseline (Leave-one-seed-out pooled)
                    clim_vals = climatology_pools[(s, sqn_idx)]
                    clim_exp = float(np.mean(clim_vals))
                    clim_q10 = float(np.percentile(clim_vals, 10))
                    clim_q90 = float(np.percentile(clim_vals, 90))

                    cov_clim = 1.0 if (clim_q10 <= y_realised <= clim_q90) else 0.0
                    mae_clim = abs(y_realised - clim_exp)
                    brier_clim = np.mean([
                        (float(np.mean(clim_vals >= n)) - (1.0 if y_realised >= n else 0.0)) ** 2
                        for n in MC_THRESHOLDS
                    ])

                    eval_records.append({
                        "lead_days": lead,
                        "method": "climatology",
                        "coverage_80": cov_clim,
                        "mae": mae_clim,
                        "brier": float(brier_clim),
                    })

    df_eval = pd.DataFrame(eval_records)

    calib_summary_rows = []
    n_total_snaps = len(seeds) * len(SNAPSHOT_DAYS)

    for lead in LEAD_DAYS:
        for method in ["shikra_forecast", "persistence", "climatology"]:
            sub = df_eval[(df_eval["lead_days"] == lead) & (df_eval["method"] == method)]
            calib_summary_rows.append({
                "lead_days": lead,
                "method": method,
                "coverage_80": round(float(sub["coverage_80"].mean()), 4),
                "mae": round(float(sub["mae"].mean()), 4),
                "brier": round(float(sub["brier"].mean()), 4),
                "n_snapshots": n_total_snaps,
            })

    calib_df = pd.DataFrame(calib_summary_rows)
    calib_csv_path = RESULTS_DIR / "readiness_calib.csv"
    calib_df.to_csv(calib_csv_path, index=False)
    print(f"Wrote calibration summary -> {calib_csv_path}")
    print(calib_df.to_string(index=False))

    # Step 5: Generate illustrative example outputs (seed 1, D = 485)
    print("\nStep 4: Writing illustrative example outputs for seed 1, day 485...")
    ex_seed = 1
    ex_day = 485
    ex_snap = all_snapshots[(ex_seed, ex_day)]
    ex_fc_df = forecast(ex_snap, cfg, params, policy="shikra_v2", horizon_days=7, n=200, seed=ex_seed)

    ex_fc_df.insert(0, "snapshot_day", ex_day)
    ex_fc_df.insert(0, "seed", ex_seed)
    ex_csv_path = RESULTS_DIR / "readiness_example.csv"
    ex_fc_df.to_csv(ex_csv_path, index=False)
    print(f"Wrote example forecast -> {ex_csv_path}")

    # Action ranking for Squadron 2, lead day 3
    actions_df = rank_actions(ex_snap, cfg, params, sqn=2, lead_days=7, n=200, seed=ex_seed)
    actions_csv_path = RESULTS_DIR / "readiness_actions_example.csv"
    actions_df.to_csv(actions_csv_path, index=False)
    print(f"Wrote example actions -> {actions_csv_path}")

    # Step 6: Generate reliability diagram (fig_readiness_reliability.png)
    print("Step 5: Generating reliability diagram...")
    p_preds = np.array([p for p, o in reliability_pairs])
    y_obs = np.array([o for p, o in reliability_pairs])

    bins = np.linspace(0.0, 1.0, 11)  # 10 deciles
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    emp_freqs = []
    bin_counts = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_preds >= lo) & (p_preds <= hi if i == len(bins) - 2 else p_preds < hi)
        if np.sum(mask) > 0:
            emp_freqs.append(float(np.mean(y_obs[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            emp_freqs.append(bin_centers[i])
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    ax1.plot(
        bin_centers,
        emp_freqs,
        marker="o",
        color="#1f77b4",
        linewidth=2,
        label="SHIKRA readiness forecast (lead 3)",
    )
    ax1.set_ylabel("Observed Frequency", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(
        "Reliability Diagram: P(≥N MC) Readiness Forecast (Lead 3)\nsynthetic wing, 330 snapshots",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    # Histogram of predicted probabilities
    ax2.bar(bin_centers, bin_counts, width=0.08, color="#aec7e8", edgecolor="#1f77b4", alpha=0.85)
    ax2.set_xlabel("Predicted Probability Decile", fontsize=11)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = RESULTS_DIR / "fig_readiness_reliability.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close()
    print(f"Saved reliability diagram -> {fig_path}")

    elapsed = time.time() - t_start
    print(f"Readiness calibration completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")


if __name__ == "__main__":
    main()

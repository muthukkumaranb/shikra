"""Systematic-bias robustness evaluation (contract §3.2b).

Evaluates SHIKRA v2 under additive bias in RUL quantile predictions:
  bias_cycles ∈ {-20, -10, 0, +10, +20}
at pipeline_mult = 1.0, 30 seeds, Common Random Numbers (CRN), fixed R_star.

Outputs:
  - results/bias.csv
  - results/fig_bias.png

Usage:
  python scripts/bias.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"

BIAS_LEVELS = [-20, -10, 0, 10, 20]
N_SEEDS = 30
N_BOOT = 2000
BOOT_SEED = 42


def bootstrap_ci(
    data: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _run_bias_worker(args: tuple) -> dict:
    """Run single seed under given bias."""
    bias, seed, config_dict, params_dict, days = args

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shikra.rul.variants import apply_bias
    from shikra.sim.engine import run

    cfg = dict(config_dict)
    cfg["pipeline_mult"] = 1.0

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_biased = apply_bias(rul_raw, bias)

    res = run("shikra_v2", cfg, rul_biased, params_dict, seed, days)
    kpis = res.kpis
    return {
        "bias_cycles": bias,
        "seed": seed,
        "serviceability": kpis["serviceability"],
        "unscheduled_removals": kpis["unscheduled_removals"],
        "planned_removals": kpis["planned_removals"],
        "wasted_life_cycles": kpis["wasted_life_cycles"],
        "aog_days": kpis["aog_days"],
        "nmcm_pct": kpis["nmcm_pct"],
        "nmcs_pct": kpis["nmcs_pct"],
        "sorties_flown": kpis["sorties_flown"],
    }


def run_bias_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all bias experiments across seeds and compute summaries."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    days = config["ops"]["days"]

    tasks = [
        (bias, seed, config, params, days)
        for bias in BIAS_LEVELS
        for seed in range(1, N_SEEDS + 1)
    ]

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Running bias sensitivity experiments ({len(tasks)} runs) on {n_workers} workers …")

    with mp.Pool(processes=n_workers) as pool:
        records = pool.map(_run_bias_worker, tasks)

    raw_df = pd.DataFrame(records)

    summary_rows = []
    for bias in BIAS_LEVELS:
        b_sub = raw_df[raw_df["bias_cycles"] == bias]
        s_mean, s_lo, s_hi = bootstrap_ci(b_sub["serviceability"].values)
        u_mean, u_lo, u_hi = bootstrap_ci(b_sub["unscheduled_removals"].values)
        p_mean, p_lo, p_hi = bootstrap_ci(b_sub["planned_removals"].values)
        w_mean, w_lo, w_hi = bootstrap_ci(b_sub["wasted_life_cycles"].values)
        a_mean, a_lo, a_hi = bootstrap_ci(b_sub["aog_days"].values)

        summary_rows.append(
            {
                "bias_cycles": bias,
                "serviceability_mean": s_mean,
                "serviceability_ci_lo": s_lo,
                "serviceability_ci_hi": s_hi,
                "unscheduled_removals_mean": u_mean,
                "unscheduled_removals_ci_lo": u_lo,
                "unscheduled_removals_ci_hi": u_hi,
                "planned_removals_mean": p_mean,
                "planned_removals_ci_lo": p_lo,
                "planned_removals_ci_hi": p_hi,
                "wasted_life_cycles_mean": w_mean,
                "wasted_life_cycles_ci_lo": w_lo,
                "wasted_life_cycles_ci_hi": w_hi,
                "aog_days_mean": a_mean,
                "aog_days_ci_lo": a_lo,
                "aog_days_ci_hi": a_hi,
                "nmcm_pct_mean": float(b_sub["nmcm_pct"].mean()),
                "nmcs_pct_mean": float(b_sub["nmcs_pct"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return raw_df, summary_df


def plot_bias_results(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot publication-quality systematic bias sensitivity figure."""
    fig, (ax_svc, ax_rem) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    biases = summary_df["bias_cycles"].values
    s_mean = summary_df["serviceability_mean"].values * 100
    s_lo = summary_df["serviceability_ci_lo"].values * 100
    s_hi = summary_df["serviceability_ci_hi"].values * 100

    # Panel A: Serviceability
    ax_svc.plot(biases, s_mean, color="royalblue", marker="o", linewidth=2.5, label="SHIKRA v2 (pm=1.0)")
    ax_svc.fill_between(biases, s_lo, s_hi, color="royalblue", alpha=0.2, label="95% CI")
    ax_svc.axvline(0, color="gray", linestyle=":", alpha=0.8, label="Zero Bias (Nominal)")
    ax_svc.axhline(75.0, color="darkorange", linestyle="--", linewidth=1.5, label="75% IAF norm (CAG)")

    ax_svc.set_xlabel("Systematic RUL Prediction Bias (cycles)", fontsize=11, fontweight="bold")
    ax_svc.set_ylabel("Serviceability (%)", fontsize=11, fontweight="bold")
    ax_svc.set_title("(A) Serviceability vs Prediction Bias", fontsize=12, fontweight="bold")
    ax_svc.set_xticks(biases)
    ax_svc.set_xticklabels([f"{b:+d}" for b in biases], fontsize=10)
    ax_svc.grid(True, linestyle=":", alpha=0.6)
    ax_svc.legend(fontsize=9, loc="lower left")

    # Panel B: Removals & Wasted Life
    u_mean = summary_df["unscheduled_removals_mean"].values
    u_lo = summary_df["unscheduled_removals_ci_lo"].values
    u_hi = summary_df["unscheduled_removals_ci_hi"].values

    p_mean = summary_df["planned_removals_mean"].values
    p_lo = summary_df["planned_removals_ci_lo"].values
    p_hi = summary_df["planned_removals_ci_hi"].values

    ax_rem.plot(biases, p_mean, color="forestgreen", marker="^", linewidth=2, label="Planned Removals")
    ax_rem.fill_between(biases, p_lo, p_hi, color="forestgreen", alpha=0.15)

    ax_rem.plot(biases, u_mean, color="crimson", marker="s", linewidth=2, label="Unscheduled Removals")
    ax_rem.fill_between(biases, u_lo, u_hi, color="crimson", alpha=0.15)
    ax_rem.axvline(0, color="gray", linestyle=":", alpha=0.8)

    ax_rem.set_xlabel("Systematic RUL Prediction Bias (cycles)", fontsize=11, fontweight="bold")
    ax_rem.set_ylabel("Annual Removals per Fleet", fontsize=11, fontweight="bold")
    ax_rem.set_title("(B) Removal Trade-off vs Prediction Bias", fontsize=12, fontweight="bold")
    ax_rem.set_xticks(biases)
    ax_rem.set_xticklabels([f"{b:+d}" for b in biases], fontsize=10)
    ax_rem.grid(True, linestyle=":", alpha=0.6)
    ax_rem.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Robustness to Systematic Prediction Bias (§3.2b, 30 seeds per point, CRN)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "synthetic wing, pipeline_mult=1.0, fixed R*",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#333333",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bias robustness figure saved -> {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "bias.csv"
    fig_path = RESULTS_DIR / "fig_bias.png"

    _, summary_df = run_bias_experiments()
    summary_df.to_csv(csv_path, index=False)
    print(f"Bias summary saved -> {csv_path}")

    plot_bias_results(summary_df, fig_path)


if __name__ == "__main__":
    main()

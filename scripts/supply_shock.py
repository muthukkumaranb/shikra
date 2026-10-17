"""A12 Supply Shock & Indigenisation Ranking (contract §6).

Evaluates the vulnerability of the fleet to foreign supply halts:
- Policy: shikra_v2, pipeline_mult = 1.0
- At day 90, halt foreign new buys and foreign repairs for 180 days (days 90..269)
- Evaluated for each foreign LRU singly and all foreign LRUs together
- 30 seeds with Common Random Numbers (CRN) vs no-shock baseline

Outputs:
  - results/supply_shock.csv
  - results/fig_supply_shock.png

Usage:
  python scripts/supply_shock.py
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

N_SEEDS = 30
N_BOOT = 2000
BOOT_SEED = 42
SHOCK_START = 90
SHOCK_DURATION = 180
SHOCK_END = SHOCK_START + SHOCK_DURATION


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


def _run_shock_seed_worker(args: tuple) -> dict:
    """Run baseline and shocked simulation for a given (scenario, seed)."""
    scenario, shock_lrus, seed, config_dict, params_dict, days = args

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    cfg = dict(config_dict)
    cfg["pipeline_mult"] = 1.0

    # 1. Baseline run (no shock)
    res_base = run("shikra_v2", cfg, rul_shikra, params_dict, seed, days)
    base_svc = res_base.kpis["serviceability"]

    # 2. Shock run
    shock_cfg = dict(cfg)
    shock_cfg["shock"] = {
        "start": SHOCK_START,
        "duration": SHOCK_DURATION,
        "lrus": shock_lrus,
    }
    res_shock = run("shikra_v2", shock_cfg, rul_shikra, params_dict, seed, days)
    shock_svc = res_shock.kpis["serviceability"]

    loss_pp = (base_svc - shock_svc) * 100

    # 3. Compute recovery days: days after SHOCK_END (day 270) until shock backlog clears
    rec_days = 0
    for d in range(SHOCK_END, days):
        miss_s = sum(res_shock.daily.loc[d, "lru_missing"][lid] for lid in shock_lrus)
        miss_b = sum(res_base.daily.loc[d, "lru_missing"][lid] for lid in shock_lrus)
        mc_s = res_shock.daily.loc[d, "mc"]
        mc_b = res_base.daily.loc[d, "mc"]
        if miss_s > miss_b or mc_s < mc_b:
            rec_days = max(rec_days, d - SHOCK_END + 1)

    return {
        "scenario": scenario,
        "seed": seed,
        "loss_pp": loss_pp,
        "recovery_days": rec_days,
        "base_svc": base_svc,
        "shock_svc": shock_svc,
    }


def run_supply_shock_experiments() -> pd.DataFrame:
    """Execute supply shock scenarios across all foreign LRUs and seeds."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    days = config["ops"]["days"]
    foreign_lrus = [lru["id"] for lru in config["lrus"] if lru.get("source") == "foreign"]

    # Scenarios: all foreign together first, then each singly
    scenarios = ["all_foreign"] + foreign_lrus

    tasks = []
    for sc in scenarios:
        shock_lrus = foreign_lrus if sc == "all_foreign" else [sc]
        for seed in range(1, N_SEEDS + 1):
            tasks.append((sc, shock_lrus, seed, config, params, days))

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Running supply shock experiments ({len(tasks)} runs) on {n_workers} workers …")

    with mp.Pool(processes=n_workers) as pool:
        records = pool.map(_run_shock_seed_worker, tasks)

    df_raw = pd.DataFrame(records)

    summary_rows = []
    for sc in scenarios:
        sub = df_raw[df_raw["scenario"] == sc]
        loss_vals = sub["loss_pp"].values
        rec_vals = sub["recovery_days"].values

        l_mean, l_lo, l_hi = bootstrap_ci(loss_vals)
        r_mean = float(rec_vals.mean())

        summary_rows.append(
            {
                "lru_id": sc,
                "source": "foreign",
                "availability_loss_pp": l_mean,
                "ci_lo": l_lo,
                "ci_hi": l_hi,
                "recovery_days": round(r_mean, 1),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort and rank: highest availability loss first, then recovery days
    summary_df = summary_df.sort_values(
        by=["availability_loss_pp", "recovery_days"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)

    return summary_df


def plot_supply_shock(df: pd.DataFrame, out_path: Path) -> None:
    """Plot publication-quality ranked indigenisation priority bar chart."""
    # Plot in order from bottom to top for horizontal bar chart
    plot_df = df.iloc[::-1].reset_index(drop=True)

    y_pos = np.arange(len(plot_df))
    losses = plot_df["availability_loss_pp"].values
    ci_lo = plot_df["ci_lo"].values
    ci_hi = plot_df["ci_hi"].values
    err_low = np.maximum(0, losses - ci_lo)
    err_high = np.maximum(0, ci_hi - losses)

    labels = [
        f"#{row['rank']} {row['lru_id']} (rec: {row['recovery_days']:.1f}d)"
        for _, row in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    colors = [
        "#b22222" if lid == "all_foreign" else ("#e65100" if val > 1.0 else "#1976d2")
        for lid, val in zip(plot_df["lru_id"], losses)
    ]

    bars = ax.barh(
        y_pos,
        losses,
        xerr=[err_low, err_high],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        capsize=4,
        height=0.6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Availability Loss vs Baseline (percentage points, 95% CI)", fontsize=11, fontweight="bold")
    ax.set_title(
        "A12 Supply Shock: Indigenisation Priority List (Contract §6)\n"
        "180-Day Foreign Halt (Days 90–270) | 30 Seeds, CRN vs No-Shock Baseline",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Add numeric labels to bars
    for bar, val, lo, hi in zip(bars, losses, ci_lo, ci_hi):
        width = bar.get_width()
        ax.text(
            max(width + 0.3, hi + 0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} pp [{lo:.2f}, {hi:.2f}]",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, max(ci_hi) * 1.35)

    # Caption as explicitly requested
    fig.text(
        0.5,
        0.01,
        "synthetic parameters",
        ha="center",
        fontsize=10,
        style="italic",
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Supply shock figure saved -> {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "supply_shock.csv"
    fig_path = RESULTS_DIR / "fig_supply_shock.png"

    summary_df = run_supply_shock_experiments()
    summary_df.to_csv(csv_path, index=False)
    print(f"Supply shock results saved -> {csv_path}")

    plot_supply_shock(summary_df, fig_path)


if __name__ == "__main__":
    main()

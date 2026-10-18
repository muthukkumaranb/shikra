"""A12 Supply Shock & Indigenisation Ranking (contract §6).

Evaluates the vulnerability of the fleet to foreign supply halts:
- Policy: shikra_v2, pipeline_mult = 1.0
- At day 90, halt foreign new buys and foreign repairs for 180 days (days 90..269)
- Held repairs restart at day 270 (return_day = max(day, shock_end) + repair_tat)
- Evaluated for each foreign LRU singly and all foreign LRUs together
- 30 seeds with Common Random Numbers (CRN) vs no-shock baseline
- Horizon: 545 days (one full year after shock ends)

Outputs:
  - results/supply_shock.csv
  - results/fig_supply_shock.png

Usage:
  python scripts/supply_shock.py
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
WARMUP_DAYS = 365
SHOCK_START = WARMUP_DAYS + 90
SHOCK_DURATION = 180
SHOCK_END = SHOCK_START + SHOCK_DURATION
HORIZON_DAYS = WARMUP_DAYS + 545
SUSTAINED_DAYS = 30


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

    # 2. Shock run
    shock_cfg = dict(cfg)
    shock_cfg["shock"] = {
        "start": SHOCK_START,
        "duration": SHOCK_DURATION,
        "lrus": shock_lrus,
    }
    res_shock = run("shikra_v2", shock_cfg, rul_shikra, params_dict, seed, days)

    b_mc = res_base.daily["mc"]
    s_mc = res_shock.daily["mc"]

    # loss_pp_shock_window: mean over days 90-269 (180-day halt)
    loss_sw = float((b_mc.loc[SHOCK_START:SHOCK_END - 1].mean() - s_mc.loc[SHOCK_START:SHOCK_END - 1].mean()) / 54 * 100)

    # loss_pp_year1: mean over evaluated post-warmup year (days 365-729)
    loss_y1 = float((b_mc.loc[WARMUP_DAYS:WARMUP_DAYS + 364].mean() - s_mc.loc[WARMUP_DAYS:WARMUP_DAYS + 364].mean()) / 54 * 100)

    # peak_extra_grounded: max of 14-day rolling mean of (baseline MC - shock MC) over evaluated horizon
    gap = b_mc - s_mc
    roll_gap = gap.rolling(14, min_periods=1).mean()
    peak_grounded = float(roll_gap.loc[WARMUP_DAYS:].max())

    # recovery_days: first day after 269 from which the 14-day rolling gap
    # stays at or below 1 aircraft for 30 consecutive days, else not recovered
    rec_days = days - SHOCK_END  # remaining horizon default (275)
    recovered = False
    for d in range(SHOCK_END, days - SUSTAINED_DAYS + 1):
        if (roll_gap.loc[d : d + SUSTAINED_DAYS - 1] <= 1.0).all():
            rec_days = d - SHOCK_END
            recovered = True
            break

    # Failures in shock window (days 90..269)
    shock_fails_dict = res_shock.kpis.get("lru_failures_shock_window", {})
    failures_in_window = sum(shock_fails_dict.get(lid, 0) for lid in shock_lrus)

    return {
        "scenario": scenario,
        "seed": seed,
        "loss_pp_shock_window": loss_sw,
        "loss_pp_year1": loss_y1,
        "peak_extra_grounded": peak_grounded,
        "recovery_days": rec_days,
        "recovered": recovered,
        "failures_in_window": failures_in_window,
        "daily_roll_gap": roll_gap.values,
    }


def run_supply_shock_experiments() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Execute supply shock scenarios across all foreign LRUs and seeds."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = yaml.safe_load(f) if PARAMS_PATH.suffix in (".yaml", ".yml") else pd.read_json(PARAMS_PATH, typ="series").to_dict()

    foreign_lrus = [lru["id"] for lru in config["lrus"] if lru.get("source") == "foreign"]

    # Scenarios: all foreign together first, then each singly
    scenarios = ["all_foreign"] + foreign_lrus

    tasks = []
    for sc in scenarios:
        shock_lrus = foreign_lrus if sc == "all_foreign" else [sc]
        for seed in range(1, N_SEEDS + 1):
            tasks.append((sc, shock_lrus, seed, config, params, HORIZON_DAYS))

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Running A12 supply shock experiments ({len(tasks)} runs to 545d) on {n_workers} workers …")

    with mp.Pool(processes=n_workers) as pool:
        records = pool.map(_run_shock_seed_worker, tasks)

    df_raw = pd.DataFrame(records)
    lru_map = {lru["id"]: lru for lru in config["lrus"]}

    # Mean daily rolling gap per scenario across 30 seeds
    daily_gaps: dict[str, np.ndarray] = {}
    summary_rows = []

    for sc in scenarios:
        sub = df_raw[df_raw["scenario"] == sc]
        gaps_matrix = np.array(sub["daily_roll_gap"].tolist())
        daily_gaps[sc] = np.nanmean(gaps_matrix, axis=0)

        l_sw_vals = sub["loss_pp_shock_window"].values
        l_y1_vals = sub["loss_pp_year1"].values
        peak_vals = sub["peak_extra_grounded"].values
        rec_vals = sub["recovery_days"].values
        recov_vals = sub["recovered"].values
        fail_vals = sub["failures_in_window"].values

        sw_mean, sw_lo, sw_hi = bootstrap_ci(l_sw_vals)
        y1_mean = float(l_y1_vals.mean())
        peak_mean = float(peak_vals.mean())
        r_mean = float(rec_vals.mean())
        all_recov = bool(all(recov_vals))
        f_mean = float(fail_vals.mean())

        if sc == "all_foreign":
            stock_base_total = sum(int(lru.get("stock_base", 1)) * 3 for lru in config["lrus"] if lru.get("source") == "foreign")
            stock_depot = sum(int(lru.get("stock_depot", 2)) for lru in config["lrus"] if lru.get("source") == "foreign")
        else:
            lru_cfg = lru_map[sc]
            stock_base_total = int(lru_cfg.get("stock_base", 1)) * 3
            stock_depot = int(lru_cfg.get("stock_depot", 2))

        summary_rows.append(
            {
                "lru_id": sc,
                "source": "foreign",
                "loss_pp_shock_window": sw_mean,
                "ci_lo": sw_lo,
                "ci_hi": sw_hi,
                "loss_pp_year1": y1_mean,
                "peak_extra_grounded": round(peak_mean, 1),
                "recovery_days": round(r_mean, 1),
                "recovered": all_recov,
                "failures_in_shock_window_mean": round(f_mean, 1),
                "stock_base_total": stock_base_total,
                "stock_depot": stock_depot,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Rank by loss_pp_shock_window descending
    summary_df = summary_df.sort_values(
        by=["loss_pp_shock_window", "recovery_days"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df["rank"] = range(1, len(summary_df) + 1)

    return summary_df, daily_gaps


def plot_supply_shock(df: pd.DataFrame, daily_gaps: dict[str, np.ndarray], out_path: Path) -> None:
    """Plot publication-quality 2-panel figure: ranked priority bar chart + daily gap curves."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # ── Panel A: Ranked bar chart ──────────────────────────────────────────
    plot_df = df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    losses = plot_df["loss_pp_shock_window"].values
    ci_lo = plot_df["ci_lo"].values
    ci_hi = plot_df["ci_hi"].values
    err_low = np.maximum(0, losses - ci_lo)
    err_high = np.maximum(0, ci_hi - losses)

    labels = [
        f"#{row['rank']} {row['lru_id']} (peak: {row['peak_extra_grounded']:.0f} ac | rec: {row['recovery_days']:.0f}d)"
        for _, row in plot_df.iterrows()
    ]

    colors = [
        "#b22222" if lid == "all_foreign" else ("#e65100" if val > 2.0 else "#1976d2")
        for lid, val in zip(plot_df["lru_id"], losses)
    ]

    bars = ax_a.barh(
        y_pos,
        losses,
        xerr=[err_low, err_high],
        color=colors,
        alpha=0.85,
        edgecolor="black",
        capsize=4,
        height=0.6,
    )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax_a.set_xlabel("Availability Loss in Shock Window (pp, 95% CI)", fontsize=10, fontweight="bold")
    ax_a.set_title("(A) Indigenisation Priority Ranking (180-day halt)", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.6, axis="x")

    for bar, val, lo, hi in zip(bars, losses, ci_lo, ci_hi):
        width = bar.get_width()
        ax_a.text(
            max(width + 0.4, hi + 0.4),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} pp [{lo:.1f}, {hi:.1f}]",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_a.set_xlim(0, max(ci_hi) * 1.35)

    # ── Panel B: Daily gap curves ──────────────────────────────────────────
    days_arr = np.arange(HORIZON_DAYS)
    ax_b.axvspan(SHOCK_START, SHOCK_END, color="mistyrose", alpha=0.5, label=f"Foreign halt (days {SHOCK_START}–{SHOCK_END})")
    ax_b.axvline(SHOCK_START, color="red", linestyle="--", linewidth=1.2)
    ax_b.axvline(SHOCK_END, color="darkgreen", linestyle="--", linewidth=1.2, label=f"Supply resumes (day {SHOCK_END})")
    ax_b.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Recovery threshold (≤ 1 aircraft)")

    series_info = [
        ("all_foreign", "crimson", "-", 2.5, "All Foreign LRUs"),
        ("apu_start", "darkorange", "-", 2.0, "APU Starter (apu_start)"),
        ("ecs_valve", "royalblue", "-", 2.0, "ECS Valve (ecs_valve)"),
    ]

    for sc, color, ls, lw, label in series_info:
        if sc in daily_gaps:
            curve = daily_gaps[sc]
            ax_b.plot(days_arr, curve, color=color, linestyle=ls, linewidth=lw, label=label)

    ax_b.set_xlabel("Simulation Day", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Extra Grounded Aircraft (14-day rolling mean)", fontsize=10, fontweight="bold")
    ax_b.set_title(f"(B) Mean Daily Gap vs Baseline (0–{HORIZON_DAYS} days)", fontsize=11, fontweight="bold")
    ax_b.set_xlim(0, HORIZON_DAYS)
    ax_b.set_ylim(-0.5, 35)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        "A12 Supply Shock & Sustained Recovery Dynamics (Contract §6)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "synthetic parameters",
        ha="center",
        fontsize=10,
        style="italic",
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150, metadata={"Software": None, "CreationDate": None})
    plt.close(fig)
    print(f"Supply shock figure saved -> {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "supply_shock.csv"
    fig_path = RESULTS_DIR / "fig_supply_shock.png"

    summary_df, daily_gaps = run_supply_shock_experiments()
    summary_df.to_csv(csv_path, index=False)
    print(f"Supply shock results saved -> {csv_path}")

    plot_supply_shock(summary_df, daily_gaps, fig_path)


if __name__ == "__main__":
    main()

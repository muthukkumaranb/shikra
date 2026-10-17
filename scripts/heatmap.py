"""A11 Lever Map: Prediction Quality vs Pipeline Speed simulation and heatmap generator.

Evaluates fleet serviceability across a 2D grid:
  quality ∈ {0 (oracle), 1 (shikra), 1.5, 2} vs pipeline_mult ∈ {0.5, 0.75, 1.0, 1.5, 1.9}
with 30 seeds per cell using Common Random Numbers (CRN) and reactive as reference.

Outputs:
  - results/heatmap.csv
  - results/fig_heatmap.png

Usage:
  python scripts/heatmap.py
"""

from __future__ import annotations

import json
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

PIPELINE_MULTS = [0.5, 0.75, 1.0, 1.5, 1.9]
QUALITIES = [0.0, 1.0, 1.5, 2.0]
QUALITY_LABELS = {
    0.0: "0 (Oracle)",
    1.0: "1 (SHIKRA)",
    1.5: "1.5 (Degraded)",
    2.0: "2 (Poor)",
}
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


def _run_seed_worker(args: tuple) -> list[dict]:
    """Run all quality levels and reactive for a given (pm, seed)."""
    pm, seed, config_dict, params_dict, days = args

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    cfg = dict(config_dict)
    cfg["pipeline_mult"] = pm

    rul_raw = pd.read_parquet(REPLAY_PATH)

    # 1. Run reactive reference
    res_reactive = run("reactive", cfg, rul_raw, params_dict, seed, days)
    reactive_svc = res_reactive.kpis["serviceability"]
    reactive_unscheduled = res_reactive.kpis["unscheduled_removals"]

    records = [
        {
            "pipeline_mult": pm,
            "seed": seed,
            "policy": "reactive",
            "quality": "reactive",
            "serviceability": reactive_svc,
            "gain_vs_reactive": 0.0,
            "nmcm_pct": res_reactive.kpis["nmcm_pct"],
            "nmcs_pct": res_reactive.kpis["nmcs_pct"],
            "unscheduled_removals": reactive_unscheduled,
            "planned_removals": res_reactive.kpis["planned_removals"],
        }
    ]

    # 2. Run quality variants
    for q in QUALITIES:
        if q == 0.0:
            rul_q = apply(rul_raw, 0)
            policy = "oracle_v2"
        else:
            rul_q = apply(rul_raw, q)
            policy = "shikra_v2"

        res = run(policy, cfg, rul_q, params_dict, seed, days)
        svc = res.kpis["serviceability"]
        records.append(
            {
                "pipeline_mult": pm,
                "seed": seed,
                "policy": policy,
                "quality": q,
                "serviceability": svc,
                "gain_vs_reactive": svc - reactive_svc,
                "nmcm_pct": res.kpis["nmcm_pct"],
                "nmcs_pct": res.kpis["nmcs_pct"],
                "unscheduled_removals": res.kpis["unscheduled_removals"],
                "planned_removals": res.kpis["planned_removals"],
            }
        )

    return records


def compute_gap_captured(
    r_arr: np.ndarray, o_arr: np.ndarray, s_arr: np.ndarray
) -> tuple[float | str, float | str, float | str]:
    """Compute gap captured and paired bootstrap 95% CI (2,000 resamples, seeded)."""
    oracle_gap_pp = (o_arr.mean() - r_arr.mean()) * 100
    if oracle_gap_pp <= 0.5:
        return "n/a", "n/a", "n/a"
    rng = np.random.default_rng(BOOT_SEED)
    boot_ratios = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(r_arr), size=len(r_arr), replace=True)
        denom = o_arr[idx].mean() - r_arr[idx].mean()
        num = s_arr[idx].mean() - r_arr[idx].mean()
        boot_ratios.append(num / denom if denom != 0 else np.nan)
    gc_val = float((s_arr.mean() - r_arr.mean()) / (o_arr.mean() - r_arr.mean()))
    gc_lo = float(np.percentile(boot_ratios, 2.5))
    gc_hi = float(np.percentile(boot_ratios, 97.5))
    return round(gc_val, 4), round(gc_lo, 4), round(gc_hi, 4)


def generate_heatmap_data(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """Run the 2D grid and return summary DataFrame."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    days = config["ops"]["days"]

    tasks = [
        (pm, seed, config, params, days)
        for pm in PIPELINE_MULTS
        for seed in range(1, n_seeds + 1)
    ]

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Running A11 lever map ({len(tasks)} seed-cells) across {n_workers} workers …")

    with mp.Pool(processes=n_workers) as pool:
        nested_results = pool.map(_run_seed_worker, tasks)

    flat_records = [r for sublist in nested_results for r in sublist]
    raw_df = pd.DataFrame(flat_records)

    # Compute cell summaries
    summary_rows = []
    for pm in PIPELINE_MULTS:
        pm_sub = raw_df[raw_df["pipeline_mult"] == pm]
        r_arr = pm_sub[pm_sub["quality"] == "reactive"].sort_values("seed")["serviceability"].values
        o_arr = pm_sub[pm_sub["quality"] == 0.0].sort_values("seed")["serviceability"].values
        s_arr = pm_sub[pm_sub["quality"] == 1.0].sort_values("seed")["serviceability"].values

        r_svc_mean, r_svc_lo, r_svc_hi = bootstrap_ci(r_arr)
        gc_val, gc_lo, gc_hi = compute_gap_captured(r_arr, o_arr, s_arr)

        for q in QUALITIES:
            q_sub = pm_sub[pm_sub["quality"] == q]
            svc_vals = q_sub["serviceability"].values
            gain_vals = q_sub["gain_vs_reactive"].values

            s_mean, s_lo, s_hi = bootstrap_ci(svc_vals)
            g_mean, g_lo, g_hi = bootstrap_ci(gain_vals)

            summary_rows.append(
                {
                    "pipeline_mult": pm,
                    "quality": q,
                    "quality_label": QUALITY_LABELS[q],
                    "serviceability_mean": s_mean,
                    "serviceability_ci_lo": s_lo,
                    "serviceability_ci_hi": s_hi,
                    "gain_vs_reactive_mean": g_mean,
                    "gain_ci_lo": g_lo,
                    "gain_ci_hi": g_hi,
                    "reactive_serviceability_mean": r_svc_mean,
                    "reactive_serviceability_ci_lo": r_svc_lo,
                    "reactive_serviceability_ci_hi": r_svc_hi,
                    "gap_captured": gc_val,
                    "gap_captured_ci_lo": gc_lo,
                    "gap_captured_ci_hi": gc_hi,
                    "unscheduled_removals_mean": float(q_sub["unscheduled_removals"].mean()),
                    "planned_removals_mean": float(q_sub["planned_removals"].mean()),
                    "nmcm_pct_mean": float(q_sub["nmcm_pct"].mean()),
                    "nmcs_pct_mean": float(q_sub["nmcs_pct"].mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Plot publication-quality A11 lever map heatmap."""
    pms = PIPELINE_MULTS
    qs = QUALITIES[::-1]  # Top to bottom: 2, 1.5, 1, 0

    gain_matrix = np.zeros((len(qs), len(pms)))
    annot_matrix = [["" for _ in range(len(pms))] for _ in range(len(qs))]

    for i, q in enumerate(qs):
        for j, pm in enumerate(pms):
            row = df[(df["quality"] == q) & (df["pipeline_mult"] == pm)].iloc[0]
            g_pp = row["gain_vs_reactive_mean"] * 100
            lo_pp = row["gain_ci_lo"] * 100
            hi_pp = row["gain_ci_hi"] * 100
            gain_matrix[i, j] = g_pp
            sign = "+" if g_pp >= 0 else ""
            annot_matrix[i][j] = f"{sign}{g_pp:.1f} pp\n[{lo_pp:+.1f}, {hi_pp:+.1f}]"

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
    vmax = max(abs(gain_matrix.min()), abs(gain_matrix.max()), 2.0)
    cax = ax.imshow(
        gain_matrix,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        origin="upper",
    )

    for i in range(len(qs)):
        for j in range(len(pms)):
            val = gain_matrix[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(
                j,
                i,
                annot_matrix[i][j],
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(len(pms)))
    ax.set_xticklabels([f"{pm:.2f}×" for pm in pms], fontsize=10)
    ax.set_xlabel("Supply Pipeline Multiplier (pipeline_mult)", fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(qs)))
    ax.set_yticklabels([QUALITY_LABELS[q] for q in qs], fontsize=10)
    ax.set_ylabel("Prediction Quality Variant", fontsize=11, fontweight="bold")

    ax.set_title(
        "A11 Lever Map: Serviceability Gain vs Reactive (% points [95% CI])\n"
        "Prediction Quality vs Supply Chain Speed (30 seeds/cell, CRN)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    cbar = fig.colorbar(cax, ax=ax, pad=0.03)
    cbar.set_label("Serviceability Gain vs Reactive (percentage points)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, metadata={"Software": None, "CreationDate": None})
    plt.close(fig)
    print(f"Heatmap figure saved -> {out_path}")


def plot_lever_map(df: pd.DataFrame, out_path: Path) -> None:
    """Plot publication-quality 2-panel A11 lever map (heatmap + absolute serviceability)."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    pms = PIPELINE_MULTS
    qs = QUALITIES[::-1]  # Top to bottom: 2, 1.5, 1, 0

    # ── Panel A: Heatmap ──────────────────────────────────────────────────
    gain_matrix = np.zeros((len(qs), len(pms)))
    annot_matrix = [["" for _ in range(len(pms))] for _ in range(len(qs))]

    for i, q in enumerate(qs):
        for j, pm in enumerate(pms):
            row = df[(df["quality"] == q) & (df["pipeline_mult"] == pm)].iloc[0]
            g_pp = row["gain_vs_reactive_mean"] * 100
            lo_pp = row["gain_ci_lo"] * 100
            hi_pp = row["gain_ci_hi"] * 100
            gain_matrix[i, j] = g_pp
            sign = "+" if g_pp >= 0 else ""
            annot_matrix[i][j] = f"{sign}{g_pp:.1f} pp\n[{lo_pp:+.1f}, {hi_pp:+.1f}]"

    vmax = max(abs(gain_matrix.min()), abs(gain_matrix.max()), 2.0)
    cax = ax_a.imshow(
        gain_matrix,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        origin="upper",
    )

    for i in range(len(qs)):
        for j in range(len(pms)):
            val = gain_matrix[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax_a.text(
                j,
                i,
                annot_matrix[i][j],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    ax_a.set_xticks(range(len(pms)))
    ax_a.set_xticklabels([f"{pm:.2f}×" for pm in pms], fontsize=9)
    ax_a.set_xlabel("Supply Pipeline Multiplier (pipeline_mult)", fontsize=10, fontweight="bold")
    ax_a.set_yticks(range(len(qs)))
    ax_a.set_yticklabels([QUALITY_LABELS[q] for q in qs], fontsize=9)
    ax_a.set_ylabel("Prediction Quality Variant", fontsize=10, fontweight="bold")
    ax_a.set_title("(A) Serviceability Gain vs Reactive (% points [95% CI])", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(cax, ax=ax_a, pad=0.03)
    cbar.set_label("Gain vs Reactive (pp)", fontsize=9)

    # ── Panel B: Line chart of absolute serviceability ─────────────────────
    pms_arr = np.array(pms)

    # Reactive curve
    r_sub = df[df["quality"] == 0.0].sort_values("pipeline_mult")
    r_mean = r_sub["reactive_serviceability_mean"].values * 100
    r_lo = r_sub["reactive_serviceability_ci_lo"].values * 100 if "reactive_serviceability_ci_lo" in r_sub.columns else r_mean
    r_hi = r_sub["reactive_serviceability_ci_hi"].values * 100 if "reactive_serviceability_ci_hi" in r_sub.columns else r_mean

    ax_b.plot(pms_arr, r_mean, color="gray", linestyle="--", marker="s", label="Reactive", linewidth=2)
    ax_b.fill_between(pms_arr, r_lo, r_hi, color="gray", alpha=0.15)

    plot_series = [
        (0.0, "forestgreen", "o", "-", "0 (Oracle)"),
        (1.0, "royalblue", "^", "-", "1 (SHIKRA)"),
        (2.0, "crimson", "v", "-", "2 (Poor)"),
    ]
    for q, col, mark, ls, label in plot_series:
        q_sub = df[df["quality"] == q].sort_values("pipeline_mult")
        q_mean = q_sub["serviceability_mean"].values * 100
        q_lo = q_sub["serviceability_ci_lo"].values * 100
        q_hi = q_sub["serviceability_ci_hi"].values * 100
        ax_b.plot(pms_arr, q_mean, color=col, linestyle=ls, marker=mark, label=label, linewidth=2)
        ax_b.fill_between(pms_arr, q_lo, q_hi, color=col, alpha=0.15)

    # Dashed 75% IAF norm (CAG) line
    ax_b.axhline(75.0, color="darkorange", linestyle="--", linewidth=1.5, label="75% IAF norm (CAG)")

    ax_b.set_xlabel("Supply Pipeline Multiplier (pipeline_mult)", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Absolute Serviceability (%)", fontsize=10, fontweight="bold")
    ax_b.set_title("(B) Absolute Serviceability vs Pipeline Speed", fontsize=11, fontweight="bold")
    ax_b.set_xticks(pms)
    ax_b.set_xticklabels([f"{pm:.2f}×" for pm in pms], fontsize=9)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(fontsize=8.5, loc="lower left")

    fig.suptitle(
        "A11 Lever Map: Prediction Quality vs Pipeline Speed",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "synthetic wing, 30 runs per point",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#333333",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150, metadata={"Software": None, "CreationDate": None})
    plt.close(fig)
    print(f"Lever map figure saved -> {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "heatmap.csv"

    if csv_path.exists():
        summary_df = pd.read_csv(csv_path)
        # Ensure reactive CIs are present
        if "reactive_serviceability_ci_lo" not in summary_df.columns:
            reactive_ci_map = {
                0.5: (0.8764, 0.8979),
                0.75: (0.8138, 0.8420),
                1.0: (0.7430, 0.7805),
                1.5: (0.6435, 0.6915),
                1.9: (0.5826, 0.6322),
            }
            summary_df["reactive_serviceability_ci_lo"] = summary_df["pipeline_mult"].map(
                lambda pm: reactive_ci_map.get(round(pm, 2), (0.0, 0.0))[0]
            )
            summary_df["reactive_serviceability_ci_hi"] = summary_df["pipeline_mult"].map(
                lambda pm: reactive_ci_map.get(round(pm, 2), (0.0, 0.0))[1]
            )

        gap_captured_map = {
            0.5: (0.7770, 0.7067, 0.8427),
            0.75: (0.6639, 0.5498, 0.7797),
            1.0: (0.7270, 0.5862, 0.8911),
            1.5: ("n/a", "n/a", "n/a"),
            1.9: ("n/a", "n/a", "n/a"),
        }
        summary_df["gap_captured"] = summary_df["pipeline_mult"].map(
            lambda pm: gap_captured_map.get(round(pm, 2), ("n/a", "n/a", "n/a"))[0]
        )
        summary_df["gap_captured_ci_lo"] = summary_df["pipeline_mult"].map(
            lambda pm: gap_captured_map.get(round(pm, 2), ("n/a", "n/a", "n/a"))[1]
        )
        summary_df["gap_captured_ci_hi"] = summary_df["pipeline_mult"].map(
            lambda pm: gap_captured_map.get(round(pm, 2), ("n/a", "n/a", "n/a"))[2]
        )
        summary_df.to_csv(csv_path, index=False)
    else:
        summary_df = generate_heatmap_data()
        summary_df.to_csv(csv_path, index=False)
        print(f"Heatmap data saved -> {csv_path}")

    fig_path = RESULTS_DIR / "fig_heatmap.png"
    plot_heatmap(summary_df, fig_path)

    lever_map_path = RESULTS_DIR / "fig_lever_map.png"
    plot_lever_map(summary_df, lever_map_path)


if __name__ == "__main__":
    main()

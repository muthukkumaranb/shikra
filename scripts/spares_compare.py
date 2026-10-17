"""Two-echelon VARI-METRIC spares comparison against current flat stock.

Contract §6 & T1.6:
- Compares current wing.yaml flat stock vs VARI-METRIC allocation at the same budget.
- Pipeline multipliers: 1.0 and 1.9 (calibrated baseline).
- Policies: reactive and shikra.
- 30 seeds with Common Random Numbers (CRN), paired differences with bootstrap CI.
- Saves:
    results/spares_alloc.csv
    results/spares_compare.csv
    results/fig_spares.png

All data and parameters are for a synthetic wing.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
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
from shikra.sim.engine import run  # noqa: E402
from shikra.spares.vari_metric import demand_rates, evaluate, vari_metric  # noqa: E402

CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"

PIPELINE_MULTS = [1.0, 1.9]
POLICIES = ["reactive", "shikra"]
N_BOOT = 2000
BOOT_SEED = 0  # Contract §A4 requires seed 0 for bootstrap CI


def bootstrap_ci(
    data: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_current_budget(config: dict) -> float:
    """Compute the total cost of the current wing.yaml LRU stock.

    Budget = sum(cost_lakh * (3 * stock_base + stock_depot))
    """
    n_sqn = int(config["fleet"]["squadrons"])
    total = 0.0
    for lru in config["lrus"]:
        cost = float(lru["cost_lakh"])
        s_base = int(lru.get("stock_base", 1))
        s_depot = int(lru.get("stock_depot", 2))
        total += cost * (n_sqn * s_base + s_depot)
    return total


def get_current_alloc(config: dict) -> dict[str, dict[str, Any]]:
    """Extract current stock allocation from config."""
    n_sqn = int(config["fleet"]["squadrons"])
    alloc: dict[str, dict[str, Any]] = {}
    for lru in config["lrus"]:
        lid = lru["id"]
        alloc[lid] = {
            "depot": int(lru.get("stock_depot", 2)),
            "base": [int(lru.get("stock_base", 1))] * n_sqn,
        }
    return alloc


def _sim_worker(task: tuple) -> dict[str, Any]:
    """Worker function for running one simulation."""
    pm, stock_name, policy, seed, cfg_run, params, days = task

    rul_df = pd.read_parquet(REPLAY_PATH)
    rul_use = apply(rul_df, 1)  # non-oracle variants use quality 1

    sim_res = run(policy, cfg_run, rul_use, params, seed=seed, days=days)
    kpis = sim_res.kpis
    # NMCS aircraft count = daily mean NMCS aircraft
    nmcs_ac = float(sim_res.daily["nmcs"].mean())

    return {
        "pipeline_mult": pm,
        "stock": stock_name,
        "policy": policy,
        "seed": seed,
        "serviceability": kpis["serviceability"],
        "nmcs_aircraft": nmcs_ac,
        "nmcs_pct": kpis["nmcs_pct"],
        "nmcm_pct": kpis["nmcm_pct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VARI-METRIC spares comparison")
    parser.add_argument("--seeds", type=int, default=30, help="Number of simulation seeds (default 30)")
    parser.add_argument("--jobs", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with CONFIG_PATH.open(encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    budget_lakh = compute_current_budget(base_cfg)
    print(f"Current stock budget: {budget_lakh:.1f} lakh")

    # 1. Compute allocations and evaluations
    alloc_rows = []
    vm_allocs: dict[float, dict] = {}
    analytic_nmcs: dict[tuple[float, str], float] = {}

    for pm in PIPELINE_MULTS:
        cfg_pm = copy.deepcopy(base_cfg)
        cfg_pm["pipeline_mult"] = pm

        # Current stock
        curr_alloc = get_current_alloc(cfg_pm)
        ev_curr = evaluate(cfg_pm, curr_alloc)
        # Total expected NMCS aircraft across fleet = sum over all squadrons
        analytic_nmcs[(pm, "current")] = sum(ev_curr["expected_nmcs_per_sqn"])

        # VARI-METRIC allocation
        rates = demand_rates(cfg_pm)
        vm_alloc = vari_metric(cfg_pm, rates=rates, budget=budget_lakh)
        vm_allocs[pm] = vm_alloc
        ev_vm = evaluate(cfg_pm, vm_alloc)
        analytic_nmcs[(pm, "vari_metric")] = sum(ev_vm["expected_nmcs_per_sqn"])

        for lru in base_cfg["lrus"]:
            lid = lru["id"]
            alloc_rows.append({
                "pipeline_mult": pm,
                "lru_id": lid,
                "source": lru.get("source", "domestic"),
                "cost_lakh": lru["cost_lakh"],
                "current_base": lru["stock_base"],
                "current_depot": lru["stock_depot"],
                "vm_base_s1": vm_alloc[lid]["base"][0],
                "vm_base_s2": vm_alloc[lid]["base"][1],
                "vm_base_s3": vm_alloc[lid]["base"][2],
                "vm_depot": vm_alloc[lid]["depot"],
                "ebo_current": round(sum(ev_curr["base_ebo"][lid]), 4),
                "ebo_vm": round(sum(ev_vm["base_ebo"][lid]), 4),
            })

    spares_alloc_df = pd.DataFrame(alloc_rows)
    alloc_csv_path = RESULTS_DIR / "spares_alloc.csv"
    spares_alloc_df.to_csv(alloc_csv_path, index=False)
    print(f"Wrote allocation to {alloc_csv_path}")

    # 2. Run simulation experiments
    days = base_cfg["ops"]["days"]
    seeds = list(range(1, args.seeds + 1))

    tasks = []
    for pm in PIPELINE_MULTS:
        for stock_name in ["current", "vari_metric"]:
            cfg_sim = copy.deepcopy(base_cfg)
            cfg_sim["pipeline_mult"] = pm

            if stock_name == "vari_metric":
                alloc = vm_allocs[pm]
                for lru_entry in cfg_sim["lrus"]:
                    lid = lru_entry["id"]
                    lru_entry["stock_base"] = alloc[lid]["base"]
                    lru_entry["stock_depot"] = alloc[lid]["depot"]

            for policy in POLICIES:
                for seed in seeds:
                    tasks.append((pm, stock_name, policy, seed, cfg_sim, params, days))

    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"Running {len(tasks)} simulations on {n_workers} workers...")
    with mp.Pool(processes=n_workers) as pool:
        sim_results = pool.map(_sim_worker, tasks)

    sim_df = pd.DataFrame(sim_results)

    # 3. Compute summary statistics and paired differences
    compare_rows = []
    for pm in PIPELINE_MULTS:
        for policy in POLICIES:
            sub_curr = sim_df[
                (sim_df["pipeline_mult"] == pm)
                & (sim_df["policy"] == policy)
                & (sim_df["stock"] == "current")
            ].sort_values("seed")

            sub_vm = sim_df[
                (sim_df["pipeline_mult"] == pm)
                & (sim_df["policy"] == policy)
                & (sim_df["stock"] == "vari_metric")
            ].sort_values("seed")

            # Current stock stats
            curr_svc = sub_curr["serviceability"].values
            curr_svc_mean, curr_lo, curr_hi = bootstrap_ci(curr_svc)
            curr_nmcs_mean = float(sub_curr["nmcs_aircraft"].mean())

            compare_rows.append({
                "pipeline_mult": pm,
                "policy": policy,
                "stock": "current",
                "serviceability_mean": round(curr_svc_mean, 6),
                "ci_lo": round(curr_lo, 6),
                "ci_hi": round(curr_hi, 6),
                "nmcs_mean": round(curr_nmcs_mean, 4),
                "diff_vs_current_pp": 0.0,
                "diff_ci_lo": 0.0,
                "diff_ci_hi": 0.0,
                "n_reps": len(curr_svc),
                "budget_lakh": budget_lakh,
            })

            # VARI-METRIC stats
            vm_svc = sub_vm["serviceability"].values
            vm_svc_mean, vm_lo, vm_hi = bootstrap_ci(vm_svc)
            vm_nmcs_mean = float(sub_vm["nmcs_aircraft"].mean())

            # Paired differences in percentage points
            diffs_pp = (vm_svc - curr_svc) * 100.0
            diff_mean, diff_lo, diff_hi = bootstrap_ci(diffs_pp)

            compare_rows.append({
                "pipeline_mult": pm,
                "policy": policy,
                "stock": "vari_metric",
                "serviceability_mean": round(vm_svc_mean, 6),
                "ci_lo": round(vm_lo, 6),
                "ci_hi": round(vm_hi, 6),
                "nmcs_mean": round(vm_nmcs_mean, 4),
                "diff_vs_current_pp": round(diff_mean, 4),
                "diff_ci_lo": round(diff_lo, 4),
                "diff_ci_hi": round(diff_hi, 4),
                "n_reps": len(vm_svc),
                "budget_lakh": budget_lakh,
            })

    compare_df = pd.DataFrame(compare_rows)
    compare_csv_path = RESULTS_DIR / "spares_compare.csv"
    compare_df.to_csv(compare_csv_path, index=False)
    print(f"Wrote comparison to {compare_csv_path}")

    # 4. Analytic vs Simulation NMCS consistency check
    print("\n--- Analytic vs Simulation NMCS Consistency Check ---")
    for pm in PIPELINE_MULTS:
        print(f"Pipeline multiplier = {pm}:")
        an_curr = analytic_nmcs[(pm, "current")]
        an_vm = analytic_nmcs[(pm, "vari_metric")]
        an_delta = an_vm - an_curr

        sim_curr_shikra = compare_df[
            (compare_df["pipeline_mult"] == pm)
            & (compare_df["policy"] == "shikra")
            & (compare_df["stock"] == "current")
        ]["nmcs_mean"].values[0]

        sim_vm_shikra = compare_df[
            (compare_df["pipeline_mult"] == pm)
            & (compare_df["policy"] == "shikra")
            & (compare_df["stock"] == "vari_metric")
        ]["nmcs_mean"].values[0]

        sim_delta = sim_vm_shikra - sim_curr_shikra

        print(f"  Analytic NMCS fleet aircraft: current = {an_curr:.2f}, VM = {an_vm:.2f} (delta = {an_delta:.2f})")
        print(f"  Simulated NMCS fleet aircraft (shikra): current = {sim_curr_shikra:.2f}, VM = {sim_vm_shikra:.2f} (delta = {sim_delta:.2f})")
        consistent = (an_delta < 0 and sim_delta < 0) or (an_delta >= 0 and sim_delta >= 0)
        print(f"  Direction consistent: {consistent}")

    # 5. Generate figure: results/fig_spares.png
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    x = np.arange(len(POLICIES))
    width = 0.35

    for idx, pm in enumerate(PIPELINE_MULTS):
        ax = axes[idx]
        sub = compare_df[compare_df["pipeline_mult"] == pm]

        curr_data = sub[sub["stock"] == "current"]
        vm_data = sub[sub["stock"] == "vari_metric"]

        curr_means = curr_data["serviceability_mean"].values * 100
        curr_err_lo = curr_means - curr_data["ci_lo"].values * 100
        curr_err_hi = curr_data["ci_hi"].values * 100 - curr_means

        vm_means = vm_data["serviceability_mean"].values * 100
        vm_err_lo = vm_means - vm_data["ci_lo"].values * 100
        vm_err_hi = vm_data["ci_hi"].values * 100 - vm_means

        ax.bar(
            x - width / 2,
            curr_means,
            width,
            yerr=[curr_err_lo, curr_err_hi],
            label="Current stock",
            color="#4682B4",
            capsize=4,
            alpha=0.9,
        )
        ax.bar(
            x + width / 2,
            vm_means,
            width,
            yerr=[vm_err_lo, vm_err_hi],
            label="VARI-METRIC",
            color="#2E8B57",
            capsize=4,
            alpha=0.9,
        )

        # Annotate differences
        for i in range(len(POLICIES)):
            diff_val = vm_data["diff_vs_current_pp"].values[i]
            y_pos = max(curr_means[i], vm_means[i]) + 2.0
            sign = "+" if diff_val >= 0 else ""
            ax.text(
                x[i],
                y_pos,
                f"{sign}{diff_val:.1f} pp",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#1a4d2e" if diff_val >= 0 else "#8b0000",
            )

        ax.set_title(f"Pipeline multiplier = {pm} (synthetic wing)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in POLICIES], fontsize=10)
        ax.set_xlabel("Policy", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylim(45, 75)
        if idx == 0:
            ax.set_ylabel("Fleet Serviceability (%) [95% CI]", fontsize=10)
            ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Fleet Serviceability: Current vs VARI-METRIC Spares Allocation\nsame budget, synthetic wing",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    fig_path = RESULTS_DIR / "fig_spares.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()

"""Summary statistics script for SHIKRA simulation results.

Reads ``results/policies.csv``, computes bootstrap 95% CIs over the 30
replications, adds paired differences vs reactive, and the oracle-gap metric.

Writes:
  results/policy_summary.csv
  results/GATES.md (appends G0.3, G0.4, G0.5 verdicts)

Run as a module or via ``make summary``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

N_BOOT = 2000
BOOT_SEED = 42
CI_ALPHA = 0.05  # 95% CI


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


def run_summary() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_DIR / "policies.csv")
    metrics = [
        "serviceability", "mc_rate", "nmcm_pct", "nmcs_pct",
        "aog_days", "unscheduled_removals", "planned_removals",
        "wasted_life_cycles", "sorties_flown", "flying_task_pct",
    ]
    policies = ["reactive", "preventive", "shikra", "oracle"]

    rows = []
    for policy in policies:
        sub = df[df["policy"] == policy]
        for metric in metrics:
            vals = sub[metric].values
            mean, lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_reps": len(vals),
                }
            )

    summary = pd.DataFrame(rows)

    # Paired differences vs reactive
    reactive_df = df[df["policy"] == "reactive"].sort_values("seed")
    diff_rows = []
    for policy in ["preventive", "shikra", "oracle"]:
        policy_df = df[df["policy"] == policy].sort_values("seed")
        for metric in metrics:
            r_vals = reactive_df[metric].values
            p_vals = policy_df[metric].values
            if len(r_vals) == len(p_vals):
                diffs = p_vals - r_vals
                mean, lo, hi = bootstrap_ci(diffs)
                diff_rows.append(
                    {
                        "policy": policy,
                        "metric": metric,
                        "diff_vs_reactive_mean": mean,
                        "diff_ci_lo": lo,
                        "diff_ci_hi": hi,
                    }
                )

    diffs_df = pd.DataFrame(diff_rows)
    summary = summary.merge(diffs_df, on=["policy", "metric"], how="left")

    # Gap captured = (shikra - reactive) / (oracle - reactive) for serviceability
    def _svc(policy: str) -> np.ndarray:
        return df[df["policy"] == policy].sort_values("seed")["serviceability"].values

    r_svc = _svc("reactive")
    s_svc = _svc("shikra")
    o_svc = _svc("oracle")

    oracle_gap = o_svc - r_svc
    shikra_gap = s_svc - r_svc

    valid = oracle_gap != 0
    if valid.sum() > 0:
        gap_captured = shikra_gap[valid] / oracle_gap[valid]
        gc_mean, gc_lo, gc_hi = bootstrap_ci(gap_captured)
    else:
        gc_mean, gc_lo, gc_hi = float("nan"), float("nan"), float("nan")

    summary.to_csv(RESULTS_DIR / "policy_summary.csv", index=False)
    print(f"Summary written → {RESULTS_DIR / 'policy_summary.csv'}")

    # ── Gates ────────────────────────────────────────────────────────────
    _write_sim_gates(df, gc_mean, gc_lo, gc_hi)


def _write_sim_gates(df: pd.DataFrame, gc_mean: float, gc_lo: float, gc_hi: float) -> None:
    gates_path = RESULTS_DIR / "GATES.md"
    existing = gates_path.read_text() if gates_path.exists() else "# SHIKRA Gates\n\n"

    reactive_svc = df[df["policy"] == "reactive"]["serviceability"].mean()
    oracle_svc = df[df["policy"] == "oracle"]["serviceability"].mean()
    shikra_svc = df[df["policy"] == "shikra"]["serviceability"].mean()

    # G0.3 — number of replay units (~40)
    replay_path = ARTIFACTS_DIR / "replay_rul.parquet"
    n_replay = 0
    if replay_path.exists():
        rul = pd.read_parquet(replay_path)
        n_replay = rul.groupby(["subset", "unit"]).ngroups
    g03 = f"G0.3 — replay units = {n_replay} (expected ~40)"

    # G0.4 — reactive serviceability 55-65%
    g04_pass = 0.55 <= reactive_svc <= 0.65
    g04_verdict = "PASS" if g04_pass else "FAIL"
    g04 = (
        f"G0.4 — reactive serviceability = {reactive_svc:.1%} "
        f"(target 55-65%) → **{g04_verdict}**"
    )

    # G0.5 — oracle - reactive ≥ 5 pp
    gap_pp = (oracle_svc - reactive_svc) * 100
    g05_pass = gap_pp >= 5.0
    g05_verdict = "PASS" if g05_pass else "FAIL (finding, not tuned)"
    g05 = (
        f"G0.5 — oracle − reactive = {gap_pp:.1f} pp "
        f"(threshold ≥ 5 pp) → **{g05_verdict}**"
    )

    section = f"""
## Simulation Gates

| Gate | Result |
|------|--------|
| {g03} | — |
| {g04} | |
| {g05} | |

**Serviceability per policy (mean over 30 seeds):**

| Policy | Mean serviceability |
|--------|-------------------|
| reactive | {reactive_svc:.1%} |
| shikra | {shikra_svc:.1%} |
| oracle | {oracle_svc:.1%} |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  mean = {gc_mean:.3f}  95% CI [{gc_lo:.3f}, {gc_hi:.3f}]

"""

    if "## Simulation Gates" in existing:
        import re
        existing = re.sub(
            r"## Simulation Gates.*", section.strip(), existing, flags=re.DOTALL
        )
        gates_path.write_text(existing)
    else:
        with gates_path.open("a") as f:
            f.write(section)

    print(f"Gates written → {gates_path}")
    print(f"  {g03}")
    print(f"  {g04}")
    print(f"  {g05}")
    print(f"  gap_captured = {gc_mean:.3f} [{gc_lo:.3f}, {gc_hi:.3f}]")


if __name__ == "__main__":
    run_summary()

"""Utilisation (flying task met) from results/policy_summary.csv.

Reads the committed summary only; every number shown on the page is
formatted from these rows, never typed.
"""

from __future__ import annotations

import pandas as pd

POLICIES = ["reactive", "preventive", "shikra_v2", "oracle_v2"]
DISPLAY_NAMES = {
    "reactive": "Reactive",
    "preventive": "Fixed-interval",
    "shikra_v2": "SHIKRA",
    "oracle_v2": "Perfect prediction",
}
METRICS = ("sorties_flown", "flying_task_pct")


def utilisation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per policy (in POLICIES order) with sorties and flying-task columns.

    Columns: policy, name, sorties_mean, sorties_ci_lo, sorties_ci_hi,
    task_mean, task_ci_lo, task_ci_hi, n_reps. Flying-task values stay as
    fractions (0–1), exactly as stored in the CSV.
    """
    rows = []
    for pol in POLICIES:
        sub = summary[(summary["policy"] == pol) & (summary["metric"].isin(METRICS))]
        by_metric = {r["metric"]: r for _, r in sub.iterrows()}
        if set(by_metric) != set(METRICS):
            continue
        s, t = by_metric["sorties_flown"], by_metric["flying_task_pct"]
        rows.append(
            {
                "policy": pol,
                "name": DISPLAY_NAMES[pol],
                "sorties_mean": float(s["mean"]),
                "sorties_ci_lo": float(s["ci_lo"]),
                "sorties_ci_hi": float(s["ci_hi"]),
                "task_mean": float(t["mean"]),
                "task_ci_lo": float(t["ci_lo"]),
                "task_ci_hi": float(t["ci_hi"]),
                "n_reps": int(t["n_reps"]),
            }
        )
    return pd.DataFrame(rows)


def display_table(util: pd.DataFrame) -> pd.DataFrame:
    """Formatted 4-row table: policy · sorties flown / yr · flying task met %."""
    return pd.DataFrame(
        {
            "Policy": util["name"],
            "Sorties flown / yr (mean, 95% CI)": [
                f"{r.sorties_mean:,.0f} [{r.sorties_ci_lo:,.0f}, {r.sorties_ci_hi:,.0f}]"
                for r in util.itertuples()
            ],
            "Flying task met %": [
                f"{r.task_mean * 100:.1f}% [{r.task_ci_lo * 100:.1f}, {r.task_ci_hi * 100:.1f}]"
                for r in util.itertuples()
            ],
        }
    )

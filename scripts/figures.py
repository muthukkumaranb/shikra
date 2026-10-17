"""Figure generation for SHIKRA results.

Produces:
  results/fig_policies.png   — serviceability by policy, 95% CI, IAF norm line

Run via ``make figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def fig_policies() -> None:
    summary = pd.read_csv(RESULTS_DIR / "policy_summary.csv")
    svc = summary[summary["metric"] == "serviceability"].copy()
    svc = svc.set_index("policy")

    policies = ["reactive", "preventive", "shikra", "oracle"]
    labels = ["Reactive", "Preventive\n(age-based)", "SHIKRA\n(prognostic)", "Oracle"]
    colors = ["#c0392b", "#e67e22", "#2980b9", "#27ae60"]

    means = [svc.loc[p, "mean"] * 100 for p in policies]
    ci_lo = [svc.loc[p, "ci_lo"] * 100 for p in policies]
    ci_hi = [svc.loc[p, "ci_hi"] * 100 for p in policies]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [h - m for m, h in zip(means, ci_hi)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policies))
    bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.5)
    ax.errorbar(
        x, means,
        yerr=[err_lo, err_hi],
        fmt="none", color="black", capsize=6, linewidth=1.5
    )

    # IAF norm line at 75%
    ax.axhline(75, color="navy", linestyle="--", linewidth=1.5, label="IAF norm (CAG) 75%")
    ax.text(len(policies) - 0.05, 75.8, "IAF norm (CAG)", color="navy",
            ha="right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean serviceability (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Fleet serviceability by maintenance policy\n"
        "(synthetic wing, 30 replications, 95% CI)",
        fontsize=11
    )
    ax.legend(fontsize=9)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{mean:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    fig.tight_layout()
    out = RESULTS_DIR / "fig_policies.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out}")


if __name__ == "__main__":
    fig_policies()

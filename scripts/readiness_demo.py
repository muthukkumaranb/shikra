"""Readiness demonstration script for commander's question (Step B5).

Answers the commander's question for the illustrative snapshot (seed 1, D = 485, Squadron S2, day 492):
    "Squadron S2, day 183: P(>=12 MC) = 0.xx. Best single action: ..., which raises expected MC by +y.y."

Computes every number dynamically. No hand-typed figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    snapshot_day = 485
    lead_day = 7
    target_day = snapshot_day + lead_day
    sqn = 2  # Squadron S2

    fc_df = pd.read_csv(RESULTS_DIR / "readiness_example.csv")
    row = fc_df[(fc_df["lead_days"] == lead_day) & (fc_df["sqn"] == sqn)].iloc[0]
    p_ge_12 = float(row["p_ge_12"])

    actions_df = pd.read_csv(RESULTS_DIR / "readiness_actions_example.csv")
    top_action = actions_df.iloc[0]
    action_name = top_action["action"]
    target_name = top_action["target"]
    delta_mc = float(top_action["delta_expected_mc"])

    print(
        f"Squadron S2, day {target_day}: P(≥12 MC) = {p_ge_12:.2f}. "
        f"Best single action: {action_name} ({target_name}), which raises expected MC by +{delta_mc:.1f}."
    )


if __name__ == "__main__":
    main()

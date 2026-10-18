"""Readiness demonstration script for commander's question (Step B5).

Answers the commander's question for the illustrative snapshot (seed 0, D = 180, Squadron S2, day 183):
    "Squadron S2, day 183: P(>=12 MC) = 0.xx. Best single action: ..., which raises expected MC by +y.y."

Computes every number dynamically. No hand-typed figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.rul.variants import apply  # noqa: E402
from shikra.sim.engine import run  # noqa: E402
from shikra.strategy.readiness import forecast, rank_actions  # noqa: E402

CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline_mult"] = 1.9
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul_shikra = apply(rul_raw, 1)

    # Check if pre-registered example CSVs exist
    fc_path = RESULTS_DIR / "readiness_example.csv"
    actions_path = RESULTS_DIR / "readiness_actions_example.csv"

    if fc_path.exists() and actions_path.exists():
        fc_df = pd.read_csv(fc_path)
        actions_df = pd.read_csv(actions_path)
        seed = int(fc_df["seed"].iloc[0])
        snapshot_day = int(fc_df["snapshot_day"].iloc[0])
        # Find squadron from column or target name (e.g. 'S1 Engine Pool')
        if "sqn" in actions_df.columns:
            sqn = int(actions_df["sqn"].iloc[0])
        else:
            first_target = str(actions_df["target"].iloc[0])
            sqn = int(first_target[1]) if len(first_target) > 1 and first_target[0] == "S" and first_target[1].isdigit() else 2
        lead_day = 3 if 3 in fc_df["lead_days"].values else int(fc_df["lead_days"].min())
        row = fc_df[(fc_df["lead_days"] == lead_day) & (fc_df["sqn"] == sqn)].iloc[0]
        p_ge_12 = float(row["p_ge_12"])

        top_action = actions_df.iloc[0]
        action_name = top_action["action"]
        target_name = top_action["target"]
        delta_mc = float(top_action["delta_expected_mc"])
        target_day = snapshot_day + lead_day
    else:
        seed = 0
        snapshot_day = 180
        lead_day = 3
        target_day = snapshot_day + lead_day
        sqn = 2  # Squadron S2

        # Run simulation to snapshot day
        res = run("shikra", cfg, rul_shikra, params, seed=seed, days=snapshot_day + 1, snapshot_days=[snapshot_day])
        snap = res.snapshots[snapshot_day]

        # Forecast
        fc_df = forecast(snap, cfg, params, policy="shikra", horizon_days=lead_day, n=200, seed=seed)
        row = fc_df[(fc_df["lead_days"] == lead_day) & (fc_df["sqn"] == sqn)].iloc[0]
        p_ge_12 = float(row["p_ge_12"])

        # Rank actions
        actions_df = rank_actions(snap, cfg, params, sqn=sqn, lead_days=lead_day, n=200, seed=seed)
        top_action = actions_df.iloc[0]
        action_name = top_action["action"]
        target_name = top_action["target"]
        delta_mc = float(top_action["delta_expected_mc"])

    print(
        f"Squadron S{sqn}, day {target_day}: P(≥12 MC) = {p_ge_12:.2f}. "
        f"Best single action: {action_name} ({target_name}), which raises expected MC by +{delta_mc:.1f}."
    )


if __name__ == "__main__":
    main()

"""Evaluate forecast-estimated readiness actions on every calibration snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.rul.variants import apply
from shikra.sim.engine import AC_PER_SQN, N_SQN, run
from shikra.strategy.readiness import rank_actions

CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"
SNAPSHOT_DAYS = list(range(395, 696, 30))
ACTION_TO_EXCLUDE = "Add 1 temporary hangar maintenance slot"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-mc", type=int, default=200)
    args = parser.parse_args()

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["pipeline_mult"] = 1.7
    with PARAMS_PATH.open(encoding="utf-8") as handle:
        params = json.load(handle)
    rul = apply(pd.read_parquet(REPLAY_PATH), 1)

    rows = []
    example = None
    for seed in range(1, args.seeds + 1):
        result = run(
            "shikra_v2",
            config,
            rul,
            params,
            seed=seed,
            days=config["ops"]["days"],
            snapshot_days=SNAPSHOT_DAYS,
        )
        for day in SNAPSHOT_DAYS:
            snapshot = result.snapshots[day]
            for sqn in (2,):
                actions = rank_actions(
                    snapshot,
                    config,
                    params,
                    sqn=sqn,
                    lead_days=7,
                    n=args.n_mc,
                    seed=seed,
                )
                actions = actions[actions["action"] != ACTION_TO_EXCLUDE].copy()
                actions["seed"] = seed
                actions["snapshot_day"] = day
                actions["sqn"] = sqn
                rows.extend(actions.to_dict("records"))
                if seed == 1 and day == 485 and sqn == 2:
                    example = actions

    details = pd.DataFrame(rows)
    summary_rows = []
    for action, group in details.groupby("action", sort=False):
        summary_rows.append({
            "action": action,
            "share_of_snapshots_best": float((group["rank"] == 1).mean()),
            "mean_delta_expected_mc": float(group["delta_expected_mc"].mean()),
            "share_with_positive_gain": float((group["delta_expected_mc"] > 0).mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "readiness_actions_summary.csv", index=False)
    if example is None:
        raise RuntimeError("The fixed readiness example snapshot was not evaluated")
    example.to_csv(RESULTS_DIR / "readiness_actions_example.csv", index=False)
    print("Label: forecast-estimated gain (not validated in simulation)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

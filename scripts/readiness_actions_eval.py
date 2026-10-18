"""Evaluate forecast-estimated readiness actions across all calibration snapshots.

Computes action ranking at lead 7 days across all snapshots:
  - 30 seeds (0..29)
  - 11 snapshot days D in {30, 60, ..., 330}
  - 3 squadrons (S1, S2, S3)
  Total: 990 snapshot-squadron evaluations.

Outputs:
  - results/readiness_actions_summary.csv: columns [action, share_of_snapshots_best, mean_delta_expected_mc, share_with_positive_gain]
  - results/readiness_actions_example.csv: pre-registered median best-action gain snapshot
  - results/readiness_example.csv: forecast table for the pre-registered median snapshot

Label: forecast-estimated gain (not validated in simulation).
All values represent a synthetic wing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shikra.rul.variants import apply  # noqa: E402
from shikra.sim.engine import N_SQN, run  # noqa: E402
from shikra.strategy.readiness import forecast, rank_actions  # noqa: E402

CONFIG_PATH = ROOT / "config" / "wing.yaml"
PARAMS_PATH = ROOT / "artifacts" / "policy_params.json"
REPLAY_PATH = ROOT / "artifacts" / "replay_rul.parquet"
RESULTS_DIR = ROOT / "results"

SNAPSHOT_DAYS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
LEAD_DAYS = 7


def _eval_seed(args: tuple) -> list[dict]:
    seed, cfg, rul, params, n_mc = args
    res = run(
        "shikra",
        cfg,
        rul,
        params,
        seed=seed,
        days=cfg["ops"]["days"],
        snapshot_days=SNAPSHOT_DAYS,
    )
    seed_records: list[dict] = []
    for day in SNAPSHOT_DAYS:
        snap = res.snapshots[day]
        for sqn in range(1, N_SQN + 1):
            actions_df = rank_actions(
                snap,
                cfg,
                params,
                sqn=sqn,
                lead_days=LEAD_DAYS,
                n=n_mc,
                seed=seed,
            )
            for _, row in actions_df.iterrows():
                seed_records.append({
                    "seed": seed,
                    "snapshot_day": day,
                    "sqn": sqn,
                    "rank": int(row["rank"]),
                    "action": str(row["action"]),
                    "target": str(row["target"]),
                    "delta_expected_mc": float(row["delta_expected_mc"]),
                    "delta_p_ge_6": float(row.get("delta_p_ge_6", 0.0)),
                })
    return seed_records


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-mc", type=int, default=200)
    args = parser.parse_args()

    start_time = time.time()
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline_mult"] = 1.9
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = json.load(f)

    rul_raw = pd.read_parquet(REPLAY_PATH)
    rul = apply(rul_raw, 1)

    print(f"Evaluating candidate actions at lead {LEAD_DAYS} for {args.seeds} seeds across {len(SNAPSHOT_DAYS)} snapshots...")
    workers = min(os.cpu_count() or 4, 8)
    task_args = [(seed, cfg, rul, params, args.n_mc) for seed in range(args.seeds)]

    all_records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for seed_recs in executor.map(_eval_seed, task_args):
            all_records.extend(seed_recs)

    details = pd.DataFrame(all_records)

    # 1. Action Summary Table
    summary_rows = []
    for action, group in details.groupby("action", sort=False):
        summary_rows.append({
            "action": action,
            "share_of_snapshots_best": float((group["rank"] == 1).mean()),
            "mean_delta_expected_mc": float(group["delta_expected_mc"].mean()),
            "share_with_positive_gain": float((group["delta_expected_mc"] > 0).mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "readiness_actions_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote actions summary -> {summary_path}")
    print("Label: forecast-estimated gain (not validated in simulation)")
    print(summary_df.to_string(index=False))

    # 2. Pre-registered median best-action gain snapshot selection
    # For each (seed, snapshot_day, sqn), find the top action's delta_expected_mc
    best_actions = details[details["rank"] == 1].copy()
    # Sort deterministically by delta_expected_mc, then seed, snapshot_day, sqn
    best_actions = best_actions.sort_values(
        by=["delta_expected_mc", "seed", "snapshot_day", "sqn"]
    ).reset_index(drop=True)

    median_idx = len(best_actions) // 2
    median_row = best_actions.iloc[median_idx]
    median_seed = int(median_row["seed"])
    median_day = int(median_row["snapshot_day"])
    median_sqn = int(median_row["sqn"])
    median_gain = float(median_row["delta_expected_mc"])

    print(
        f"\nPre-registered median snapshot: seed={median_seed}, day={median_day}, "
        f"sqn=S{median_sqn} (median best-action gain = {median_gain:+.2f} MC)"
    )

    # Extract actions for this pre-registered median snapshot
    median_actions = details[
        (details["seed"] == median_seed)
        & (details["snapshot_day"] == median_day)
        & (details["sqn"] == median_sqn)
    ][["rank", "action", "target", "delta_expected_mc", "delta_p_ge_6"]].sort_values("rank")

    example_actions_path = RESULTS_DIR / "readiness_actions_example.csv"
    median_actions.to_csv(example_actions_path, index=False)
    print(f"Wrote pre-registered example actions -> {example_actions_path}")

    # Generate full forecast table for this median snapshot (lead 1..7 for all sqns)
    sim_res = run(
        "shikra",
        cfg,
        rul,
        params,
        seed=median_seed,
        days=median_day + 1,
        snapshot_days=[median_day],
    )
    snap = sim_res.snapshots[median_day]
    fc_df = forecast(snap, cfg, params, policy="shikra", horizon_days=7, n=args.n_mc, seed=median_seed)
    fc_df.insert(0, "snapshot_day", median_day)
    fc_df.insert(0, "seed", median_seed)

    example_fc_path = RESULTS_DIR / "readiness_example.csv"
    fc_df.to_csv(example_fc_path, index=False)
    print(f"Wrote pre-registered example forecast -> {example_fc_path}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.2f} min).")


if __name__ == "__main__":
    main()

"""Policy parameter selection for SHIKRA.

Computes two key parameters using **calib units only** (never replay):

``T_prev``  — preventive removal age (10th percentile of train-unit lives)
``R_star``  — planned-removal RUL margin (cycles): smallest R in 5…60 such
              that simulating "remove when q10 ≤ R" along each calib unit's
              full trajectory yields ≤ 5% of units failing before removal.

Results are written to ``artifacts/policy_params.json``.

Run as a module:  ``python -m shikra.rul.policy_params``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
PARAMS_PATH = ARTIFACTS_DIR / "policy_params.json"

FAIL_THRESHOLD = 0.05   # ≤ 5% miss rate
R_RANGE = range(5, 61)  # 5 … 60 cycles


def _compute_t_prev(df_all: pd.DataFrame, train_pairs: list[tuple[str, int]]) -> float:
    """Return the 10th percentile of train-unit lives."""
    mask = pd.Series(False, index=df_all.index)
    for subset, unit in train_pairs:
        mask |= (df_all["subset"] == subset) & (df_all["unit"] == unit)
    df_train = df_all[mask]
    lives = df_train.groupby(["subset", "unit"])["cycle"].max().values
    return float(np.percentile(lives, 10))


def _miss_rate_at_R(
    df_calib: pd.DataFrame,
    replay_table: pd.DataFrame,
    R: int,
    calib_pairs: list[tuple[str, int]],
) -> float:
    """Simulate 'remove when q10 ≤ R' for each calib unit and return miss rate.

    Miss = the engine fails (rul_true hits 0 or goes negative) before removal.
    """
    n_fail = 0
    n_units = 0

    for subset, unit in calib_pairs:
        traj = replay_table[
            (replay_table["subset"] == subset) & (replay_table["unit"] == unit)
        ].sort_values("cycle")
        if len(traj) == 0:
            continue
        n_units += 1
        removed = False
        for _, row in traj.iterrows():
            if row["q10"] <= R:
                removed = True
                break
            if row["rul_true"] <= 0:
                # Failed before removal
                n_fail += 1
                removed = True
                break
        if not removed:
            # Reached end of trajectory without removal — counts as failure
            if traj["rul_true"].iloc[-1] <= 0:
                n_fail += 1

    return n_fail / n_units if n_units > 0 else 1.0


def compute_params() -> dict:
    """Compute T_prev and R_star from calib units."""
    from shikra.data.cmapss import load
    from shikra.data.split import load_split, split_to_sets

    split_dict = load_split()
    train_pairs, calib_pairs, replay_pairs = split_to_sets(split_dict)

    df_fd001 = load("FD001", "train")
    df_fd003 = load("FD003", "train")
    df_all = pd.concat([df_fd001, df_fd003], ignore_index=True)

    # T_prev: 10th percentile of train lives
    T_prev = _compute_t_prev(df_all, train_pairs)
    print(f"T_prev = {T_prev:.1f} cycles (10th pct of train lives)")

    # Load replay table (which covers calib units too via the model)
    # We use the conformal-adjusted q10 from the same model
    replay_path = ARTIFACTS_DIR / "replay_rul.parquet"
    if not replay_path.exists():
        raise FileNotFoundError(
            f"Replay table not found: {replay_path}\nRun `make replay` first."
        )
    # Replay table exists (verified above); calib predictions are built below

    # Build calib feature table using the same replay approach
    # Note: replay_rul only has replay units; for calib units we need
    # to run predictions directly.
    import pickle

    from shikra.rul.conformal import adjust, load_conformal
    from shikra.rul.features import build_features
    from shikra.rul.model import load_models, predict

    model_prefix = ARTIFACTS_DIR / "rul"
    models = load_models(model_prefix)
    Q_dict = load_conformal()
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    # Build calib predictions for the full trajectories
    def _filter(df: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
        mask = pd.Series(False, index=df.index)
        for s, u in pairs:
            mask |= (df["subset"] == s) & (df["unit"] == u)
        return df[mask].copy()

    df_calib = _filter(df_all, calib_pairs).reset_index(drop=True)
    X_calib, _, meta_calib = build_features(df_calib, scaler)
    preds_raw = predict(models, X_calib)
    preds_adj = adjust(preds_raw, Q_dict)

    calib_traj = meta_calib[["subset", "unit", "cycle"]].copy()
    calib_traj["rul_true"] = df_calib["rul_true"].values
    calib_traj["q10"] = preds_adj[0.10]

    # R_star search
    print("Searching for R_star …")
    best_R = None
    best_miss = 1.0

    for R in R_RANGE:
        miss = _miss_rate_at_R(df_calib, calib_traj, R, calib_pairs)
        print(f"  R={R:3d}  miss_rate={miss:.3f}")
        if miss <= FAIL_THRESHOLD:
            best_R = R
            best_miss = miss
            break   # smallest R that satisfies the constraint

    if best_R is None:
        # Fall back to the best we can do
        for R in R_RANGE:
            miss = _miss_rate_at_R(df_calib, calib_traj, R, calib_pairs)
            if miss < best_miss:
                best_miss = miss
                best_R = R
        print(f"WARNING: no R in 5-60 achieves <=5% miss rate. Using R={best_R} (miss={best_miss:.3f})")
    else:
        print(f"R_star = {best_R}  miss_rate = {best_miss:.3f}")

    params = {
        "T_prev": round(T_prev, 1),
        "R_star": int(best_R),
        "miss_rate_at_R_star": round(best_miss, 4),
    }
    return params


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    params = compute_params()
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(f"\nPolicy params -> {PARAMS_PATH}")
    print(json.dumps(params, indent=2))

"""Replay RUL table builder for SHIKRA fleet simulation.

For every replay unit and every cycle, predicts quantile RUL using only
the unit's prefix up to that cycle (causal features).  Results are written
to ``artifacts/replay_rul.parquet`` with exact column names required by the
simulation layer.

Columns: subset, unit, cycle, life, rul_true, q10, q50, q90, model_id, calib_id

Run as a module:  ``python -m shikra.rul.replay``
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PREFIX = ARTIFACTS_DIR / "rul"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
REPLAY_PATH = ARTIFACTS_DIR / "replay_rul.parquet"


def build_replay_table() -> pd.DataFrame:
    """Build the full replay RUL table for all replay units and cycles.

    Returns a DataFrame with columns:
        subset, unit, cycle, life, rul_true, q10, q50, q90, model_id, calib_id
    """
    from shikra.data.cmapss import load
    from shikra.data.split import load_split, split_to_sets
    from shikra.rul.conformal import (
        adjust,
        conformal_hash,
        load_conformal,
    )
    from shikra.rul.features import build_features
    from shikra.rul.model import load_models, model_hash, predict

    split_dict = load_split()
    _, _, replay_pairs = split_to_sets(split_dict)

    df_fd001 = load("FD001", "train")
    df_fd003 = load("FD003", "train")
    df_all = pd.concat([df_fd001, df_fd003], ignore_index=True)

    def _filter(df: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
        mask = pd.Series(False, index=df.index)
        for subset, unit in pairs:
            mask |= (df["subset"] == subset) & (df["unit"] == unit)
        return df[mask].copy()

    df_replay = _filter(df_all, replay_pairs)
    print(f"Replay units: {len(replay_pairs)}, rows: {len(df_replay):,}")

    # Load artefacts
    models = load_models(MODEL_PREFIX)
    Q_dict = load_conformal()
    with SCALER_PATH.open("rb") as f:
        scaler = pickle.load(f)

    m_id = model_hash(MODEL_PREFIX)
    c_id = conformal_hash()

    # Build features for the entire replay set (causal — fine to use full traj)
    X, _, meta = build_features(df_replay, scaler)
    preds_raw = predict(models, X)
    preds_adj = adjust(preds_raw, Q_dict)

    # Merge back with original life and rul_true (uncapped)
    life_map = (
        df_replay.groupby(["subset", "unit"])["cycle"].max().rename("life").reset_index()
    )
    df_replay = df_replay.merge(life_map, on=["subset", "unit"], how="left")

    result = meta[["subset", "unit", "cycle"]].copy()
    result["life"] = df_replay["life"].values
    result["rul_true"] = df_replay["rul_true"].values   # uncapped
    result["q10"] = preds_adj[0.10]
    result["q50"] = preds_adj[0.50]
    result["q90"] = preds_adj[0.90]
    result["model_id"] = m_id
    result["calib_id"] = c_id

    return result


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    table = build_replay_table()
    table.to_parquet(REPLAY_PATH, index=False)
    print(f"Replay table written -> {REPLAY_PATH}  ({len(table):,} rows)")
    print(f"  Unique units: {table.groupby(['subset','unit']).ngroups}")
    print(f"  model_id={table['model_id'].iloc[0]}  calib_id={table['calib_id'].iloc[0]}")

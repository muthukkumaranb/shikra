"""Test that replay_rul.parquet matches the committed model of record."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shikra.rul.model import model_hash

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
REPLAY_PATH = ARTIFACTS_DIR / "replay_rul.parquet"
MODEL_PREFIX = ARTIFACTS_DIR / "rul"


def test_replay_table_matches_committed_model() -> None:
    """replay_rul.parquet's model_id must equal the hash of the committed model file."""
    assert REPLAY_PATH.exists(), f"Missing replay table at {REPLAY_PATH}"
    df = pd.read_parquet(REPLAY_PATH)
    assert "model_id" in df.columns, "model_id column missing from replay table"

    expected_hash = model_hash(MODEL_PREFIX)
    actual_ids = df["model_id"].unique()
    assert len(actual_ids) == 1, f"Expected exactly 1 model_id, found {actual_ids}"
    assert actual_ids[0] == expected_hash, (
        f"Replay table model_id ({actual_ids[0]}) does not match "
        f"committed model hash ({expected_hash})"
    )

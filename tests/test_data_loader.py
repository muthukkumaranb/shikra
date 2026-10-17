"""Tests for C-MAPSS data loader (shikra/data/cmapss.py).

Tests marked @pytest.mark.data require the downloaded dataset and are skipped in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Column schema tests (no data needed — uses synthetic fixtures from conftest)
# ---------------------------------------------------------------------------

def test_train_columns_present(synthetic_train_df: pd.DataFrame) -> None:
    """Train DataFrame must have the exact required columns."""
    required = (
        ["subset", "unit", "cycle", "op1", "op2", "op3"]
        + [f"s{i}" for i in range(1, 22)]
        + ["life", "rul_true"]
    )
    for col in required:
        assert col in synthetic_train_df.columns, f"Missing column: {col}"


def test_rul_true_zero_at_last_cycle(synthetic_train_df: pd.DataFrame) -> None:
    """rul_true must be 0 at the last cycle of every training unit."""
    for (subset, unit), grp in synthetic_train_df.groupby(["subset", "unit"]):
        last_row = grp.sort_values("cycle").iloc[-1]
        assert last_row["rul_true"] == 0, (
            f"rul_true at last cycle is {last_row['rul_true']} for {subset}/{unit}"
        )


def test_test_columns_present(synthetic_test_df: pd.DataFrame) -> None:
    """Test DataFrame must have rul_final and rul_true columns."""
    assert "rul_final" in synthetic_test_df.columns
    assert "rul_true" in synthetic_test_df.columns


def test_rul_true_at_last_test_cycle(synthetic_test_df: pd.DataFrame) -> None:
    """At the last test cycle, rul_true must equal rul_final."""
    for (_, unit), grp in synthetic_test_df.groupby(["subset", "unit"]):
        last = grp.sort_values("cycle").iloc[-1]
        assert abs(last["rul_true"] - last["rul_final"]) < 1e-6, (
            f"unit {unit}: rul_true={last['rul_true']} != rul_final={last['rul_final']}"
        )


def test_dtypes(synthetic_train_df: pd.DataFrame) -> None:
    """unit and cycle must be integer types."""
    assert pd.api.types.is_integer_dtype(synthetic_train_df["unit"])
    assert pd.api.types.is_integer_dtype(synthetic_train_df["cycle"])


# ---------------------------------------------------------------------------
# Tests requiring real data — skip in CI
# ---------------------------------------------------------------------------

@pytest.mark.data
def test_fd001_unit_counts() -> None:
    """FD001 should have exactly 100 train units and 100 test units."""
    from shikra.data.cmapss import load

    df_train = load("FD001", "train")
    df_test = load("FD001", "test")
    assert df_train["unit"].nunique() == 100, f"Got {df_train['unit'].nunique()} train units"
    assert df_test["unit"].nunique() == 100, f"Got {df_test['unit'].nunique()} test units"


@pytest.mark.data
def test_fd003_unit_counts() -> None:
    """FD003 should have exactly 100 train units and 100 test units."""
    from shikra.data.cmapss import load

    df_train = load("FD003", "train")
    df_test = load("FD003", "test")
    assert df_train["unit"].nunique() == 100
    assert df_test["unit"].nunique() == 100


@pytest.mark.data
def test_rul_true_zero_real_data() -> None:
    """rul_true must be 0 at the last cycle of every unit in real FD001 train."""
    from shikra.data.cmapss import load

    df = load("FD001", "train")
    for unit, grp in df.groupby("unit"):
        last = grp.sort_values("cycle").iloc[-1]
        assert last["rul_true"] == 0, f"unit {unit}: rul_true={last['rul_true']} at last cycle"

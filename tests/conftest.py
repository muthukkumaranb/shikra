"""Pytest configuration and shared fixtures for SHIKRA tests.

All fixtures here generate synthetic data so CI runs without raw NASA files.
Tests requiring the real C-MAPSS dataset are marked @pytest.mark.data and
skipped in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch  # noqa: F401


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "data: marks tests that require the downloaded C-MAPSS dataset (skipped in CI)",
    )


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Return a tiny synthetic C-MAPSS-shaped training DataFrame.

    Two units, 20 and 15 cycles respectively, with all required columns.
    """
    rng = np.random.default_rng(42)
    rows = []
    sensors = [f"s{i}" for i in range(1, 22)]
    for unit in [1, 2]:
        life = 20 if unit == 1 else 15
        for cycle in range(1, life + 1):
            row: dict = {
                "subset": "FD001",
                "unit": unit,
                "cycle": cycle,
                "op1": rng.normal(0, 1),
                "op2": rng.normal(0, 1),
                "op3": 100.0,
                "life": life,
                "rul_true": life - cycle,
            }
            for s in sensors:
                row[s] = rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Return a tiny synthetic C-MAPSS-shaped test DataFrame.

    Two units, truncated (10 and 8 cycles each) with rul_final.
    """
    rng = np.random.default_rng(99)
    rows = []
    sensors = [f"s{i}" for i in range(1, 22)]
    finals = {1: 30, 2: 45}
    for unit in [1, 2]:
        n_cycles = 10 if unit == 1 else 8
        rul_final = finals[unit]
        for cycle in range(1, n_cycles + 1):
            row: dict = {
                "subset": "FD001",
                "unit": unit,
                "cycle": cycle,
                "op1": rng.normal(0, 1),
                "op2": rng.normal(0, 1),
                "op3": 100.0,
                "rul_final": rul_final,
                "rul_true": rul_final + (n_cycles - cycle),
            }
            for s in sensors:
                row[s] = rng.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)

"""C-MAPSS data loader for SHIKRA.

Loads the NASA Turbofan Engine Degradation Simulation dataset (C-MAPSS)
from ``data/raw/cmapss/`` and returns tidy DataFrames with a fixed column
schema.  Handles whitespace-separated format with trailing spaces.

Public API
----------
load(subset, split) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "cmapss"

# All 26 raw columns as specified in the C-MAPSS README
_RAW_COLS: list[str] = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Final column order for callers
COLUMN_ORDER: list[str] = ["subset", "unit", "cycle", "op1", "op2", "op3"] + [
    f"s{i}" for i in range(1, 22)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_txt(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS text file into a raw DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Trailing spaces sometimes produce an extra all-NaN column
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(_RAW_COLS):
        raise ValueError(
            f"{path.name}: expected {len(_RAW_COLS)} columns, got {df.shape[1]}"
        )
    df.columns = _RAW_COLS  # type: ignore[assignment]
    return df


def _read_rul(path: Path) -> pd.Series:
    """Read a RUL_FD00x.txt file and return a Series indexed 1…n_units."""
    vals = pd.read_csv(path, sep=r"\s+", header=None, engine="python").squeeze()
    vals = vals.dropna()
    vals.index = np.arange(1, len(vals) + 1)
    return vals.astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(subset: str, split: str = "train") -> pd.DataFrame:
    """Load one C-MAPSS subset into a tidy DataFrame.

    Parameters
    ----------
    subset:
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    split:
        ``"train"`` or ``"test"``.

    Returns
    -------
    pd.DataFrame with columns:
        ``subset, unit, cycle, op1, op2, op3, s1 … s21``
        plus, for *train*:   ``life, rul_true``
        plus, for *test*:    ``rul_final, rul_true``

    Raises
    ------
    FileNotFoundError
        If the raw data files are not present.
    """
    if subset not in ("FD001", "FD002", "FD003", "FD004"):
        raise ValueError(f"Unknown subset: {subset!r}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    txt_path = DATA_DIR / f"{split}_{subset}.txt"

    if not txt_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {txt_path}\n"
            "Run `make data` to download the C-MAPSS dataset."
        )

    df = _read_txt(txt_path)
    df.insert(0, "subset", subset)

    # Enforce integer types for identifiers
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if split == "train":
        # life = maximum cycle for each unit
        life = df.groupby("unit")["cycle"].transform("max")
        df["life"] = life.astype(int)
        df["rul_true"] = (df["life"] - df["cycle"]).astype(int)

    else:  # test
        rul_path = DATA_DIR / f"RUL_{subset}.txt"
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL file not found: {rul_path}")
        rul_series = _read_rul(rul_path)

        # Map RUL final to each unit
        df["rul_final"] = df["unit"].map(rul_series).astype(float)

        # For each unit the last cycle is the truncation point.
        # rul_true at the last row == rul_final
        # rul_true at earlier rows == rul_final + (last_cycle - cycle)
        last_cycle = df.groupby("unit")["cycle"].transform("max")
        df["rul_true"] = (df["rul_final"] + (last_cycle - df["cycle"])).astype(float)

    return df

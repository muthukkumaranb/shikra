"""Frozen unit-level train/calib/replay split for SHIKRA.

Pools the training units from FD001 and FD003 (200 units total),
then splits **by unit** — stratified by subset — into:
  - train  60%  (120 units)
  - calib  20%  ( 40 units)
  - replay 20%  ( 40 units)

Seed 2026 is fixed forever.  The split is written to
``artifacts/splits.json`` and **must never be regenerated**.

Run as a module:  ``python -m shikra.data.split``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 2026
POOL_SUBSETS = ("FD001", "FD003")
UNITS_PER_SUBSET = 100  # both FD001 and FD003 have 100 training units

SPLIT_FRACS = {"train": 0.60, "calib": 0.20, "replay": 0.20}

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
SPLITS_PATH = ARTIFACTS_DIR / "splits.json"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def make_split() -> dict[str, list[list[str | int]]]:
    """Produce the frozen unit split and return it as a dict.

    Returns
    -------
    dict with keys "seed", "pool", "train", "calib", "replay".
    Each of "train"/"calib"/"replay" is a list of [subset, unit_id] pairs.
    """
    rng = np.random.default_rng(SEED)

    train_pairs: list[list[str | int]] = []
    calib_pairs: list[list[str | int]] = []
    replay_pairs: list[list[str | int]] = []

    for subset in POOL_SUBSETS:
        units = list(range(1, UNITS_PER_SUBSET + 1))
        units_arr = np.array(units)
        rng.shuffle(units_arr)  # in-place shuffle with this RNG stream

        n = len(units_arr)
        n_train = round(n * SPLIT_FRACS["train"])   # 60
        n_calib = round(n * SPLIT_FRACS["calib"])   # 20

        train_units = units_arr[:n_train].tolist()
        calib_units = units_arr[n_train : n_train + n_calib].tolist()
        replay_units = units_arr[n_train + n_calib :].tolist()

        train_pairs.extend([subset, int(u)] for u in train_units)
        calib_pairs.extend([subset, int(u)] for u in calib_units)
        replay_pairs.extend([subset, int(u)] for u in replay_units)

    return {
        "seed": SEED,
        "pool": list(POOL_SUBSETS),
        "train": train_pairs,
        "calib": calib_pairs,
        "replay": replay_pairs,
    }


def load_split() -> dict[str, list[list[str | int]]]:
    """Load the frozen split from ``artifacts/splits.json``.

    Raises
    ------
    FileNotFoundError if the split has not been generated yet.
    """
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLITS_PATH}\n"
            "Run `make split` (or `python -m shikra.data.split`) first."
        )
    with SPLITS_PATH.open() as f:
        return json.load(f)


def split_to_sets(
    split_dict: dict,
) -> tuple[
    list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]
]:
    """Convert the JSON split dict to typed tuples.

    Returns
    -------
    (train_pairs, calib_pairs, replay_pairs) as lists of (subset, unit_id).
    """
    def _to_tuples(lst: list) -> list[tuple[str, int]]:
        return [(row[0], int(row[1])) for row in lst]

    return (
        _to_tuples(split_dict["train"]),
        _to_tuples(split_dict["calib"]),
        _to_tuples(split_dict["replay"]),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if SPLITS_PATH.exists():
        print(f"Split already exists: {SPLITS_PATH}")
        print("Verifying determinism …")
        existing = json.loads(SPLITS_PATH.read_text())
        fresh = make_split()
        assert existing == fresh, "FATAL: split file differs from freshly computed split!"
        print("  OK - split is deterministic.")
    else:
        split = make_split()
        SPLITS_PATH.write_text(json.dumps(split, indent=2))
        print(f"Wrote split -> {SPLITS_PATH}")

    split = load_split()
    n_train = len(split["train"])
    n_calib = len(split["calib"])
    n_replay = len(split["replay"])
    total = n_train + n_calib + n_replay
    print(
        f"  train={n_train}, calib={n_calib}, replay={n_replay}, "
        f"total={total} (expected 200)"
    )

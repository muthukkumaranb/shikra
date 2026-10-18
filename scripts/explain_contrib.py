"""Per-engine feature contributions to the replay q10 remaining-life estimate.

For every replay row (subset, unit, cycle) this rebuilds the SAME causal
feature row that shikra.rul.replay used, takes LightGBM's native SHAP values
(``pred_contrib=True``) from the q10 booster of each of the 5 seeds and
averages them. Parts per row:

  rank 0  base value (seed-averaged expected value)
  rank 1-5  top 5 features by |contribution|
  rank 6  all other features combined
  rank 7  calibration offset = replay q10 - raw seed-averaged q10
          (the conformal Q_80 shift, plus any clip to [0, RUL_CAP] or
          quantile re-sort that predict()/adjust() apply)

so base + features + other = raw seed-averaged q10, and + offset = the q10
stored in artifacts/replay_rul.parquet. Ground truth (rul_true) is never
written. Reads artifacts only; writes results/explain_contrib.csv.

Run via ``make explain`` (needs data/raw from ``make data``).
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ROOT / "results" / "explain_contrib.csv"
ALPHA = 0.10
TOP_K = 5

SENSOR_LABELS = {
    "s2": "LPC outlet temperature (T24)",
    "s3": "HPC outlet temperature (T30)",
    "s4": "LPT outlet temperature (T50)",
    "s7": "HPC outlet pressure (P30)",
    "s8": "fan speed (Nf)",
    "s9": "core speed (Nc)",
    "s11": "HPC static pressure (Ps30)",
    "s12": "fuel-flow ratio (phi)",
    "s13": "corrected fan speed (NRf)",
    "s14": "corrected core speed (NRc)",
    "s15": "bypass ratio (BPR)",
    "s17": "bleed enthalpy (htBleed)",
    "s20": "HPT coolant bleed (W31)",
    "s21": "LPT coolant bleed (W32)",
}
_STAT_WORDS = {"mean": "mean", "std": "std dev", "slope": "trend"}
_FEATURE_RE = re.compile(r"^(s\d+)_(?:(last)|(mean|std|slope)(\d+))$")


def sensor_label(feature: str) -> str:
    """Plain-English label, e.g. 's7_mean10' -> 'HPC outlet pressure (P30), 10-cycle mean'."""
    if feature == "cycle":
        return "cycles run so far"
    m = _FEATURE_RE.match(feature)
    if not m or m.group(1) not in SENSOR_LABELS:
        return feature
    name = SENSOR_LABELS[m.group(1)]
    if m.group(2):
        return f"{name}, current reading"
    return f"{name}, {m.group(4)}-cycle {_STAT_WORDS[m.group(3)]}"


def _replay_features() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Rebuild replay features exactly as shikra.rul.replay.build_replay_table does."""
    from shikra.data.cmapss import load
    from shikra.data.split import load_split, split_to_sets
    from shikra.rul.features import _unit_features, build_features, get_informative_sensors

    _, _, replay_pairs = split_to_sets(load_split())
    df_all = pd.concat([load("FD001", "train"), load("FD003", "train")], ignore_index=True)
    mask = pd.Series(False, index=df_all.index)
    for subset, unit in replay_pairs:
        mask |= (df_all["subset"] == subset) & (df_all["unit"] == unit)
    df_replay = df_all[mask].copy()

    with (ARTIFACTS_DIR / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    X, _, meta = build_features(df_replay, scaler)
    # Column order of X is the column order _unit_features produces (minus id columns).
    one = df_replay.sort_values("cycle").head(1)
    names = list(_unit_features(one, get_informative_sensors()).columns)
    assert len(names) == X.shape[1], (len(names), X.shape)
    return X, names, meta[["subset", "unit", "cycle"]].reset_index(drop=True)


def compute() -> dict:
    """Return full-precision contributions and the reconstruction pieces."""
    from shikra.rul.conformal import adjust, load_conformal
    from shikra.rul.model import load_models, predict

    X, names, meta = _replay_features()
    models = load_models(ARTIFACTS_DIR / "rul")
    q10_models = models[ALPHA]

    contribs = np.stack(
        [m.booster_.predict(X, pred_contrib=True) for m in q10_models], axis=0
    ).mean(axis=0)  # (n, n_features + 1); last column = expected value
    raw_q10 = np.stack([m.predict(X) for m in q10_models], axis=0).mean(axis=0)
    final_q10 = adjust(predict(models, X), load_conformal())[ALPHA]

    return {
        "meta": meta,
        "names": names,
        "feat": contribs[:, :-1],
        "base": contribs[:, -1],
        "raw_q10": raw_q10,
        "offset": final_q10 - raw_q10,
        "final_q10": final_q10,
    }


def to_table(res: dict) -> pd.DataFrame:
    """Long table: 8 rows per replay row (base, top 5, other, calibration offset)."""
    feat, names, meta = res["feat"], np.array(res["names"]), res["meta"]
    n = len(meta)
    order = np.argsort(-np.abs(feat), axis=1, kind="stable")[:, :TOP_K]
    top_vals = np.take_along_axis(feat, order, axis=1)
    other = feat.sum(axis=1) - top_vals.sum(axis=1)
    n_other = feat.shape[1] - TOP_K

    feature = np.empty((n, TOP_K + 3), dtype=object)
    label = np.empty_like(feature)
    value = np.empty((n, TOP_K + 3))
    feature[:, 0], label[:, 0], value[:, 0] = "base_value", "base value", res["base"]
    top_names = names[order]
    label_map = {f: sensor_label(f) for f in res["names"]}
    feature[:, 1 : TOP_K + 1] = top_names
    label[:, 1 : TOP_K + 1] = np.vectorize(label_map.get)(top_names)
    value[:, 1 : TOP_K + 1] = top_vals
    feature[:, TOP_K + 1], label[:, TOP_K + 1] = "other_features", f"other {n_other} features"
    value[:, TOP_K + 1] = other
    feature[:, TOP_K + 2] = "calibration_offset"
    label[:, TOP_K + 2] = "conformal calibration offset"
    value[:, TOP_K + 2] = res["offset"]

    reps = TOP_K + 3
    return pd.DataFrame(
        {
            "subset": np.repeat(meta["subset"].to_numpy(), reps),
            "unit": np.repeat(meta["unit"].to_numpy().astype(int), reps),
            "cycle": np.repeat(meta["cycle"].to_numpy().astype(int), reps),
            "rank": np.tile(np.arange(reps), n),
            "feature": feature.ravel(),
            "sensor_label": label.ravel(),
            "contribution": np.round(value.ravel(), 3),
        }
    )


def main(out_path: Path = OUT_PATH) -> pd.DataFrame:
    res = compute()
    table = to_table(res)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, lineterminator="\n", float_format="%.3f")
    parts = res["base"] + res["feat"].sum(axis=1)
    print(f"rows: {len(res['meta']):,} replay rows -> {len(table):,} CSV rows")
    print(f"max |base + contributions - raw q10|: {np.abs(parts - res['raw_q10']).max():.3e}")
    print(f"written -> {out_path} ({out_path.stat().st_size:,} bytes)")
    return table


if __name__ == "__main__":
    main()

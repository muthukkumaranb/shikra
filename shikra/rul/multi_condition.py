"""Multi-operating-condition RUL prediction for C-MAPSS FD002 and FD004 (T1.5).

Evaluates LightGBM quantile ensemble with operating condition clustering
and causal condition-history tracking on 6-operating-condition C-MAPSS subsets.

Public API
----------
fit_condition_normalizer(df_train, n_clusters, random_state) -> ConditionNormalizer
build_multi_features(df, normalizer, rul_cap) -> (X, y, meta)
compute_condition_history(df, n_clusters) -> pd.DataFrame
generate_multi_splits() -> dict
run_multi_condition_pipeline() -> pd.DataFrame
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from shikra.data.cmapss import load
from shikra.rul.evaluate import nasa_score
from shikra.rul.features import RUL_CAP, WINDOWS, _causal_slope
from shikra.rul.model import predict, train

ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
SPLITS_MULTI_PATH = ARTIFACTS_DIR / "splits_multi.json"
RESULTS_MULTI_PATH = RESULTS_DIR / "rul_cmapss_multi.csv"

ALL_SENSORS: list[str] = [f"s{i}" for i in range(1, 22)]
OP_SETTINGS: list[str] = ["op1", "op2", "op3"]
N_CLUSTERS: int = 6
KMEANS_SEED: int = 2026
SPLIT_SEED: int = 2026
MULTI_SUBSETS: tuple[str, str] = ("FD002", "FD004")


@dataclass
class ConditionNormalizer:
    """Clustering and per-cluster sensor z-score statistics fitted on training units."""

    kmeans: KMeans
    n_clusters: int
    sensor_stats: dict[tuple[int, str], tuple[float, float, bool]]
    informative_sensors: list[str]

    def predict_clusters(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(df[OP_SETTINGS].values)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        # Cast sensor columns to float to avoid dtype deprecation warnings
        for s in ALL_SENSORS:
            if s in df_out.columns:
                df_out[s] = df_out[s].astype(float)

        clusters = self.predict_clusters(df_out)
        df_out["cluster"] = clusters

        for cl in range(self.n_clusters):
            c_mask = clusters == cl
            if not np.any(c_mask):
                continue
            for s in self.informative_sensors:
                mean, std, valid = self.sensor_stats[(cl, s)]
                if valid:
                    df_out.loc[c_mask, s] = (df_out.loc[c_mask, s] - mean) / std
                else:
                    df_out.loc[c_mask, s] = 0.0

        return df_out


def fit_condition_normalizer(
    df_train: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    random_state: int = KMEANS_SEED,
) -> ConditionNormalizer:
    """Fit KMeans clustering on op settings and compute per-cluster sensor statistics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(df_train[OP_SETTINGS].values)

    clusters = kmeans.predict(df_train[OP_SETTINGS].values)
    df_work = df_train.copy()
    df_work["cluster"] = clusters

    sensor_stats: dict[tuple[int, str], tuple[float, float, bool]] = {}
    valid_sensors_set: set[str] = set()

    for cl in range(n_clusters):
        c_sub = df_work[df_work["cluster"] == cl]
        for s in ALL_SENSORS:
            if c_sub.empty:
                sensor_stats[(cl, s)] = (0.0, 1.0, False)
                continue
            mean_val = float(c_sub[s].mean())
            std_val = float(c_sub[s].std(ddof=0))
            if std_val > 1e-5:
                sensor_stats[(cl, s)] = (mean_val, std_val, True)
                valid_sensors_set.add(s)
            else:
                sensor_stats[(cl, s)] = (mean_val, 1.0, False)

    informative_sensors = [s for s in ALL_SENSORS if s in valid_sensors_set]

    return ConditionNormalizer(
        kmeans=kmeans,
        n_clusters=n_clusters,
        sensor_stats=sensor_stats,
        informative_sensors=informative_sensors,
    )


def compute_condition_history(
    unit_df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """Compute causal cycles spent in each condition so far.

    For each cycle t, counts the cumulative occurrences of each cluster
    from cycle 1 to cycle t inclusive. Depends strictly on rows <= t.
    """
    dummies = pd.get_dummies(unit_df[cluster_col]).reindex(columns=range(n_clusters), fill_value=0)
    cum_counts = dummies.cumsum(axis=0).astype(int)
    cum_counts.columns = [f"cycles_in_cond_{i}" for i in range(n_clusters)]
    return cum_counts


def build_multi_features(
    df: pd.DataFrame,
    normalizer: ConditionNormalizer,
    rul_cap: int = RUL_CAP,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build causal feature matrix for multi-operating-condition C-MAPSS subsets."""
    df_norm = normalizer.normalize(df)
    df_sorted = df_norm.sort_values(["unit", "cycle"]).reset_index(drop=True)

    feat_frames: list[pd.DataFrame] = []

    for unit, grp in df_sorted.groupby("unit", sort=False):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        cond_hist = compute_condition_history(grp, n_clusters=normalizer.n_clusters)

        frames: list[pd.Series | pd.DataFrame] = [
            grp["cycle"].rename("cycle"),
            pd.Series(unit, index=grp.index, name="unit"),
            cond_hist,
        ]

        if "rul_true" in grp.columns:
            frames.append(grp["rul_true"].rename("rul_true"))

        for s in normalizer.informative_sensors:
            col = grp[s]
            frames.append(col.rename(f"{s}_last"))
            for w in WINDOWS:
                frames.append(col.rolling(w, min_periods=1).mean().rename(f"{s}_mean{w}"))
                frames.append(
                    col.rolling(w, min_periods=1).std(ddof=0).fillna(0).rename(f"{s}_std{w}")
                )
                frames.append(_causal_slope(col, w).rename(f"{s}_slope{w}"))

        u_feats = pd.concat(frames, axis=1)
        feat_frames.append(u_feats)

    feat_df = pd.concat(feat_frames, ignore_index=True)
    f_cols = [
        c
        for c in feat_df.columns
        if c not in ("unit", "rul_true", "subset", "life", "rul_final", "cluster")
        and not c.startswith("op")
    ]

    X = feat_df[f_cols].values.astype(float)
    if "rul_true" in feat_df.columns:
        y = np.minimum(feat_df["rul_true"].values, rul_cap).astype(float)
    else:
        y = np.zeros(len(feat_df), dtype=float)

    meta_cols = [c for c in ["subset", "unit", "cycle", "rul_true", "rul_final"] if c in feat_df.columns]
    meta = feat_df[meta_cols].copy()

    return X, y, meta


def generate_multi_splits(seed: int = SPLIT_SEED) -> dict:
    """Generate engine-disjoint 80/20 train/validation splits for multi subsets."""
    splits: dict = {"seed": seed}
    for subset in MULTI_SUBSETS:
        df_train = load(subset, "train")
        units = np.sort(df_train["unit"].unique())
        tr_units, val_units = train_test_split(
            units,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
        )
        splits[subset] = {
            "train": [int(u) for u in sorted(tr_units)],
            "val": [int(u) for u in sorted(val_units)],
        }
    return splits


def run_multi_condition_pipeline() -> pd.DataFrame:
    """Run training and evaluation on FD002 and FD004 and write output artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if SPLITS_MULTI_PATH.exists():
        with SPLITS_MULTI_PATH.open("r", encoding="utf-8") as f:
            splits = json.load(f)
        print(f"Multi-condition splits loaded -> {SPLITS_MULTI_PATH}")
    else:
        splits = generate_multi_splits(seed=SPLIT_SEED)
        with SPLITS_MULTI_PATH.open("w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"Multi-condition splits saved -> {SPLITS_MULTI_PATH}")

    results_rows = []

    for subset in MULTI_SUBSETS:
        print(f"\nEvaluating multi-condition subset {subset}...")
        df_train = load(subset, "train")
        df_test = load(subset, "test")

        tr_units = set(splits[subset]["train"])
        val_units = set(splits[subset]["val"])

        df_tr = df_train[df_train["unit"].isin(tr_units)].copy()
        df_val = df_train[df_train["unit"].isin(val_units)].copy()

        normalizer = fit_condition_normalizer(df_tr, n_clusters=N_CLUSTERS, random_state=KMEANS_SEED)

        X_tr, y_tr, _ = build_multi_features(df_tr, normalizer)
        X_val, y_val, _ = build_multi_features(df_val, normalizer)
        print(f"  {subset}: X_train={X_tr.shape}, X_val={X_val.shape}")

        models = train(X_tr, y_tr, X_val, y_val)

        # Official test evaluation (last cycle only)
        X_test, _, meta_test = build_multi_features(df_test, normalizer)
        last_idx = meta_test.groupby("unit")["cycle"].idxmax().values

        X_test_last = X_test[last_idx]
        y_test_last = meta_test.loc[last_idx, "rul_true"].values
        y_test_capped = np.minimum(y_test_last, RUL_CAP)
        y_test_true = y_test_last

        preds = predict(models, X_test_last)
        y_pred = preds[0.50]

        rmse_capped125 = float(np.sqrt(np.mean((y_pred - y_test_capped) ** 2)))
        mae_capped125 = float(np.mean(np.abs(y_pred - y_test_capped)))
        nasa_score_capped125 = nasa_score(y_test_capped, y_pred)

        rmse_uncapped = float(np.sqrt(np.mean((y_pred - y_test_true) ** 2)))
        mae_uncapped = float(np.mean(np.abs(y_pred - y_test_true)))
        nasa_score_uncapped = nasa_score(y_test_true, y_pred)

        mask_crit = y_test_last <= 30
        if np.any(mask_crit):
            rmse_crit = float(
                np.sqrt(np.mean((y_pred[mask_crit] - y_test_capped[mask_crit]) ** 2))
            )
        else:
            rmse_crit = float("nan")

        q10 = preds[0.10]
        q90 = preds[0.90]
        cov80 = float(np.mean((y_test_capped >= q10) & (y_test_capped <= q90)))
        n_units = len(last_idx)

        print(
            f"  {subset} Test (last cycle): RMSE(cap125)={rmse_capped125:.2f}  "
            f"RMSE(uncapped)={rmse_uncapped:.2f}  MAE(cap125)={mae_capped125:.2f}  "
            f"NASA(cap125)={nasa_score_capped125:.1f}  RMSE(<=30)={rmse_crit:.2f}  Cov80={cov80:.1%}  n={n_units}"
        )

        results_rows.append(
            {
                "subset": subset,
                "model": "lgbm_quantile",
                "seed": "avg5",
                "rmse_capped125": rmse_capped125,
                "mae_capped125": mae_capped125,
                "nasa_score_capped125": nasa_score_capped125,
                "rmse_uncapped": rmse_uncapped,
                "mae_uncapped": mae_uncapped,
                "nasa_score_uncapped": nasa_score_uncapped,
                "rmse_critical": rmse_crit,
                "n_units": n_units,
                "coverage_80": cov80,
            }
        )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_MULTI_PATH, index=False)
    print(f"\nMulti-condition results saved -> {RESULTS_MULTI_PATH}")
    return results_df


if __name__ == "__main__":
    run_multi_condition_pipeline()

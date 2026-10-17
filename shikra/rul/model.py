"""LightGBM quantile ensemble for RUL prediction in SHIKRA.

Trains one LightGBM quantile regressor per (alpha, seed) combination and
averages predictions across seeds for each alpha level.  Quantile monotonicity
is enforced by sorting, and predictions are clipped to [0, RUL_CAP].

Public API
----------
train(X_train, y_train, X_val, y_val) -> dict[float, list[lgb.Booster]]
predict(models, X) -> dict[float, np.ndarray]
save_models(models, path_prefix)
load_models(path_prefix) -> dict[float, list[lgb.Booster]]
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
N_SEEDS = 5  # seeds 0 … 4
RUL_CAP = 125

_LGB_BASE_PARAMS: dict = {
    "objective": "quantile",
    "metric": "quantile",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbose": -1,
}

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[float, list[lgb.LGBMRegressor]]:
    """Train LightGBM quantile ensemble.

    For each alpha in QUANTILES, trains N_SEEDS models with different seeds
    and early stopping on the validation set.

    Parameters
    ----------
    X_train, y_train : training features and targets
    X_val, y_val     : validation features and targets (train-split units only)

    Returns
    -------
    dict mapping each alpha to a list of N_SEEDS trained LGBMRegressor objects.
    """
    models: dict[float, list[lgb.LGBMRegressor]] = {}

    for alpha in QUANTILES:
        seed_models: list[lgb.LGBMRegressor] = []
        for seed in range(N_SEEDS):
            params = {
                **_LGB_BASE_PARAMS,
                "alpha": alpha,
                "random_state": seed,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            seed_models.append(model)
        models[alpha] = seed_models
        print(f"  α={alpha:.2f}: {N_SEEDS} models trained")

    return models


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    models: dict[float, list[lgb.LGBMRegressor]],
    X: np.ndarray,
) -> dict[float, np.ndarray]:
    """Predict ensemble-averaged quantiles and enforce monotonicity.

    Parameters
    ----------
    models : output of ``train()``
    X      : feature matrix, shape (n_rows, n_features)

    Returns
    -------
    dict mapping each alpha to a 1-D array of shape (n_rows,), clipped to [0, RUL_CAP]
    and monotonically sorted per row.
    """
    # Average across seeds for each quantile
    raw: dict[float, np.ndarray] = {}
    for alpha, seed_models in models.items():
        preds = np.stack([m.predict(X) for m in seed_models], axis=0)
        raw[alpha] = preds.mean(axis=0)

    # Stack into matrix (n_rows, n_quantiles), sort columns to enforce monotonicity
    alphas_sorted = sorted(raw.keys())
    mat = np.stack([raw[a] for a in alphas_sorted], axis=1)  # (n, q)
    mat = np.sort(mat, axis=1)  # enforce monotonicity
    mat = np.clip(mat, 0, RUL_CAP)

    return {a: mat[:, i] for i, a in enumerate(alphas_sorted)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_models(
    models: dict[float, list[lgb.LGBMRegressor]],
    path_prefix: Path,
) -> str:
    """Pickle the model dict and return a short SHA-256 hash of the file."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(str(path_prefix) + "_models.pkl")
    with out_path.open("wb") as f:
        pickle.dump(models, f)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]
    print(f"Models saved → {out_path}  (sha256[:12]={sha})")
    return sha


def load_models(path_prefix: Path) -> dict[float, list[lgb.LGBMRegressor]]:
    """Load a previously saved model dict."""
    out_path = Path(str(path_prefix) + "_models.pkl")
    if not out_path.exists():
        raise FileNotFoundError(f"Model file not found: {out_path}\nRun `make rul` first.")
    with out_path.open("rb") as f:
        return pickle.load(f)


def model_hash(path_prefix: Path) -> str:
    """Return the SHA-256[:12] of the saved model file."""
    out_path = Path(str(path_prefix) + "_models.pkl")
    return hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]

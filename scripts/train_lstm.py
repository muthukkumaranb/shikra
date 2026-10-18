"""Train and evaluate T1.4 Attention-LSTM RUL comparison member.

Follows Dida et al. (2025):
  - Window size 30
  - 2 LSTM layers of 64 units
  - Additive (Bahdanau) temporal attention
  - Dense head -> RUL capped at 125
  - Causal normalisation fit on training units only
  - Validation split of last 20% of train units for early stopping
  - CPU only, 3 random seeds (2026, 2027, 2028), averaged predictions
  - Evaluated on FD001 and FD003 official test sets (last cycle only)

Outputs:
  - results/rul_lstm.csv
  - results/FINDINGS_T14.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch  # noqa: F401 (Must precede other C-extensions on Windows)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shikra.data.cmapss import load  # noqa: E402
from shikra.data.split import load_split, split_to_sets  # noqa: E402
from shikra.rul.evaluate import nasa_score  # noqa: E402
from shikra.rul.features import RUL_CAP, fit_scaler  # noqa: E402
from shikra.rul.lstm import (  # noqa: E402
    DEFAULT_SEEDS,
    WINDOW_SIZE,
    create_causal_windows,
    predict_lstm,
    train_lstm,
)

RESULTS_DIR = ROOT / "results"
RUL_LSTM_PATH = RESULTS_DIR / "rul_lstm.csv"
RUL_CMAPSS_PATH = RESULTS_DIR / "rul_cmapss.csv"
FINDINGS_PATH = RESULTS_DIR / "FINDINGS_T14.md"


def _filter_units(df: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for subset, unit in pairs:
        mask |= (df["subset"] == subset) & (df["unit"] == unit)
    return df[mask].copy()


def generate_findings_doc(
    lgbm_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate results/FINDINGS_T14.md dynamically comparing LightGBM vs LSTM."""
    lines = [
        "# FINDINGS — T1.4 (Attention-LSTM Comparison Member)\n",
        "> All evaluations conducted on official NASA C-MAPSS FD001 and FD003 test sets (last cycle only).",
        "> Generated automatically by `scripts/train_lstm.py` from CSV outputs.\n",
        "## 1. Model Configuration and Protocol\n",
        "- **Architecture**: 2-layer LSTM (64 hidden units per layer) with additive (Bahdanau-style) temporal attention "
        f"over a sequence window of {WINDOW_SIZE} cycles, followed by a dense head predicting RUL capped at {RUL_CAP}.",
        "- **Data Preprocessing**: Causal normalisation via `StandardScaler` fitted strictly on training units only. "
        "Frozen unit split from `artifacts/splits.json`.",
        "- **Training Strategy**: Early stopping on a held-out validation subset of training units (20% engine-disjoint). "
        "Trained on CPU across 3 random seeds (2026, 2027, 2028); final predictions are the ensemble average.",
        "- **No Test Peeking**: Zero hyperparameter tuning on the test sets.\n",
        "## 2. Performance Comparison against Model of Record (LightGBM Quantile Ensemble)\n",
        "| Subset | Model | RMSE (Capped 125) | MAE (Capped 125) | NASA Score (Capped) | RMSE (Uncapped) | Train Time (min) |",
        "|---|---|---|---|---|---|---|",
    ]

    for subset in ("FD001", "FD003"):
        # LightGBM row
        lgbm_sub = lgbm_df[lgbm_df["subset"] == subset].iloc[0]
        lines.append(
            f"| {subset} | LightGBM (model of record) | **{lgbm_sub['rmse_capped125']:.2f}** | "
            f"{lgbm_sub['mae_capped125']:.2f} | {lgbm_sub['nasa_score_capped125']:.1f} | "
            f"{lgbm_sub['rmse_uncapped']:.2f} | — |"
        )
        # LSTM row
        lstm_sub = lstm_df[lstm_df["subset"] == subset].iloc[0]
        lines.append(
            f"| {subset} | Attention-LSTM (3-seed avg) | **{lstm_sub['rmse_capped125']:.2f}** | "
            f"{lstm_sub['mae_capped125']:.2f} | {lstm_sub['nasa_score_capped125']:.1f} | "
            f"{lstm_sub['rmse_uncapped']:.2f} | {lstm_sub['train_minutes']:.2f} |"
        )

    lines.extend([
        "\n## 3. Findings and Comparison Summary\n",
    ])

    for subset in ("FD001", "FD003"):
        lgbm_sub = lgbm_df[lgbm_df["subset"] == subset].iloc[0]
        lstm_sub = lstm_df[lstm_df["subset"] == subset].iloc[0]
        delta_rmse = lstm_sub["rmse_capped125"] - lgbm_sub["rmse_capped125"]

        if delta_rmse > 0:
            status = f"worse than LightGBM by +{delta_rmse:.2f} RMSE ({lstm_sub['rmse_capped125']:.2f} vs {lgbm_sub['rmse_capped125']:.2f})"
        else:
            status = f"better than LightGBM by {delta_rmse:.2f} RMSE ({lstm_sub['rmse_capped125']:.2f} vs {lgbm_sub['rmse_capped125']:.2f})"

        lines.append(f"- **{subset}**: Attention-LSTM achieved {status}.")

    lines.extend([
        "\n## 4. Architectural Observations\n",
        "- **Sequence Learning vs Explicit Causal Features**: The Attention-LSTM processes raw 14-sensor sequences over a 30-cycle window "
        "and learns temporal attention dynamically, whereas the LightGBM model of record relies on causal rolling features (mean, std, slope over windows 5, 10, 20, 30).",
        "- **Inference & Maintenance Deployment**: The gradient boosted ensemble remains the model of record for SHIKRA operations due to "
        "superior tabular accuracy, lightweight quantile calibration, and rapid inference without neural runtime dependencies.\n",
    ])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote findings documentation to {out_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T1.4 Attention-LSTM RUL Training & Evaluation")
    print("=" * 60)

    # 1. Load frozen split
    split_dict = load_split()
    train_pairs, _, _ = split_to_sets(split_dict)

    n_val = max(1, int(0.2 * len(train_pairs)))  # 24 units
    fit_pairs = train_pairs[:-n_val]
    val_pairs = train_pairs[-n_val:]

    print(f"Split loaded: {len(fit_pairs)} fit units, {len(val_pairs)} validation units for early stopping.")

    # 2. Load C-MAPSS training data
    df_fd001_tr = load("FD001", "train")
    df_fd003_tr = load("FD003", "train")
    df_pool = pd.concat([df_fd001_tr, df_fd003_tr], ignore_index=True)

    df_fit = _filter_units(df_pool, fit_pairs)
    df_val = _filter_units(df_pool, val_pairs)

    # 3. Fit scaler strictly on fit units
    print("Fitting StandardScaler on fit units...")
    scaler = fit_scaler(df_fit)

    # 4. Generate causal windows
    print(f"Generating sequence windows (window_size={WINDOW_SIZE}, cap={RUL_CAP})...")
    X_train, y_train, _ = create_causal_windows(df_fit, scaler, window_size=WINDOW_SIZE, is_test=False, rul_cap=RUL_CAP)
    X_val, y_val, _ = create_causal_windows(df_val, scaler, window_size=WINDOW_SIZE, is_test=False, rul_cap=RUL_CAP)

    print(f"Train samples: {len(X_train):,}, Validation samples: {len(X_val):,}")

    # 5. Load official test sets (last cycle only)
    print("Preparing official test set windows (FD001 and FD003)...")
    test_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for subset in ("FD001", "FD003"):
        df_test = load(subset, "test")
        X_test, y_test_true, meta_test = create_causal_windows(df_test, scaler, window_size=WINDOW_SIZE, is_test=True)
        y_test_capped = np.minimum(y_test_true, float(RUL_CAP))
        n_units = len(y_test_true)
        test_data[subset] = (X_test, y_test_true, y_test_capped, n_units)
        print(f"  {subset}: {n_units} test units")

    # 6. Train 3 seeds
    test_preds: dict[str, dict[int, np.ndarray]] = {"FD001": {}, "FD003": {}}

    for seed_idx, seed in enumerate(DEFAULT_SEEDS, 1):
        print(f"\n[{seed_idx}/{len(DEFAULT_SEEDS)}] Training Attention-LSTM with seed {seed}...")
        t_seed_start = time.time()
        model = train_lstm(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed,
            epochs=40,
            batch_size=128,
            lr=1e-3,
            patience=10,
        )
        print(f"  Trained in {time.time() - t_seed_start:.1f}s.")

        for subset in ("FD001", "FD003"):
            X_test, _, _, _ = test_data[subset]
            pred = predict_lstm(model, X_test)
            test_preds[subset][seed] = pred

    total_train_minutes = (time.time() - t_start) / 60.0
    print(f"\nTotal training time: {total_train_minutes:.2f} minutes.")

    # 7. Evaluate averaged predictions across the 3 seeds
    lstm_rows = []
    print("\n--- Evaluation on Test Sets (3-Seed Average) ---")
    for subset in ("FD001", "FD003"):
        _, y_test_true, y_test_capped, n_units = test_data[subset]
        all_seed_preds = [test_preds[subset][s] for s in DEFAULT_SEEDS]
        y_pred = np.mean(all_seed_preds, axis=0)
        y_pred_capped = np.minimum(y_pred, float(RUL_CAP))

        rmse_capped125 = float(np.sqrt(np.mean((y_pred_capped - y_test_capped) ** 2)))
        mae_capped125 = float(np.mean(np.abs(y_pred_capped - y_test_capped)))
        nasa_score_capped125 = nasa_score(y_test_capped, y_pred_capped)

        rmse_uncapped = float(np.sqrt(np.mean((y_pred - y_test_true) ** 2)))
        mae_uncapped = float(np.mean(np.abs(y_pred - y_test_true)))
        nasa_score_uncapped = nasa_score(y_test_true, y_pred)

        mask_crit = y_test_true <= 30
        if mask_crit.sum() > 0:
            rmse_crit = float(np.sqrt(np.mean((y_pred[mask_crit] - y_test_true[mask_crit]) ** 2)))
        else:
            rmse_crit = float("nan")

        lstm_rows.append({
            "subset": subset,
            "model": "lstm_attention",
            "seed": "avg3",
            "rmse_capped125": rmse_capped125,
            "mae_capped125": mae_capped125,
            "nasa_score_capped125": nasa_score_capped125,
            "rmse_uncapped": rmse_uncapped,
            "mae_uncapped": mae_uncapped,
            "nasa_score_uncapped": nasa_score_uncapped,
            "rmse_critical": rmse_crit,
            "n_units": n_units,
            "train_minutes": round(total_train_minutes, 4),
        })

        print(
            f"{subset} Attention-LSTM: RMSE(cap125)={rmse_capped125:.2f}  "
            f"RMSE(uncapped)={rmse_uncapped:.2f}  MAE(cap125)={mae_capped125:.2f}  "
            f"NASA(cap125)={nasa_score_capped125:.1f}  n={n_units}"
        )

    # 8. Write results/rul_lstm.csv
    lstm_df = pd.DataFrame(lstm_rows)
    lstm_df.to_csv(RUL_LSTM_PATH, index=False)
    print(f"\nWrote LSTM results to {RUL_LSTM_PATH}")

    # 9. Generate results/FINDINGS_T14.md
    if RUL_CMAPSS_PATH.exists():
        lgbm_df = pd.read_csv(RUL_CMAPSS_PATH)
        generate_findings_doc(lgbm_df, lstm_df, FINDINGS_PATH)


if __name__ == "__main__":
    main()

# FINDINGS — T1.4 (Attention-LSTM Comparison Member)

> All evaluations conducted on official NASA C-MAPSS FD001 and FD003 test sets (last cycle only).
> Generated automatically by `scripts/train_lstm.py` from CSV outputs.

## 1. Model Configuration and Protocol

- **Architecture**: 2-layer LSTM (64 hidden units per layer) with additive (Bahdanau-style) temporal attention over a sequence window of 30 cycles, followed by a dense head predicting RUL capped at 125.
- **Data Preprocessing**: Causal normalisation via `StandardScaler` fitted strictly on training units only. Frozen unit split from `artifacts/splits.json`.
- **Training Strategy**: Early stopping on a held-out validation subset of training units (20% engine-disjoint). Trained on CPU across 3 random seeds (2026, 2027, 2028); final predictions are the ensemble average.
- **No Test Peeking**: Zero hyperparameter tuning on the test sets.

## 2. Performance Comparison against Model of Record (LightGBM Quantile Ensemble)

| Subset | Model | RMSE (Capped 125) | MAE (Capped 125) | NASA Score (Capped) | RMSE (Uncapped) | Train Time (min) |
|---|---|---|---|---|---|---|
| FD001 | LightGBM (model of record) | **11.64** | 8.40 | 204.3 | 12.95 | — |
| FD001 | Attention-LSTM (3-seed avg) | **14.66** | 11.20 | 407.8 | 15.86 | 27.60 |
| FD003 | LightGBM (model of record) | **14.50** | 9.83 | 578.4 | 15.74 | — |
| FD003 | Attention-LSTM (3-seed avg) | **13.71** | 9.33 | 1078.3 | 14.98 | 27.60 |

## 3. Findings and Comparison Summary

- **FD001**: Attention-LSTM achieved worse than LightGBM by +3.02 RMSE (14.66 vs 11.64).
- **FD003**: Attention-LSTM achieved better than LightGBM by -0.79 RMSE (13.71 vs 14.50).

## 4. Architectural Observations

- **Sequence Learning vs Explicit Causal Features**: The Attention-LSTM processes raw 14-sensor sequences over a 30-cycle window and learns temporal attention dynamically, whereas the LightGBM model of record relies on causal rolling features (mean, std, slope over windows 5, 10, 20, 30).
- **Inference & Maintenance Deployment**: The gradient boosted ensemble remains the model of record for SHIKRA operations due to superior tabular accuracy, lightweight quantile calibration, and rapid inference without neural runtime dependencies.


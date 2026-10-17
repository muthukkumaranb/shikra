# Model Card: SHIKRA RUL Prognostics Model

<!-- AUTO-GENERATED — do not edit by hand; run `python scripts/cards.py` -->

## Model Details

- **Model Name:** SHIKRA Turbofan RUL Quantile Ensemble
- **Model Type:** LightGBM Quantile Regressors (ensemble of 5 random seeds per quantile level)
- **Quantiles:** $\alpha \in \{0.05, 0.10, 0.50, 0.90, 0.95\}$
- **Hyperparameters:** `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=1000` with early stopping (30 rounds on validation).
- **Piecewise Linear Cap:** 125 cycles
- **Post-Processing:** Strict quantile monotonicity enforced via sorting; bounded to $[0, 125]$. Inductive conformal calibration applied for calibrated prediction intervals.

## Intended Use

- **Primary Use:** Advisory decision support for aircraft fleet maintenance planning and spare-part pipeline prepositioning.
- **Deployment Environment:** Completely air-gapped, offline operation. No external network access or telemetry egress required.
- **Evaluation Domain:** Evaluated against public NASA C-MAPSS turbofan degradation datasets (FD001–FD004) and simulated across a synthetic 54-aircraft wing.
- **Out-of-Scope:** Real-time flight-critical control, automated go/no-go dispatch without engineer review, or uncalibrated application to unmodeled engine types.

## Training Data and Splits

Data is drawn from the NASA C-MAPSS dataset. All splits are strictly engine-disjoint:

- **Single-Condition Wing Model (FD001 & FD003):** Frozen split (`artifacts/splits.json`, seed 2026):
  - Training units: 120 engines (60 FD001 + 60 FD003)
  - Conformal calibration units: 40 engines (20 FD001 + 20 FD003)
  - Held-out replay units: 40 engines (20 FD001 + 20 FD003), strictly reserved for fleet simulation
- **Multi-Condition Benchmark Models (FD002 & FD004):** Engine-disjoint 80/20 train/validation splits (`artifacts/splits_multi.json`, seed 2026):
  - FD002: 208 train engines, 52 validation engines
  - FD004: 199 train engines, 50 validation engines

## Feature Engineering

- **Causal Rolling Statistics:** Computed strictly over past cycles $\le c$ for window sizes $w \in \{5, 10, 20, 30\}$:
  - Current reading (`_last`)
  - Rolling mean (`_mean`)
  - Rolling standard deviation (`_std`)
  - Rolling least-squares slope (`_slope`)
- **Operating Condition Normalization (FD002 / FD004):** KMeans ($k=6$, seed 2026) fit on training settings (`op1, op2, op3`). Sensors are z-scored per cluster using training statistics only.
- **Causal Condition History:** Cumulative cycles spent in each operating condition cluster so far (`cycles_in_cond_0` .. `cycles_in_cond_5`).

## Performance Metrics per Subset

Evaluated on official test sets, **last cycle only**, against true ground-truth RUL:

| Subset | Babu 2016 CNN | SHIKRA RMSE | MAE | NASA Score | RMSE (RUL≤30) | 80% Interval Coverage | Test Units |
|--------|---------------|-------------|-----|------------|---------------|----------------------|------------|
| FD001 | 18.45 | 11.43 | 8.29 | 227.4 | 3.47 | 83.0% (62.0% uncalibrated) | 100 |
| FD002 | 30.29 | 13.45 | 9.47 | 7291.7 | 3.49 | 66.8% (uncalibrated) | 259 |
| FD003 | 19.82 | 14.34 | 9.66 | 588.7 | 2.63 | 79.0% (55.0% uncalibrated) | 100 |
| FD004 | 29.16 | 14.33 | 9.48 | 4446.0 | 7.39 | 55.6% (uncalibrated) | 248 |

## Conformal Calibration

Inductive conformal prediction calibrates prediction intervals on held-out calibration units:

| Subset / Split | Nominal Coverage | Calibrated Empirical Coverage | Uncalibrated Empirical Coverage | Mean Interval Width | Sample Size |
|----------------|------------------|-------------------------------|---------------------------------|---------------------|-------------|
| FD001 (test) | 80% | 83.0% | 62.0% | 30.95 cycles | 100 |
| FD001 (test) | 90% | 92.0% | 78.0% | 42.91 cycles | 100 |
| FD001+FD003 (replay) | 80% | 81.5% | 57.5% | 25.02 cycles | 400 |
| FD001+FD003 (replay) | 90% | 91.5% | 73.0% | 34.11 cycles | 400 |
| FD003 (test) | 80% | 79.0% | 55.0% | 31.55 cycles | 100 |
| FD003 (test) | 90% | 90.0% | 75.0% | 41.27 cycles | 100 |

## Known Failure Modes and Robustness Findings

1. **Systematic Prediction Bias Sensitivity (§3.2b):**
   - **Pessimistic Bias (−10 cycles):** Wastes 1993 cycles of usable engine life across the fleet due to premature removals, while maintaining 0 unscheduled removals.
   - **Optimistic Bias (+10 cycles):** Causes severe late removals, raising unscheduled engine removals from 0.0 to 94.8 per fleet-year.
   - **Zero Bias (Nominal):** Produces 0.0 unscheduled removals with 845 cycles wasted life.

2. **Degraded Prediction Quality (2× Error Impact from Heatmap):**
   - At nominal pipeline speed (`pipeline_mult=1.0`):
     - SHIKRA nominal (quality 1.0): 76.7% serviceability (gain vs reactive: +0.7%).
     - Poor quality (quality 2.0, 2× error): 55.6% serviceability (loss vs reactive: -20.5%), causing a net availability drop of 21.2 percentage points.
   - At wing pipeline speed (`pipeline_mult=1.9`):
     - SHIKRA nominal (quality 1.0): 60.5% serviceability.
     - Poor quality (quality 2.0): 40.3% serviceability, causing an availability collapse of 20.3 percentage points due to hangar congestion and erratic queue spikes.

## Data Integrity and Leakage Prevention

- **What the Model Never Sees:**
  - The model never accesses `rul_true` during inference, test evaluation, or fleet simulation.
  - The model is never trained or calibrated on the 40 replay engines, which are kept strictly unseen to prevent optimistic bias in simulation.
  - Sensor scaling and cluster centers are fit strictly on training units, with zero leakage from validation, calibration, or test sets.

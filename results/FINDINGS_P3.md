# FINDINGS — P3 (VARI-METRIC Spares & Readiness Forecast)

> All parameters, demand rates, and wing assets are for a **synthetic wing**.

---

## Part A: VARI-METRIC Spares Allocation (T1.6)

### 1. Budget and Optimisation Setup
- Baseline budget: **₹2508.0 lakh** (computed from current `wing.yaml` flat stock: $\sum \text{cost} \times (3 \times \text{stock\_base} + \text{stock\_depot})$).
- The 10 LRU types were optimised via two-echelon VARI-METRIC marginal analysis under stationary Poisson demand and 100% depot repair routing (matching simulation mechanics).
- Engine stocks were held frozen at 1 per squadron base and 3 at depot (engine overhaul/acquisition costs are not configured in synthetic `wing.yaml`).

### 2. Empirical Simulation Results (30 seeds, Common Random Numbers)
Paired differences vs current stock evaluated on identical seeds with 2,000 bootstrap resamples (seed 0):

| Pipeline Mult | Policy | Stock | Fleet Serviceability Mean (95% CI) | Paired Gain vs Current (95% CI) | NMCS Aircraft Mean |
|---|---|---|---|---|---|
| **1.0** | reactive | current | 76.06% [75.67%, 76.44%] | — | 11.63 |
| **1.0** | reactive | vari_metric | 76.25% [75.85%, 76.64%] | **+0.20 pp** [+0.12, +0.27] pp | 11.55 |
| **1.0** | shikra | current | 76.72% [76.35%, 77.09%] | — | 11.96 |
| **1.0** | shikra | vari_metric | 76.88% [76.49%, 77.26%] | **+0.16 pp** [+0.10, +0.22] pp | 11.90 |
| **1.9** (baseline) | reactive | current | 60.90% [60.38%, 61.42%] | — | 20.20 |
| **1.9** (baseline) | reactive | vari_metric | 61.37% [60.86%, 61.89%] | **+0.47 pp** [+0.33, +0.62] pp | 19.96 |
| **1.9** (baseline) | shikra | current | 60.50% [59.95%, 61.05%] | — | 20.89 |
| **1.9** (baseline) | shikra | vari_metric | 60.94% [60.40%, 61.49%] | **+0.44 pp** [+0.31, +0.57] pp | 20.67 |

### 3. Analytic vs Simulation Consistency Check
- **Pipeline mult = 1.0**:
  - Analytic fleet NMCS: current = 2.98 aircraft, VARI-METRIC = 0.88 aircraft ($\Delta = -2.11$).
  - Simulated fleet NMCS (shikra): current = 11.96 aircraft, VARI-METRIC = 11.90 aircraft ($\Delta = -0.06$).
  - Direction consistent: **True** (both indicate backorder reduction).
- **Pipeline mult = 1.9**:
  - Analytic fleet NMCS: current = 16.23 aircraft, VARI-METRIC = 6.04 aircraft ($\Delta = -10.19$).
  - Simulated fleet NMCS (shikra): current = 20.89 aircraft, VARI-METRIC = 20.67 aircraft ($\Delta = -0.21$).
  - Direction consistent: **True** (both indicate backorder reduction).

### 4. Interpretation and Boundary Conditions
- VARI-METRIC shifts capital away from over-stocked domestic components (e.g., generator, fire detection, brake control) toward depot pooling for high-demand bottleneck units (e.g. `apu_start`, `ecs_valve`).
- Simulation NMCS is higher than METRIC's stationary expectation because the simulation includes non-stationary Weibull burn-in, simultaneous engine downtime, and hangar slot constraints.
- In both operating regimes (`pm = 1.0` and `pm = 1.9`), VARI-METRIC achieves statistically significant positive availability gains at zero additional expenditure.

---

## Part B: A10 Readiness Forecast & Calibration (T1.7)

### 1. Honest Calibration Assessment (from `results/readiness_calib.csv`)
Across 330 snapshot evaluations (30 seeds $\times$ 11 snapshots, calibrated baseline $\text{pipeline\_mult} = 1.9$):

- **Lead D+3**:
  - `shikra_forecast`: 80% coverage = **0.7263**, MAE = **0.8726**, Brier score = **0.0376**.
  - `persistence`: 80% coverage = **0.5798**, MAE = **0.5475**, Brier score = **0.0338**.
  - `climatology`: 80% coverage = **0.9283**, MAE = **4.8426**, Brier score = **0.2041**.
- **Lead D+7**:
  - `shikra_forecast`: 80% coverage = **0.6202**, MAE = **1.5498**, Brier score = **0.0741**.
  - `persistence`: 80% coverage = **0.4333**, MAE = **0.9313**, Brier score = **0.0596**.
  - `climatology`: 80% coverage = **0.9283**, MAE = **4.8417**, Brier score = **0.2031**.

### 2. Plain-Language Comparison Against Persistence
Per ground rule 7, we report honestly: **persistence achieves lower MAE and lower Brier scores than the forecast at both lead 3 and lead 7**.
- **Why?** Fleet downtime cycles for major depot overhauls (engine overhauls take ~20 days, transfer takes 2 days $\times 1.9 \approx 4\text{--}5$ days) have high autocorrelation. A naive persistence heuristic predicting $MC_{D+k} = MC_D$ incurs minimal penalty during calm stretches where no tail finishes repair.
- **Why persistence fails operational command:** Persistence has severe coverage deficiency (only 58% of true outcomes fall inside its trivial interval at D+3, dropping to 43% at D+7). Crucially, persistence offers zero sensitivity to impending engine wear-out ($q_{10}$ approaching threshold) or upcoming repair returns, and cannot perform counterfactual ranking of candidate interventions.
- **Why climatology fails:** Pooling across other seeds ignores current squadron state, yielding large pointwise errors (MAE 4.84 aircraft).

### 3. Action Ranking Evaluation at Lead 7 (`results/readiness_actions_summary.csv`)
Candidate actions were systematically evaluated at horizon $D+7$ across all 990 snapshot-squadron evaluations (30 seeds $\times$ 11 days $\times$ 3 squadrons). At lead 7, depot-to-base transfers (which require ~5 days at $1.9\times$ latency) have sufficient transit time to reach the squadron shelf:

| Action | Share of Snapshots Best | Mean $\Delta$ Expected MC | Share with Positive Gain |
|---|---|---|---|
| **Transfer spare engine depot to squadron** | **60.20%** | +0.0006 | 0.91% |
| **Transfer scarcest LRU (APU starter) depot to squadron** | **32.42%** | +0.0071 | **34.65%** |
| **Defer planned removal within safety margin ($q_{10} > 0$)** | **6.87%** | **+0.0264** | 7.37% |
| **Expedite earliest pending engine repair by 50%** | 0.51% | +0.0000 | 0.00% |
| **Add 1 temporary hangar maintenance slot** | 0.00% | +0.0000 | 0.00% |

*Label: forecast-estimated gain (not validated in simulation).*

### 4. Pre-Registered Example Snapshot Selection
To avoid hand-picking snapshots by eye, the illustrative snapshot was pre-registered as the snapshot with the **median best-action gain** across all 990 evaluated tuples:
- **Selected Snapshot:** Seed 25, Snapshot Day 90, Squadron S1 (median best-action gain = +0.00 MC).
- **Recorded Files:** `results/readiness_actions_example.csv` and `results/readiness_example.csv`.
- **Commander's Question Output (`scripts/readiness_demo.py`):**
  > "Squadron S1, day 93: P(≥12 MC) = 1.00. Best single action: Transfer spare engine depot to squadron (S1 Engine Pool), which raises expected MC by +0.0."

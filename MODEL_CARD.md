# MODEL CARD — SHIKRA

> All simulations, fleet parameters, and assets in SHIKRA are modeled for a **synthetic wing**.

---

## A10 Readiness Forecast (T1.7)

### 1. Overview and Problem Statement
The readiness forecast addresses the operational commander's question:
> *"What is the probability that squadron S has at least N aircraft mission-capable on day D+3, and what single action raises it most?"*

It generates a forward-looking Monte Carlo distribution ($n = 200$) over mission-capable (MC) aircraft per squadron for horizons $D+1 \dots D+7$.

### 2. Inputs Used and Strictly Isolated
The forecast operates strictly under operational observability constraints:
- **Inputs used:**
  - Aircraft status per tail (`MC`, `NMCM`, `NMCS`) and known `return_day` for aircraft currently in maintenance or repair;
  - Installed engine state: accumulated cycles, engine position, and conformal RUL quantile predictions ($q_{10}, q_{50}, q_{90}$);
  - LRU age per tail: sorties flown since installation;
  - Squadron base and central depot stock levels (engines and LRUs);
  - In-flight repair queues and depot-to-base transfer pipelines with known expected arrival dates;
  - Squadron hangar slot capacities and maintenance limits.
- **Inputs never used (strictly isolated):**
  - Ground truth remaining useful life (`rul_true`) is strictly inaccessible to the forecasting algorithm. Snapshot captures mark `rul_true` solely for post-hoc evaluation, and test suites verify that injecting corrupted/garbage `rul_true` values yields identical forecasts.

### 3. Forecasting Methodology
- **Engines:** Samples remaining engine life from the predictive conformal quantile distribution using the exact piecewise-linear inverse CDF through $(q_{10}, 0.1)$, $(q_{50}, 0.5)$, and $(q_{90}, 0.9)$. Under the `shikra` policy, engines whose $q_{10}$ reaches $R^*$ within the horizon trigger planned removals.
- **LRUs:** Modeled via conditional Weibull failure probabilities, computing $P(\text{failure in next } k \text{ sorties} \mid \text{survived age } a)$ per installed unit.
- **Groundings & Logistics:** Failures immediately consume available base shelf spares. If unfulfilled, aircraft transition to NMCS/NMCM and enter repair pipelines adhering to repair times, transfer latencies, and hangar capacity limits.
- **Counterfactual Action Ranking:** Evaluates single-step candidate actions (depot-to-base engine/LRU transfers, repair expediting, planned removal deferral, temporary hangar slot addition) using common random numbers against the unmitigated baseline.

### 4. Empirical Calibration Results (`results/readiness_calib.csv`)
Evaluated across 330 operational snapshots (30 random seeds $\times$ 11 snapshot days $D \in \{30, 60, \dots, 330\}$) under the calibrated baseline ($\text{pipeline\_mult} = 1.9$):

| Lead Days | Method | 80% Interval Coverage | MAE (Aircraft) | Brier Score ($P(\ge N \text{ MC})$) | Sample Size ($n$) |
|---|---|---|---|---|---|
| **D+3** | `shikra_forecast` | **0.7263** | 0.8726 | 0.0376 | 330 |
| **D+3** | `persistence` | 0.5798 | **0.5475** | **0.0338** | 330 |
| **D+3** | `climatology` | 0.9283 | 4.8426 | 0.2041 | 330 |
| **D+7** | `shikra_forecast` | **0.6202** | 1.5498 | 0.0741 | 330 |
| **D+7** | `persistence` | 0.4333 | **0.9313** | **0.0596** | 330 |
| **D+7** | `climatology` | 0.9283 | 4.8417 | 0.2031 | 330 |

### 5. Honest Calibration Assessment
- **Comparison to Naive Baselines:**
  - *Persistence:* Assumes status quo ($MC_{D+k} = MC_D$). Because fleet downtime cycles span 10–30 days, aircraft status exhibits substantial short-term autocorrelation, resulting in lower pointwise MAE (0.55 vs 0.87 at D+3) and lower Brier scores during quiescent periods. However, persistence produces degenerate point estimates with poor 80% coverage (0.58 at D+3, 0.43 at D+7) and cannot evaluate counterfactual maintenance actions.
  - *Climatology:* Unconditional squadron MC distributions yield excessive dispersion (MAE ~4.84) and high Brier penalty (~0.20).
  - *`shikra_forecast`:* Delivers calibrated prediction intervals ($0.73$ coverage for nominal $0.80$ band at lead 3) and action sensitivities while maintaining a low Brier score (0.0376).
- **Reliability Diagram:** Stored at `results/fig_readiness_reliability.png`.
- **Illustrative Output:** Stored at `results/readiness_example.csv` and `results/readiness_actions_example.csv` (seed 0, $D = 180$, Squadron S2).

### 6. Limitations
1. **Forecast-Estimated Action Gains:** Action ranking deltas reflect the Monte Carlo forecast model's prospective estimate under sampled failure draws; they are not validated against closed-loop simulation trajectory counterfactuals.
2. **Synthetic Parameters:** Weibull parameters, engine RUL distributions, and flight profiles are derived from synthetic C-MAPSS and configured wing values.
3. **Unit-Level Independence:** LRU wear is tracked per tail and sorties flown, assuming independent component degradation.

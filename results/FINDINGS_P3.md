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

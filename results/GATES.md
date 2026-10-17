
## RUL Model Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.1 — FD001 RMSE ≤ 18.5 | **PASS** | FD001 RMSE = 11.43 (threshold <= 18.5, target <= 13) |
| G0.2 — 90% replay coverage 0.85–0.95 | **PASS** | 90% replay coverage = 0.915 (target 0.85-0.95) |
## Simulation Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.3 — replay units ~40 | **PASS** | replay units = 40 (expected ~40) |
| G0.4 — reactive serviceability 55–65% | **PASS** | reactive serviceability = 60.9% (target 55-65%) |
| G0.5 — oracle gap ≥ 5 pp | **FAIL (finding, not tuned)** | oracle - reactive = -0.0 pp (threshold >= 5 pp) |

**Serviceability per policy (mean over 30 seeds):**

| Policy | Mean serviceability |
|--------|-------------------|
| reactive | 60.9% |
| preventive | 51.8% |
| shikra | 60.5% |
| oracle | 60.9% |
| shikra_v2 | 60.5% |
| oracle_v2 | 60.9% |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  mean = -0.594  95% CI [-3.134, 1.479]
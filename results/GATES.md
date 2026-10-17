
## RUL Model Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.1 — FD001 RMSE ≤ 18.5 | **PASS** | FD001 RMSE = 11.43 (threshold <= 18.5, target <= 13) |
| G0.2 — 90% replay coverage 0.85–0.95 | **PASS** | 90% replay coverage = 0.915 (target 0.85-0.95) |
## Simulation Gates

| Gate | Result |
|------|--------|
| G0.3 - replay units = 40 (expected ~40) | — |
| G0.4 - reactive serviceability = 80.1% (target 55-65%) -> **FAIL** | |
| G0.5 - oracle - reactive = 0.0 pp (threshold >= 5 pp) -> **FAIL (finding, not tuned)** | |

**Serviceability per policy (mean over 30 seeds):**

| Policy | Mean serviceability |
|--------|-------------------|
| reactive | 80.1% |
| shikra | 80.1% |
| oracle | 80.1% |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  mean = nan  95% CI [nan, nan]



## RUL Model Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.1 — FD001 RMSE ≤ 18.5 | **PASS** | FD001 RMSE = 11.64 (threshold <= 18.5, target <= 13) |
| G0.2 — 90% replay coverage 0.85–0.95 | **PASS** | 90% replay coverage = 0.927 (target 0.85-0.95) |
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
| shikra_v2 | 60.6% |
| oracle_v2 | 60.9% |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  mean = -0.890  95% CI [-4.621, 1.685]

> **Finding on G0.5 & Serviceability:** SHIKRA removes all unscheduled engine removals (0 vs 79.6 per year) but, in this wing, serviceability is unchanged (60.6% vs 60.9%), because the engine-spare pipeline, not warning time, is the binding constraint (see A11).
>
> **A12 Supply Shock:** A 180-day halt of foreign supply grounds up to 29 extra aircraft of 54 (15.7 pp over the halt); recovery takes 95 days after supply resumes. Ranking: apu_start (13.6 pp), ecs_valve (3.7 pp). Parameters are synthetic; the method is the claim.
>
> **Systematic Bias Robustness (§3.2b):** −10 cycles wastes 1993 cycles; +10 raises unscheduled removals from 0 to 94.8.
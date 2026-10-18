
## RUL Model Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.1 — FD001 RMSE ≤ 18.5 | **PASS** | FD001 RMSE = 11.64 (threshold <= 18.5, target <= 13) |
| G0.2 — 90% replay coverage 0.85–0.95 | **PASS** | 90% replay coverage = 0.927 (target 0.85-0.95) |
## Simulation Gates

| Gate | Verdict | Detail |
|------|---------|--------|
| G0.3 — replay units ~40 | **PASS** | replay units = 40 (expected ~40) |
| G0.4 — reactive serviceability 55–65% | **PASS** | reactive serviceability = 56.5% (target 55-65%) |
| G0.5 — oracle gap ≥ 5 pp | **FAIL (finding, not tuned)** | oracle - reactive = 0.0 pp (threshold >= 5 pp) |

**Serviceability per policy (mean over 30 seeds):**

| Policy | Mean serviceability |
|--------|-------------------|
| reactive | 56.5% |
| preventive | 47.5% |
| shikra | 56.2% |
| oracle | 56.6% |
| shikra_v2 | 56.2% |
| oracle_v2 | 56.5% |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  mean = 1.742  95% CI [-0.225, 4.896]

> **Finding on G0.5 & Serviceability:** SHIKRA removes all unscheduled engine removals (0 vs 72.1 per year) but, in this wing, serviceability is unchanged (56.2% vs 56.5%), because the engine-spare pipeline, not warning time, is the binding constraint (see A11).
>
> **A12 Supply Shock:** A 180-day halt of foreign supply grounds up to 32 extra aircraft of 54 (28.8 pp over the halt); recovery takes 54 days after supply resumes. Ranking: apu_start (22.0 pp), ecs_valve (8.0 pp). Parameters are synthetic; the method is the claim.
>
> **Systematic Bias Robustness (§3.2b):** −10 cycles wastes 2030 cycles; +10 raises unscheduled removals from 0 to 97.0.
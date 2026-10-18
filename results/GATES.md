
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
| G0.5 — oracle gap ≥ 5 pp | **FAIL (finding, not tuned)** | oracle - reactive = 0.2 pp (threshold >= 5 pp) |

**Serviceability per policy (mean over 30 seeds):**

| Policy | Mean serviceability |
|--------|-------------------|
| reactive | 56.5% |
| preventive | 47.6% |
| shikra | 56.3% |
| oracle (perfect prediction) | 56.6% |
| shikra (ablation: without lead-time-aware prepositioning) | 56.2% |
| oracle (ablation: without lead-time-aware prepositioning) | 56.5% |

**Gap captured (shikra − reactive) / (oracle − reactive):**
  not defined: perfect prediction gives no significant gain at this pipeline speed

> **Finding on G0.5 & Serviceability:** SHIKRA removes all unscheduled engine removals (0 vs 72.7 per year) but, in this wing, serviceability is unchanged (56.3% vs 56.5%), because the engine-spare pipeline, not warning time, is the binding constraint (see A11).
>
> **A12 Supply Shock (at 1.0× pipeline speed):** A 180-day halt of foreign supply grounds up to 32 extra aircraft of 54 (28.1 pp over the halt); recovery takes 45 days after supply resumes. Ranking: apu_start (21.2 pp), ecs_valve (7.9 pp). Parameters are synthetic; the method is the claim.
>
> **Systematic Bias Robustness (§3.2b, at 1.0× pipeline speed):** −10 cycles wastes 2019 cycles; +10 raises unscheduled removals from 0 to 97.8.
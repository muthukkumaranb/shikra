# CP-SAT Hangar Scheduling: Findings

Generated from `results/cpsat_benchmark.csv` and `results/cpsat_wing.csv`.

## 1 — Synthetic Benchmark

**200** random instances (5–16 jobs, 1–3 bays, seed 2026) solved by
OR-Tools CP-SAT (deterministic time ≤ 5.0, 1 worker, seed 2026, EDF hint)
versus earliest-due-first (EDF) baseline.

| Metric | EDF | CP-SAT | Δ (EDF − CP-SAT) | 95 % CI |
|---|--:|--:|--:|---|
| Weighted tardiness | 174.9 | 62.3 | +112.6 | [+87.4, +141.6] |
| Makespan (days) | 26.88 | 25.39 | +1.50 | [+1.27, +1.73] |

CP-SAT wins 170/200 (85%), ties 30/200 (15%), EDF wins 0/200 (0%).
CP-SAT is never worse than EDF (lexicographic guard on weighted tardiness then makespan).

## 2 — Simulated Wing (shikra_v2)

**60** squadron-level snapshots (days 485, 545, 605, 665 × seeds 1–5 × 3 squadrons).
Total bay jobs across all snapshots: **309** (mean **5.15** per squadron).
Each squadron has **3 hangar bays**.

| Metric | EDF | CP-SAT | Δ (EDF − CP-SAT) | 95 % CI |
|---|--:|--:|--:|---|
| Weighted tardiness | 0.0667 | 0.0667 | +0.0000 | [+0.0000, +0.0000] |
| Makespan (days) | 5.4333 | 5.4333 | +0.0000 | [+0.0000, +0.0000] |

Outcome: **60/60 ties** (100%), 0/60 CP-SAT wins.
Only **3/60** snapshots have any tardiness at all.

With ~5 jobs per squadron and 3 available bays, EDF already produces
near-optimal or optimal schedules. The solver confirms this: every snapshot
is a tie.

## 3 — Conclusion

**Hangar bays are not the bottleneck in this wing; engine spares are.**

The simulation consistently shows ~5 engines queued per squadron against
3 hangar bays. At this load, even the simplest EDF rule completes all
maintenance within the 7-day due window in 57/60 snapshots, and the
remaining 3 carry negligible tardiness.
CP-SAT adds nothing because there is nothing to optimise: the constraint
is waiting for a spare engine to arrive from depot (transfer pipeline),
not sequencing jobs through bays.

CP-SAT remains valuable as insurance: if fleet policy, bay count, or
maintenance tempo changes, the solver will find improvements that EDF
cannot.

Hangar bays are not the constraint in this wing. Aircraft wait for engines and LRUs
from the repair and supply pipeline; per results/heatmap.csv, pipeline speed is the
dominant lever, and spares re-allocation (results/spares_compare.csv) adds little.
(Across pipeline speeds 0.5× to 1.9×, serviceability spans 88.3% down to
52.9%, whereas spares re-allocation yields at most +0.45 pp
gain under shikra_v2.)

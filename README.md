# SHIKRA

> **Prediction, spares or depots: which buys aircraft readiness, and what should India make itself first?**
> SHIKRA is offline, advisory decision support evaluated on NASA public data and a synthetic wing.

SHIKRA (Smart Heuristic for Integrated Knowledge-based Readiness Assessment) is an entry for
Smart India Hackathon 2026, problem statement **SIH26249** — "Air Power – Predictive Maintenance
& Fleet Availability" (Ministry of Defence / Defence Services Staff College).

The system advises **when** to proactively remove engines for maintenance and **which spares** to
pre-position, using calibrated quantile RUL predictions from the NASA C-MAPSS dataset evaluated
against a synthetic 54-aircraft wing.

---

## Results

<!-- AUTO-GENERATED — do not edit by hand; run `python scripts/readme_table.py` -->

## Results

### RUL Model (LightGBM quantile ensemble, 5 seeds)

| Subset | RMSE | MAE | NASA Score | RMSE (RUL≤30) | n units |
|--------|------|-----|-----------|---------------|---------|
| FD001 | 11.43 | 8.29 | 227.4 | 3.47 | 100 |
| FD003 | 14.34 | 9.66 | 588.7 | 2.63 | 100 |

### Fleet Simulation (synthetic wing, 30 replications)

| Policy | Serviceability | 95% CI | Unscheduled removals |
|--------|---------------|--------|---------------------|
| Reactive | 60.9% | [60.4%, 61.4%] | 79.6 |
| Preventive | 51.8% | [51.4%, 52.3%] | 13.5 |
| Shikra | 60.5% | [59.9%, 61.0%] | 0.0 |
| Oracle | 60.9% | [60.4%, 61.4%] | 0.0 |
| Shikra_v2 | 60.5% | [60.0%, 61.1%] | 0.0 |
| Oracle_v2 | 60.9% | [60.4%, 61.5%] | 0.0 |

> **Key Finding:** SHIKRA removes all unscheduled engine removals (0 vs 79.6 per year) but, in this wing, serviceability is unchanged (60.5% vs 60.9%), because the engine-spare pipeline, not warning time, is the binding constraint (see A11).

> **A12 Supply Shock:** A 180-day halt of foreign supply grounds up to 29 extra aircraft of 54 (15.7 pp over the halt); recovery takes 95 days after supply resumes. Ranking: apu_start (13.6 pp), ecs_valve (3.7 pp). Parameters are synthetic; the method is the claim.

> **Systematic Bias Robustness (§3.2b):** −10 cycles wastes 1993 cycles; +10 raises unscheduled removals from 0 to 94.8.

---

## Quickstart

```bash
make setup          # create .venv and install requirements
make data           # download C-MAPSS, verify checksums
make reproduce      # run the full pipeline (split → rul → replay → params → sim → summary → figures)
```

---

## Repository layout

```
shikra/          Python package (data, rul, sim, spares, strategy, ingest, nlp)
config/          wing.yaml — synthetic wing parameters
artifacts/       frozen artefacts: splits.json, policy_params.json, replay_rul.parquet, model files
results/         every CSV/figure committed here
tests/           pytest suite (no raw data needed in CI)
scripts/         utility scripts (readme_table.py, …)
DATA_CARD.md     data sources, checksums, licences, all synthetic parameters
LIMITATIONS.md   honest limitations
```

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).

---

## Citation

If you use this work, please cite:

```
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
Damage propagation modeling for aircraft engine run-to-failure simulation.
2008 International Conference on Prognostics and Health Management.
```

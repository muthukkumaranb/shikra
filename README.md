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
| Reactive | 61.6% | [61.0%, 62.1%] | 80.2 |
| Preventive | 52.7% | [52.2%, 53.2%] | 13.7 |
| Shikra | 61.3% | [60.8%, 61.8%] | 0.0 |
| Oracle | 61.6% | [61.1%, 62.2%] | 0.0 |

--------|------|-----|-----------|---------------|---------|
| FD001 | 11.61 | 8.30 | 237.0 | 3.62 | 100 |
| FD003 | 14.16 | 9.71 | 490.8 | 2.41 | 100 |

### Fleet Simulation (synthetic wing, 30 replications)

| Policy | Serviceability | 95% CI | Unscheduled removals |
|--------|---------------|--------|---------------------|
| Reactive | 80.1% | [79.8%, 80.4%] | 94.6 |
| Preventive | 80.1% | [79.8%, 80.4%] | 19.4 |
| Shikra | 80.1% | [79.8%, 80.4%] | 0.0 |
| Oracle | 80.1% | [79.8%, 80.4%] | 0.0 |

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

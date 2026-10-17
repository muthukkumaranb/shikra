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
<!-- results table will appear here after make reproduce -->

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

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

| Subset | Babu 2016 CNN | SHIKRA RMSE | MAE | NASA Score | RMSE (RUL≤30) | 80% Coverage (uncalibrated) | n units |
|--------|---------------|-------------|-----|------------|---------------|-----------------------------|---------|
| FD001 | 18.45 | 11.43 | 8.29 | 227.4 | 3.47 | 62.0% | 100 |
| FD002 | 30.29 | 13.45 | 9.47 | 7291.7 | 3.49 | 66.8% | 259 |
| FD003 | 19.82 | 14.34 | 9.66 | 588.7 | 2.63 | 55.0% | 100 |
| FD004 | 29.16 | 14.33 | 9.48 | 4446.0 | 7.39 | 55.6% | 248 |

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

### 1. Environment Setup

```bash
# Linux / macOS
make setup

# Windows (PowerShell)
python -m venv .venv
.venv/Scripts/pip install --upgrade pip
.venv/Scripts/pip install -r requirements.txt
```

### 2. Reproduction from Committed Artifacts (No Raw Data Needed)

To reproduce all simulation results from committed artifacts (`artifacts/`):

**Linux / macOS:**
```bash
make reproduce-sim PY=python
python scripts/check_reproduce.py
```

**Windows (PowerShell):**
```powershell
make reproduce-sim PY=.venv/Scripts/python
.venv/Scripts/python scripts/check_reproduce.py
```

#### Measured Runtimes per Target

| Target | Description | Measured Runtime | Output Artifacts |
|--------|-------------|------------------|------------------|
| `sim` | Fleet simulation (6 policies, 30 seeds) | ~18.5s | `results/policies.csv` |
| `summary` | Bootstrapped KPI summary & gates | ~3.5s | `results/policy_summary.csv` |
| `heatmap` | Pipeline multiplier lever sweep & bootstrap | ~2.0s | `results/heatmap.csv`, `results/fig_lever_map.png` |
| `bias` | Systematic RUL bias sweep (§3.2b, 150 runs) | ~32.1s | `results/bias.csv`, `results/fig_bias.png` |
| `shock` | Supply-shock resilience & recovery (545 days, 30 seeds) | ~65.0 min | `results/supply_shock.csv`, `results/fig_supply_shock.png` |
| `figures` | Policy comparison charts | ~1.3s | `results/fig_policies.png` |
| `readme` | Auto-generate README / GATES summary tables | ~0.7s | `README.md`, `GATES.md` |
| **`reproduce-sim`** | **Full simulation reproduction** | **~66 min** | All committed CSVs & figures |

> **CI Quick Mode (`reproduce-sim-quick`):** Because `shock` runs 60 full 545-day simulations and takes ~65 minutes (exceeding standard CI time budgets), CI runs in `--quick` mode:
> ```bash
> make reproduce-sim-quick PY=python
> python scripts/check_reproduce.py --quick
> ```
> This executes `sim` + `summary` + `bias` in ~54 seconds and validates `policies.csv`, `policy_summary.csv`, and `bias.csv`.

### 3. Full Pipeline Reproduction (Requires C-MAPSS)

```bash
make data       # download C-MAPSS FD001–FD004
make reproduce  # full pipeline: split → rul → replay → params → sim → summary → figures
```

### 4. Offline Containerized Reproduction (Docker)

To verify full offline reproducibility with zero network access:

```bash
# Build offline Docker image
docker build -t shikra .

# Run container completely air-gapped
docker run --network none shikra
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

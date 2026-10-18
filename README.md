# SHIKRA

> **Prediction, spares or depots: which buys aircraft readiness, and what should India make itself first?**
> SHIKRA is offline, advisory decision support evaluated on NASA public data and a synthetic wing.

SHIKRA is named after the shikra, a small Indian hawk. It is an entry for
Smart India Hackathon 2026, problem statement **SIH26249** — "Air Power — Predictive Maintenance
& Fleet Availability" (Ministry of Defence / Defence Services Staff College).

The system advises **when** to proactively remove engines for maintenance and **which spares** to
pre-position, using calibrated quantile RUL predictions from the NASA C-MAPSS dataset evaluated
against a synthetic 54-aircraft wing.

---

## In 60 seconds

- **What it is**: Offline, advisory decision support for maintenance scheduling and spares prepositioning; advisory only, never autonomous.
- **The data**: Open NASA C-MAPSS turbofan run-to-failure benchmarks (FD001–FD004) paired with a synthetic 54-aircraft wing simulation evaluated under common random numbers (CRN).
- **Which lever pays?**: Warning time alone cannot overcome saturated depot pipelines; spares inventory depth and pipeline throughput are the binding constraints on readiness.
- **How much gain does prediction capture?**: Eliminates all unscheduled engine removals, but captures fleet serviceability gains only when maintenance turnaround is fast enough to exploit lead time (see Fleet Simulation table below).
- **Which part to indigenise first?**: Under a 180-day foreign supply halt scenario, LRU vulnerability rankings identify the APU starter and ECS valve as highest-priority components for domestic manufacturing (see A12 Supply Shock table below).

---

<!-- AUTO-GENERATED — do not edit by hand; run `python scripts/readme_table.py` -->

## Results

### RUL Model (LightGBM quantile ensemble, 5 seeds)

Official test sets, last cycle only; RMSE shown against true RUL capped at 125 (piecewise convention) and uncapped.

| Subset | Babu 2016 CNN | RMSE (capped 125) | RMSE (uncapped) | MAE (capped 125) | MAE (uncapped) | NASA Score (capped 125) | NASA Score (uncapped) | RMSE (RUL≤30) | 80% Coverage (uncalibrated) | n units |
|--------|---------------|-------------------|-----------------|------------------|----------------|-------------------------|-----------------------|---------------|-----------------------------|---------|
| FD001 | 18.45 | 11.64 | 12.95 | 8.40 | 9.47 | 204.3 | 235.4 | 3.37 | 61.0% | 100 |
| FD002 | 30.29 | 13.45 | 25.47 | 9.47 | 16.96 | 975.8 | 7291.7 | 3.49 | 66.8% | 259 |
| FD003 | 19.82 | 14.50 | 15.74 | 9.83 | 11.39 | 578.4 | 618.4 | 2.60 | 57.0% | 100 |
| FD004 | 29.16 | 14.33 | 25.99 | 9.48 | 18.17 | 1151.2 | 4446.0 | 7.39 | 55.6% | 248 |

### Fleet Simulation (synthetic wing, 30 replications)

| Policy | Serviceability | 95% CI | Unscheduled removals |
|--------|---------------|--------|---------------------|
| Reactive | 56.5% | [56.0%, 56.9%] | 72.7 |
| Preventive | 47.6% | [47.2%, 48.1%] | 13.2 |
| SHIKRA | 56.3% | [55.7%, 56.8%] | 0.0 |
| Oracle (perfect prediction) | 56.6% | [56.1%, 57.2%] | 0.0 |
| SHIKRA (ablation: without lead-time-aware prepositioning) | 56.2% | [55.6%, 56.8%] | 0.0 |
| Oracle (ablation: without lead-time-aware prepositioning) | 56.5% | [56.0%, 57.1%] | 0.0 |

> **Key Finding:** SHIKRA removes all unscheduled engine removals (0 vs 72.7 per year) but, in this wing, serviceability is unchanged (56.3% vs 56.5%), because the engine-spare pipeline, not warning time, is the binding constraint (see A11).

> **A12 Supply Shock (at 1.0× pipeline speed):** A 180-day halt of foreign supply grounds up to 32 extra aircraft of 54 (28.1 pp over the halt); recovery takes 45 days after supply resumes. Ranking: apu_start (21.2 pp), ecs_valve (7.9 pp). Parameters are synthetic; the method is the claim.

> **Systematic Bias Robustness (§3.2b, at 1.0× pipeline speed):** −10 cycles wastes 2019 cycles; +10 raises unscheduled removals from 0 to 97.8.

> **A10 Readiness Forecast:** At lead 7, forecast MAE is 2.274 vs 1.552 for persistence and Brier is 0.132 vs 0.117; action gains are forecast-estimated and not validated in simulation.

---

## Quickstart

## Judge quickstart

Full reproducibility validation in a single command:

**Linux / macOS:**
```bash
git clone https://github.com/muthukkumaranb/shikra.git
cd shikra
python -m venv .venv
.venv/bin/pip install -r requirements.txt
make verify PY=.venv/bin/python
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/muthukkumaranb/shikra.git
cd shikra
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/verify.ps1 -Py .venv\Scripts\python.exe
```

#### Measured Benchmark Runtimes (Clean Fresh Clone)
- **`git clone`**: ~34.5s
- **`python -m venv .venv`**: ~3.2s (Python 3.11.16)
- **`pip install -r requirements.txt`**: ~4m 26s (from wheel cache)
- **`pytest -q -m "not data"`**: ~55.5s (68 passed, 1 skipped, 4 deselected)
- **`make verify` / `verify.ps1`**: Executes test suite, full simulation reproduction, bitwise verification against 13 committed CSVs, and git working tree diff check in one command.

---

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
| `readme` | Auto-generate README / GATES summary tables | ~0.7s | `README.md`, `results/GATES.md` |
| **`reproduce-sim`** | **Full simulation reproduction** | **~66 min** | All committed CSVs & figures |

> **CI Quick Mode (`reproduce-sim-quick`):** Because `shock` runs 60 full 545-day simulations and takes ~65 minutes (exceeding standard CI time budgets), CI runs in `--quick` mode:
> ```bash
> make reproduce-sim-quick PY=python
> python scripts/check_reproduce.py --quick
> ```
> This executes `sim` + `summary` + `bias` in ~54 seconds and validates `results/policies.csv`, `results/policy_summary.csv`, and `results/bias.csv`.

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

## What is not claimed

- **Synthetic wing, not IAF data**: All fleet structures, flight schedules, failure rates, and spares budgets represent a synthetic 54-aircraft wing for research and demonstration; no classified or proprietary IAF operational data is used (see [DATA_CARD.md](DATA_CARD.md)).
- **Advisory, not autonomous**: SHIKRA provides offline recommendations and risk bounds to assist maintenance engineers; it does not issue automated dispatch directives or replace maintenance authority (see [LIMITATIONS.md](LIMITATIONS.md)).
- **Readiness forecast does not beat persistence**: Multi-step readiness forecasting models evaluated in development did not outperform a simple persistence baseline (see [LIMITATIONS.md](LIMITATIONS.md)).
- **LSTM did not beat LightGBM**: The 2-layer temporal attention-LSTM comparison member achieved comparable RMSE but ~2x worse NASA penalty score on FD003 compared to the LightGBM quantile ensemble of record (see [FINDINGS_T14.md](results/FINDINGS_T14.md)).
- **Hangar scheduling showed no gain on the simulated wing**: Opportunistic hangar co-scheduling heuristics yielded no statistically significant availability gain over standard threshold-based removals on the simulated wing structure (see [LIMITATIONS.md](LIMITATIONS.md)).

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

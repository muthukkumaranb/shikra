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

### Environment and Python Requirement
- **Python Version**: **Python 3.11.x strictly required** (tested and verified on CPython 3.11.16).
- *Compatibility Finding*: Python 3.13 and 3.14 fail during installation because pre-built wheels for pinned packages (`numpy==1.26.4`, `scipy==1.13.1`, `lightgbm==4.3.0`) are not available on PyPI for 3.13+; pip falls back to building `numpy-1.26.4.tar.gz` from source, which fails without configured MSVC C/C++ and Fortran compilers (the optional LSTM extras, `requirements-lstm.txt`, add torch).

### Verification Scope & Commands

- **`scripts/verify.ps1` (default, ~8 min)**: Windows one-line validation executing the full test suite (`pytest -q -m "not data"`), every simulation (`sim`, `summary`, `heatmap`, `bias`, `figures`, `deck-figures`, `readme`) EXCEPT the ~65-min supply-shock run, followed by a bitwise byte-compare of ALL 17 committed results CSVs (`scripts/check_reproduce.py`) and a git working tree diff check. Add `-Full` to also regenerate supply shock (~73 min).
- **`make verify` (~70+ min)**: Full pipeline reproduction including the 545-day supply-shock simulation (`reproduce-sim`), followed by bitwise validation against all committed CSVs and git diff verification.

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

#### Measured Benchmark Runtimes (Clean Fresh Clone, Python 3.11.16)
- **`git clone`**: ~16.3s
- **`python -m venv .venv`**: ~26.6s
- **`pip install -r requirements.txt`**: ~4m 13.6s (253.6s)
- **`pytest -q -m "not data"`**: ~48.5s (full test suite excluding raw data)
- **`shikra.sim.runner` (30 seeds, 6 policies, `n_jobs=2`)**: ~2m 40.6s (160.6s)
- **`scripts/verify.ps1` (default)**: ~7m 51.5s (471.5s) (tests + sim + summary + heatmap + bias + figures + deck figures + readme + byte-compare of all 17 CSVs)
- **`scripts/verify.ps1 -Full` / `make verify`**: ~73 min (includes ~65 min 545-day supply shock run)

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
- **Readiness forecast does not beat persistence**: Multi-step readiness point forecasting models evaluated in development did not outperform a simple persistence baseline (see [FINDINGS_A10.md](results/FINDINGS_A10.md)).
- **Attention-LSTM did not beat LightGBM**: The 2-layer temporal attention-LSTM comparison member achieved comparable RMSE but ~2x worse NASA penalty score on FD003 compared to the LightGBM quantile ensemble of record (see [FINDINGS_T14.md](results/FINDINGS_T14.md)).
- **CP-SAT gave no gain on the simulated wing**: Exact constraint programming (OR-Tools CP-SAT) for hangar co-scheduling heuristics yielded no availability gain over standard threshold-based removals on the simulated wing structure ([FINDINGS_CPSAT.md](results/FINDINGS_CPSAT.md), once merged; link anyway).
- **Tamper-evident, not tamper-proof audit log**: The SHA-256 HMAC cryptographic chain guarantees tamper-evidence and detection of post-hoc modifications, but is not tamper-proof against root adversaries because the signing key is stored locally in deployment (see [LIMITATIONS.md](LIMITATIONS.md)).

---

## Repository map

- `shikra/`: Core Python library containing data loaders, RUL estimation, discrete-event fleet simulation, spares allocation, and strategy.
- `app/`: Streamlit interactive offline decision-support dashboard (Commander, Engineering Officer, Logistics, Technician, Scenario, and Evidence pages).
- `artifacts/`: Frozen reproducibility artifacts (unit splits, RUL models, baseline replay parquet, calibration parameters).
- `config/`: Simulation configuration (`wing.yaml`) specifying synthetic wing parameters, LRU Weibull rates, and squadron structure.
- `data/`: Ingestion pipelines and mock enterprise extracts (`emms`, `immols`, `hums`, `brd`).
- `results/`: Committed benchmark CSVs, validation gates, and analytical evaluation figures.
- `scripts/`: Operational utilities for model training, reproduction verification, deck generation, and README table updates.
- `tests/`: Pytest test suite covering simulation mechanics, RUL variants, audit chaining, and application components.
- `results/FINDINGS_A10.md`: Conformal quantile regression (CQR) readiness forecasting findings vs persistence baseline.
- `results/FINDINGS_T14.md`: Performance and penalty evaluation of Attention-LSTM comparison vs LightGBM ensemble.
- `results/FINDINGS_T16.md`: Two-echelon VARI-METRIC spares inventory optimization findings and consistency checks.
- `results/FINDINGS_CPSAT.md`: Constraint programming (CP-SAT) hangar schedule evaluation (link pre-registered).

---

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).

---

## Further reading

- [DRDO ETAI Trustworthy AI Mapping](docs/ETAI_MAPPING.md): Engineering self-assessment against DRDO's Evaluating Trustworthy AI principles.
- [Integration Architecture & Pilot Specification](docs/INTEGRATION.md): How SHIKRA interfaces with e-MMS/IMMOLS exports and proposal for a minimal 90-day shadow pilot.
- [AI Assistance Disclosure](AI_USAGE.md): Transparent disclosure of AI tools, verification protocols, failure types caught, and team design decisions.
- [Limitations & Operational Scope](LIMITATIONS.md): Hard boundaries, synthetic assumptions, and safety constraints.
- [Model Card](MODEL_CARD.md): LightGBM quantile ensemble architecture, training procedure, and conformal calibration.
- [Data Card](DATA_CARD.md): NASA C-MAPSS degradation benchmark structure and synthetic wing operational parameters.
- [Negative Findings & Technical Memos](results/):
  - [FINDINGS_A10.md](results/FINDINGS_A10.md): Conformal readiness forecasting vs persistence baseline.
  - [FINDINGS_T14.md](results/FINDINGS_T14.md): Temporal Attention-LSTM comparison vs LightGBM ensemble.
  - [FINDINGS_T16.md](results/FINDINGS_T16.md): Two-echelon VARI-METRIC spares inventory optimization findings.
  - [FINDINGS_CPSAT.md](results/FINDINGS_CPSAT.md): Exact constraint programming (CP-SAT) hangar schedule evaluation.

---

## Citation

If you use this work, please cite:

```
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
Damage propagation modeling for aircraft engine run-to-failure simulation.
2008 International Conference on Prognostics and Health Management.
```

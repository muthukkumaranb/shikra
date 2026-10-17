# SHIKRA Unattended Run Log (Sat 17 Oct, Evening)

## Block 0: A12 Supply Shock & Verification
- Commits already present on `main` and verified:
  - `9f077a6 fix(sim): held repairs restart at end of supply halt`
  - `ccefd7b fix(runner): scalar KPIs only`
  - `c530b4e results(a12): shock-window loss, peak grounded, sustained recovery`
  - `abba367 docs: regenerate A12 and bias lines`
- Block 0 verified complete; proceeding to Block 1.

## Block 1: One-Command Reproducibility (T1.9)
- **Makefile:** Converted to cross-platform format with configurable `PY ?= python`, `PIP ?= pip`, `PYTEST ?= pytest`, `RUFF ?= ruff`. Added targets `shock`, `bias`, `heatmap`, `readme`, `reproduce-sim`, and `reproduce-sim-quick`.
- **Measured execution times (Windows 11, Intel Core i7 / 12-thread):**
  - `sim`: 18.46s
  - `summary`: 3.47s
  - `heatmap`: 2.01s
  - `bias`: 32.07s
  - `shock`: 3901.78s (~65.0 min; 545-day paired baseline + shock across 30 seeds)
  - `figures`: 1.28s
  - `readme`: 0.73s
  - `reproduce-sim` (full sim pipeline): ~3960s (~66 min)
  - `reproduce-sim-quick` (`sim` + `summary` + `bias`): 53.94s
  - `test` (`pytest -q -m "not data"`): 8.78s
  - `lint` (`ruff check .`): 0.15s
- **CI reproduction job:**
  - Full `shock` target takes ~65 min, exceeding the 30-minute GitHub Actions limit. As instructed, added `--quick` mode used for CI: runs `sim` + `summary` + `bias` (`make reproduce-sim-quick PY=python`) and checks `policies.csv`, `policy_summary.csv`, and `bias.csv` with `python scripts/check_reproduce.py --quick`. Total CI runtime is ~1 minute.
  - `gh run list --limit 3`: `gh` CLI is not installed on this Windows host.
- **Docker & Compose:**
  - Created `Dockerfile` (based on `python:3.11-slim`, installs `make` and `git`, copies codebase, default command reproduces and checks results).
  - Created `docker-compose.yml` (`network_mode: none`, `./data` mounted read-only at `/app/data`).
  - Docker daemon check: Docker CLI 29.8.0 is present, but Docker Desktop service is stopped and cannot be started without administrative privileges. Logged: `Docker build not run locally (Docker not installed); verified by CI only`.

## Block 2: FD002 / FD004 Multi-Condition RUL (T1.5)
- **Clustering & Normalisation:** Fitted KMeans ($k=6$, seed 2026) on `op1, op2, op3` using training units only. Standardized sensors per cluster using training statistics.
- **Causal History Feature:** Added cumulative cycle counts per operating condition (`cycles_in_cond_0` .. `cycles_in_cond_5`), verified strictly causal (depends only on cycles $\le c$).
- **Frozen Multi-Split:** Generated engine-disjoint 80/20 train/validation split (seed 2026) in `artifacts/splits_multi.json` (FD002: 208 train, 52 val; FD004: 199 train, 50 val). Frozen `splits.json` was kept untouched.
- **Evaluation on Official Test Sets (Last Cycle Only):**
  - **FD002:** RMSE = 13.45 (vs Babu 2016 CNN: 30.29), MAE = 9.47, NASA score = 7291.7, critical RMSE ($\le 30$) = 3.49, $n=259$, uncalibrated 80% coverage = 66.8%.
  - **FD004:** RMSE = 14.33 (vs Babu 2016 CNN: 29.16), MAE = 9.48, NASA score = 4446.0, critical RMSE ($\le 30$) = 7.39, $n=248$, uncalibrated 80% coverage = 55.6%.
  - Both subsets outperformed the Babu (2016) CNN benchmark on a single hyperparameter set chosen on validation only, without test set tuning.
- **Output:** Created `results/rul_cmapss_multi.csv` and updated `scripts/readme_table.py` to publish the combined 4-subset table in `README.md`.

## Block 3: Model Card and Limitations (T1.11)
- **MODEL_CARD.md:** Generated entirely from committed CSVs via `scripts/cards.py` with zero hand-typed numbers. Covers:
  - Intended advisory decision support use in air-gapped deployment;
  - Training splits ({n_tr_single} engines in single-condition wing model, 208/199 in multi-condition models);
  - Causal rolling features and condition-history tracking;
  - Conformal calibration empirical coverage and widths;
  - Test set results per subset compared against Babu (2016) CNN;
  - Robustness failure modes (systematic bias and 2× error heatmap impact);
  - Strict data isolation (unseen true RUL and 40 held-out replay units).
- **LIMITATIONS.md:** Refreshed to transparently articulate:
  - Synthetic nature of the 54-aircraft wing and base/depot parameters;
  - Purely reactive status of all non-engine LRUs across all policies;
  - Lack of multi-echelon engine spare optimization (VARI-METRIC);
  - Empirical failure of Gate G0.5 (oracle − reactive $\le 1.3$ pp, $+0.0$ pp nominal) due to the engine-spare pipeline bottleneck;
  - Dependence of foreign supply-shock rankings on synthetic initial buffers;
  - Distinction between the C-MAPSS simulator and Su-30MKI / AL-31FP operational flight regimes.
- **Vocabulary check: passed.**

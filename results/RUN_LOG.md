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

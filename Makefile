.PHONY: setup data split rul replay params sim summary heatmap bias shock figures readme cards test lint reproduce reproduce-sim reproduce-sim-quick ingest

PY     ?= python
PIP    ?= pip
PYTEST ?= pytest
RUFF   ?= ruff

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	$(PY) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment ready."

# ── Data download & verification ─────────────────────────────────────────────
data:
	$(PY) scripts/download_cmapss.py

# ── Frozen unit split ────────────────────────────────────────────────────────
split:
	$(PY) -m shikra.data.split

# ── RUL model training + evaluation ──────────────────────────────────────────
rul:
	$(PY) -m shikra.rul.evaluate

# ── Replay RUL table ─────────────────────────────────────────────────────────
replay:
	$(PY) -m shikra.rul.replay

# ── Policy parameters ────────────────────────────────────────────────────────
params:
	$(PY) -m shikra.rul.policy_params

# ── Fleet simulation ─────────────────────────────────────────────────────────
sim:
	$(PY) -m shikra.sim.runner

# ── Summary statistics and gates ─────────────────────────────────────────────
summary:
	$(PY) scripts/summary.py

# ── Heatmap (lever map) ──────────────────────────────────────────────────────
heatmap:
	$(PY) scripts/heatmap.py

# ── Bias sweep ───────────────────────────────────────────────────────────────
bias:
	$(PY) scripts/bias.py


# ── Supply shock ─────────────────────────────────────────────────────────────
shock:
	$(PY) scripts/supply_shock.py

# ── Figures ──────────────────────────────────────────────────────────────────
figures:
	$(PY) scripts/figures.py

# ── README table ─────────────────────────────────────────────────────────────
readme:
	$(PY) scripts/readme_table.py

# ── Model card ───────────────────────────────────────────────────────────────
cards:
	$(PY) scripts/cards.py

# ── Ingestion & DQ checks ──────────────────────────────────────────────────
ingest:
	$(PY) scripts/ingest.py

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PY) -m pytest -q -m "not data"

# ── Lint ─────────────────────────────────────────────────────────────────────
lint:
	$(PY) -m ruff check .

# ── Full reproduction (after data) ───────────────────────────────────────────
reproduce: split rul replay params sim summary heatmap bias shock figures readme
	@echo "Full pipeline complete. Results are in results/"

# ── Simulation reproduction (no raw data needed) ─────────────────────────────
reproduce-sim: sim summary heatmap bias shock figures readme
	@echo "Simulation reproduction complete. Results are in results/"

# ── Quick simulation reproduction (CI mode) ──────────────────────────────────
reproduce-sim-quick: sim summary bias
	@echo "Quick simulation reproduction complete."


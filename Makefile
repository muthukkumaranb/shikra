.PHONY: setup data split rul replay params sim summary figures test lint reproduce

PYTHON := .venv/Scripts/python
PIP    := .venv/Scripts/pip

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment ready."

# ── Data download & verification ─────────────────────────────────────────────
data:
	$(PYTHON) scripts/download_cmapss.py

# ── Frozen unit split ────────────────────────────────────────────────────────
split:
	$(PYTHON) -m shikra.data.split

# ── RUL model training + evaluation ──────────────────────────────────────────
rul:
	$(PYTHON) -m shikra.rul.evaluate

# ── Replay RUL table ─────────────────────────────────────────────────────────
replay:
	$(PYTHON) -m shikra.rul.replay

# ── Policy parameters ────────────────────────────────────────────────────────
params:
	$(PYTHON) -m shikra.rul.policy_params

# ── Fleet simulation ─────────────────────────────────────────────────────────
sim:
	$(PYTHON) -m shikra.sim.runner

# ── Summary statistics and gates ─────────────────────────────────────────────
summary:
	$(PYTHON) scripts/summary.py

# ── Figures ──────────────────────────────────────────────────────────────────
figures:
	$(PYTHON) scripts/figures.py

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	.venv/Scripts/pytest -q --ignore=tests/fixtures

# ── Lint ─────────────────────────────────────────────────────────────────────
lint:
	.venv/Scripts/ruff check .

# ── Full reproduction (after data) ───────────────────────────────────────────
reproduce: split rul replay params sim summary figures
	@echo "Full pipeline complete. Results are in results/"

# ── README table ─────────────────────────────────────────────────────────────
readme:
	$(PYTHON) scripts/readme_table.py

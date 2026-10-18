# JUDGE READINESS AUDIT REPORT

**Date:** 2026-10-18  
**Branch:** `t2/judge-ready`  
**Base Commit (origin/main):** `ef9cad8`  
**Status:** All verification targets passed; negative tests verified; pushed to origin.

---

## 1. Git Commit Trajectory

`git log --oneline origin/main..HEAD`:
```
4e104b8 docs(readme): ensure Quickstart header precedes Judge quickstart for table script
e636dc7 build: add pre-execution check to verify target and scripts/verify.ps1
8953f87 build: make verify.ps1 memory-safe on windows and add deck_figures
ed7bebb docs(readme): add Judge quickstart, non-claims, and repository map
5c2ea97 build: declare supported Python versions
5363348 chore: add *.webm to .gitignore
47c449f merge: merge origin/main into t2/judge-ready
620c876 docs(readme): add 60-second summary, judge quickstart, and non-claims
91f8016 docs(datacard): fix relative paths for synthetic mock exports
2eb7125 chore: add audit.key to .gitignore
2d801be build: add verify target to Makefile and scripts/verify.ps1
```

---

## 2. Environment & Python Compatibility

- **Python Version Used:** `CPython 3.11.16` (64-bit Windows)
  Path: `C:\Users\MUTHUKUMARAN\AppData\Roaming\uv\python\cpython-3.11.16-windows-x86_64-none\python.exe` (accessible via `py -V:Astral\CPython3.11.16` or `python` within `.venv`).
- **Python Version Compatibility Finding:**
  - **Python 3.11.x is strictly required.**
  - **Python 3.13 and Python 3.14 fail during installation:**
    ```
    Collecting numpy==1.26.4 (from -r requirements.txt (line 1))
      Using cached numpy-1.26.4.tar.gz (15.8 MB)
      Installing build dependencies ...
    ```
    PyPI does not provide pre-compiled wheels for `numpy==1.26.4`, `scipy==1.13.1`, or `lightgbm==4.3.0` for Python 3.13+ (the optional LSTM extras, `requirements-lstm.txt`, add torch). Pip falls back to building `numpy-1.26.4.tar.gz` from source, which fails without configured MSVC C/C++ and Fortran toolchains.

---

## 3. Measured Benchmark Runtimes (Clean Fresh Clone to `D:\judge-test-2`)

| Operation | Command | Measured Runtime | Notes |
|---|---|---|---|
| **Git Clone** | `git clone https://github.com/muthukkumaranb/shikra.git D:\judge-test-2` | **16.35s** | Fresh clone from GitHub |
| **Virtualenv Creation** | `& python.exe -m venv D:\judge-test-2\.venv` | **26.65s** | CPython 3.11.16 |
| **Dependency Installation** | `pip install -r requirements.txt` | **253.61s** (4m 13.6s) | Complete installation from cache |
| **Test Suite** | `pytest -q -m "not data"` | **85.92s** (1m 25.9s) | 88 passed, 5 deselected on `t2/judge-ready` |
| **Fleet Simulation** | `sim.runner (n_jobs=2)` | **160.63s** (2m 40.6s) | 180 runs (6 policies × 30 seeds) |
| **Summary & Gates** | `python scripts/summary.py` | **3.47s** | Generates `policy_summary.csv` |
| **Lever Heatmap** | `python scripts/heatmap.py` | **39.82s** | 180 seed-cells across 7 multipliers |
| **Systematic Bias Sweep** | `python scripts/bias.py` | **32.07s** | 150 runs across 5 bias levels |
| **Policy Figures** | `python scripts/figures.py` | **1.28s** | Policy comparisons |
| **Deck Figures** | `python scripts/deck_figures.py` | **2.15s** | 4 deck figures & manifest |
| **README Table Update** | `python scripts/readme_table.py` | **0.73s** | Regenerates markdown table |
| **Full Verify Pipeline** | `powershell -File scripts/verify.ps1` | **471.49s** (7m 51.5s) | All 4 steps clean execution |

---

## 4. Verification & Reproducibility Results

### Clean Run Output (`scripts/verify.ps1`)
```
== Step 0/4: Validating artifact baseline before execution ==
REPRODUCED: 17 files identical
== Step 1/4: Running pytest ==
88 passed, 5 deselected, 14 warnings in 48.47s
== Step 2/4: Reproducing fleet simulations and summary artifacts ==
--- sim.runner (n_jobs=2) ---
Results written -> D:\shikra\results\policies.csv  (180 rows)
--- summary.py ---
Summary written -> D:\shikra\results\policy_summary.csv
Gates written -> D:\shikra\results\GATES.md
  G0.3: PASS - replay units = 40 (expected ~40)
  G0.4: PASS - reactive serviceability = 56.5% (target 55-65%)
  G0.5: FAIL (finding, not tuned) - oracle - reactive = 0.2 pp (threshold >= 5 pp)
  gap_captured: not defined: perfect prediction gives no significant gain at this pipeline speed
--- heatmap.py ---
Running A11 lever map (180 seed-cells) across 11 workers …
Heatmap data saved -> D:\shikra\results\heatmap.csv
Heatmap figure saved -> D:\shikra\results\fig_heatmap.png
Lever map figure saved -> D:\shikra\results\fig_lever_map.png
--- bias.py ---
Running bias sensitivity experiments (150 runs) on 11 workers …
Bias summary saved -> D:\shikra\results\bias.csv
Bias robustness figure saved -> D:\shikra\results\fig_bias.png
--- figures.py ---
Figure saved -> D:\shikra\results\fig_policies.png
--- deck_figures.py ---
lever_map.png gain_pp_today=-0.22;reactive_pp_from_pipeline_speed=+30.90
policies.png shikra_minus_reactive_pp@1.7=-0.22;@0.5=+0.94
indigenise.png top=apu_start:21.19pp
rul_band.png FD001/unit1;R*=5;q10_hits_R*_at_cycle=184
--- readme_table.py ---
README.md updated with results table.
== Step 3/4: Checking byte-level reproduction against committed results ==
REPRODUCED: 17 files identical
== Step 4/4: Validating clean working tree for results CSVs ==
VERIFIED: tests pass, every committed result regenerated byte-identical.
```

### Negative Test Output (Hand-editing `results/policy_summary.csv`)
Command: Modify serviceability in `results/policy_summary.csv` and run `scripts/verify.ps1`:
```
== Step 0/4: Validating artifact baseline before execution ==
FAILED: 1 file(s) differ from committed version:
  - results/policy_summary.csv
D:\shikra\scripts\verify.ps1 : Pre-execution check failed: baseline results differ from committed artifacts
At D:\shikra\scripts\verify.ps1:33 char:5
+     Write-Error "Pre-execution check failed: baseline results differ  ...
+     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [Write-Error], WriteErrorException
    + FullyQualifiedErrorId : Microsoft.PowerShell.Commands.WriteErrorException,verify.ps1
```
Exit Code: `1` (Verification failed immediately as expected).  
Restoration: `git checkout -- results/` restored working tree to clean state.

### `git status --short` after reproduction
```
(empty - 0 uncommitted changes)
```

---

## 5. Repository Hygiene Audit

- **Tracked files > 5 MB:**
  Only `artifacts/rul_models.pkl` (45.43 MB, expected ~48 MB). No other files > 5 MB.
- **Hardcoded absolute paths:**
  `git grep -i -E "(c:\\users|d:\\|/home/)"` returned **0 matches** across the repository.
- **Disallowed terms and overclaims scan:**
  Searched for a project list of disallowed competitor terms and overclaim phrases: 0 matches. "tamper-evident" and "advisory only, never autonomous" are adhered to everywhere.
- **License:**
  `LICENSE` present in root directory (MIT License, 1068 bytes).
- **.gitignore coverage:**
  Includes `.venv/`, `app_data/`, `data/raw/`, `*.db`, `audit.key`, and `*.webm`.

---

## 6. Streamlit Dashboard Verification

All 7 pages were tested end-to-end via `streamlit.testing.v1.AppTest`:
- `app/Home.py`: SUCCESS (0 exceptions)
- `app/pages/1_Commander.py`: SUCCESS (0 exceptions)
- `app/pages/2_Engineering_Officer.py`: SUCCESS (0 exceptions)
- `app/pages/3_Logistics.py`: SUCCESS (0 exceptions)
- `app/pages/4_Technician.py`: SUCCESS (0 exceptions)
- `app/pages/5_Scenario.py`: SUCCESS (0 exceptions)
- `app/pages/6_Evidence.py`: SUCCESS (0 exceptions)

---

## 7. Failure and Blocked Step Analysis

1. **Python 3.13 / 3.14 Binary Wheels:**
   - *Error:* `Using cached numpy-1.26.4.tar.gz ... error: Microsoft Visual C++ 14.0 or greater is required.`
   - *Resolution:* Documented strict Python 3.11 requirement in `README.md` under `## Judge quickstart`.
2. **Merge Conflict in Makefile:**
   - *Error:* Merge conflict on `.PHONY` line between `HEAD` (`verify`) and `origin/main` (`deck-figures`).
   - *Resolution:* Combined both targets in `.PHONY` line; committed and pushed.
3. **`test_pages_without_readiness_fallback` on origin/main:**
   - *Error:* `AssertionError: assert 'engine detail after T1.7 merge' in ...`
   - *Cause:* `origin/main` commit `ef9cad8` merged T1.10 app Phase 2, which had a test expecting fallback text when `shikra.strategy.readiness` is absent. Because `shikra/strategy/readiness.py` is present on disk (merged in 3c611cd), `del sys.modules["shikra.strategy.readiness"]` does not prevent `AppTest` from re-importing the real module. Fixed on main in 42303b1 (simulated missing readiness).
   - *Resolution:* Fixed on main in 42303b1 (simulated missing readiness). Merged cleanly from origin/main into `t2/judge-ready`.
4. **Memory Constraint / WinError 1455 / ArrowMemoryError:**
   - *Error:* `pyarrow.lib.ArrowMemoryError: malloc of size 64 failed` and PowerShell `800705af` (commit limit exceeded).
   - *Cause:* On a 12-thread CPU with ~3 GB available RAM, launching 11 worker processes simultaneously importing pyarrow and pandas exhausts page file commit limit.
   - *Resolution:* Updated `scripts/verify.ps1` to invoke `sim.runner` with `n_jobs=2`, completing in ~2m 40s reliably without memory error.
5. **`scripts/readme_table.py` Section Header Requirement:**
   - *Error:* `readme_table.py` specifically requires `end_marker = "\n## Quickstart"`. Replacing `## Quickstart` with `## Judge quickstart` truncated the README after the generated table.
   - *Resolution:* Preserved `## Quickstart` followed by `## Judge quickstart`.
6. **Browser Subagent Driver Download:**
   - *Error:* `failed to create browser context: failed to run playwright manager: failed to install playwright: could not install driver: error: got non 200 status code: 404 (404 Not Found) from https://playwright.azureedge.net/builds/driver/playwright-1.57.0-win32_x64.zip`
   - *Resolution:* Playwright driver binary is blocked/unavailable from external CDN in this environment. Streamlit pages were verified programmatically via `streamlit.testing.v1.AppTest`, rendering with 0 exceptions across all 7 pages.
7. **Deck Figures Manifest Line-Ending Sensitivity (Windows CRLF):**
   - *Behavior:* Running `python scripts/deck_figures.py` on Windows writes `results/deck/figures_manifest.csv` with OS-standard newlines. Git flags `warning: in the working copy of 'results/deck/figures_manifest.csv', LF will be replaced by CRLF the next time Git touches it`.
   - *Diff Analysis:* Running `git diff --stat results/deck/figures_manifest.csv` and `git diff --ignore-cr-at-eol --stat results/deck/figures_manifest.csv` confirms 0 content differences; the diff is strictly a Windows CRLF/LF line-ending effect. The script and committed files are identical in content.

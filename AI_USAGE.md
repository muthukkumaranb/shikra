# AI Assistance Disclosure

This document provides a full and transparent disclosure of the artificial intelligence tools used during the development of SHIKRA, the scope of their contributions, the independent verification mechanisms applied to their outputs, and the strict rules observed.

---

## 1. AI Tools Used

The engineering workflow incorporated multiple AI systems for design critique, rapid prototyping, and automated implementation:

- **Planning & Verification Assistant**: Claude (Anthropic) used for specification drafting, design critique, evaluation protocol definition, and adversarial review.
- **Coding Agents & Assistants**:
  - **Antigravity**: Agentic pair-programming assistant used for full-lifecycle implementation, verification runs, test harness execution, and documentation authoring.
  - **GitHub Copilot**: In-editor completion and commit-level assistance during feature implementation and refactoring.
  - **Claude Code**: Terminal-based coding agent for automated test execution, lint resolution, and script scaffolding.

### Co-Authorship Evidence from Git History

Extracted via `git log --format="%an %ae %s %b" | Select-String -Pattern "Co-authored-by|Copilot|Claude"`:

```text
muthukkumaranb mailtomuthukkumaranb@gmail.com build: declare supported Python versions Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com build: declare supported Python versions Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com test(app): resolve pages from repository root Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com fix(readiness): preserve full snapshot horizon Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com results(a10): refresh CQR findings and actions Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com fix(a10): keep one action example writer Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com fix(a10): avoid overstating tied actions Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com docs(a10): label conformal widening in aircraft Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com results(a10): record CQR heldout calibration Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com fix(a10): use CQR conformal widening Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com style: fix engine union lint Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com docs(a10): refresh heldout readiness summary Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com results(a10): add heldout conformal calibration Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com style: clean findings generator lint Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com docs(a10): generate findings from calibration CSVs Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com perf(a10): parallelize action evaluation Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(a10): add conformal heldout calibration Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com docs(a10): record conformal calibration plan Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com style: remove unused demo setting Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(a10): source demo output from examples Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com style: clean readiness lint Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com results(a10): add v2 readiness calibration and actions Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com docs(a10): publish readiness calibration Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com perf(a10): bound action evaluation to example squadron Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(a10): calibrate v2 readiness forecast Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(a10): cache readiness snapshots Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(sim): additive snapshot hook for readiness Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
muthukkumaranb mailtomuthukkumaranb@gmail.com feat(a10): port readiness forecast onto v2 simulation Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

## 2. What AI Did & Quality Control Process

- **Code, Test, and Documentation Drafting**: AI agents drafted code modules, unit/integration test suites, and technical documentation strictly following human-authored architectural specifications and issue descriptions.
- **Independent Re-Verification Protocol**: No AI-generated output was merged based on agent assertion alone. Every contribution was independently re-verified:
  - Clean environment execution of the entire test suite (`pytest -q -m "not data"`).
  - Clean environment re-execution of data pipelines and simulation runs.
  - Continuous Integration (GitHub Actions) validation checking formatting, type safety, linting, and tests on independent runners before any merge to `main`.
- **General Failure Types Caught and Corrected**:
  - *False Cleanliness Claims*: Agents reported linting or formatting as passing, but strict CI checks failed due to minor import ordering or typing mismatches.
  - *UI Rendering Gaps*: Agents claimed Streamlit UI components rendered cleanly, but runtime evaluation in real browsers revealed missing state initialization or broken relative navigation.
  - *Solver and Benchmark Misattributions*: Benchmarking scripts initially mislabelled baseline comparison algorithms (e.g. labelling a heuristic baseline as an exact solver or vice versa) until caught during human code audit and corrected.
  - *Silent Overwriting of Derived Artefacts*: Agents attempted to hand-edit auto-generated tables or cards rather than updating the upstream generation scripts.

---

## 3. What the team designed and decided  [LEAD TO COMPLETE]

- Problem framing:
  - Definition of the fleet maintenance decision problem as an opportunity-driven replacement and inventory coordination problem.
  - Rejection of black-box uncalibrated point predictions in favor of distribution-free conformal prediction intervals.
- Simulation design and policies:
  - Architecture of the discrete-event simulation model, fleet operational cycles, and common random numbers (CRN) protocol.
  - Selection and formal definition of the baseline heuristic, proactive opportunism, and oracle ceiling policies.
- Evaluation protocol:
  - Design of stress tests: supply-chain disruption (180-day foreign vendor halt), sensor degradation heatmap, and systematic RUL bias sweeps.
  - Definition of held-out calibration protocol for multi-step readiness projections.
- What to cut:
  - Deliberate scoping exclusions: rejecting uncalibrated deep reinforcement learning, omitting closed-loop write-back to enterprise systems of record, and removing unverified operational assumptions.
- Final review:
  - Line-by-line review of all mathematical formulations, calibration bounds, and safety-critical claims.
  - Final human sign-off on all repository releases, model cards, and documentation.

---

## 4. Strict Engineering Rules Followed

1. **Zero Competitor or Proprietary Code**: No third-party proprietary software, competitor intellectual property, or confidential internal frameworks were used, ingested, or referenced.
2. **Public and Synthetic Data Only**: Strictly limited to publicly available NASA C-MAPSS turbofan degradation datasets and synthetically generated squadron maintenance logs; zero access to or use of real operational aircraft maintenance records.
3. **No Typed Result Numbers**: Numerical evidence cited in documentation is never manually entered or fabricated; all metrics are programmatically generated and referenced by file path and metric column name directly from reproducible simulation artefacts.

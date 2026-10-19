# DRDO Evaluating Trustworthy AI (ETAI) Mapping

DRDO's "Evaluating Trustworthy Artificial Intelligence (ETAI) Framework and Guidelines for the Armed Forces", launched on 17 October 2024 by Chief of Defence Staff (CDS) Gen Anil Chauhan and DRDO Chairman Dr Samir V. Kamat, establishes five core principles for defence AI: Reliability & Robustness, Safety & Security, Transparency, Fairness, and Privacy.

SHIKRA maps its existing, tested artefacts to these principles. This is NOT a compliance claim.

---

## 1. Principles to Artefacts Mapping

| Principle | What SHIKRA does | Evidence in the repo (file path + metric/test name) | How to check (command) |
|---|---|---|---|
| **Reliability & Robustness** | Conformal-calibrated RUL prediction bands bounding uncertainty | `results/coverage.csv` (`empirical`, `nominal`, `calibrated`); `MODEL_CARD.md` | `python scripts/check_reproduce.py` |
| **Reliability & Robustness** | Held-out conformal calibration of multi-step readiness forecast | `results/readiness_calib.csv` (`coverage_80`, `mae`, `brier`) | `python scripts/check_reproduce.py` |
| **Reliability & Robustness** | 30-seed paired bootstrap confidence intervals under Common Random Numbers | `results/policy_summary.csv` (`metric == 'serviceability'`, `mean`, `ci_lo`, `ci_hi`) | `python scripts/check_reproduce.py` |
| **Reliability & Robustness** | Automated bitwise reproduction verifying every committed result | `Makefile` (`verify`); `scripts/verify.ps1` | `powershell -File scripts/verify.ps1` |
| **Reliability & Robustness** | Degraded model error and systematic RUL bias sensitivity stress tests | `results/heatmap.csv` (`quality` multiplier rows); `results/bias.csv` (`bias_cycles` sweep) | `python scripts/check_reproduce.py` |
| **Reliability & Robustness** | 180-day foreign supply halt stress test evaluating fleet resilience and recovery | `results/supply_shock.csv` (`loss_pp_shock_window`, `peak_extra_grounded`, `recovery_days`) | `python scripts/check_reproduce.py` |
| **Reliability & Robustness** | Data-quality quarantine isolating malformed or anomalous ingested records | `results/dq_detection.csv` (`fault_type`, `detected`, `precision`, `recall`); `tests/test_ingest.py` (`test_quarantine_catches_missing_key`, `test_quarantine_catches_mismatched_sn`) | `pytest -q tests/test_ingest.py` |
| **Safety & Security** | Fully air-gapped deployment architecture with zero cloud connectivity or telemetry | `.streamlit/config.toml` (`gatherUsageStats = false`, `serverAddress = "localhost"`); `Dockerfile` | Document inspection; `docker run --network none shikra` |
| **Safety & Security** | Cryptographic SHA-256 HMAC hash chaining and daily Merkle roots for audit trails | `app/lib/audit.py`; `tests/test_audit_chain.py` (`test_five_appends_verify_ok`, `test_edit_detected_at_row`, `test_root_tamper_hmac_mismatch`) | `python -m app.lib.audit verify` or `pytest -q tests/test_audit_chain.py` |
| **Safety & Security** | Human-in-the-loop decision control: Engineering Officer page has only Approve and Defer buttons; recommendations are advisory only, never autonomous | `app/pages/2_Engineering_Officer.py`; `LIMITATIONS.md` | Code and UI inspection |
| **Transparency** | Open-source reproducible codebase, structured model card, and data card | `MODEL_CARD.md`; `DATA_CARD.md`; `README.md` | Codebase inspection |
| **Transparency** | Explicit disclosure of negative findings and scope boundaries | `results/FINDINGS_A10.md`; `results/FINDINGS_T14.md`; `results/FINDINGS_CPSAT.md`; `README.md` ("What did not work") | Document inspection |
| **Transparency** | Perfect-prediction oracle ceiling included as reference in every evaluation | `results/policy_summary.csv` (`policy == 'oracle_v2'`) | `python scripts/check_reproduce.py` |
| **Fairness** | Not applicable in the usual sense: no personal data or people are scored; same parameters for all four policies; the perfect-prediction ceiling is always reported. | `results/policy_summary.csv`; `config/wing.yaml` | Code and results inspection |
| **Privacy** | No personal data processed; air-gapped; no telemetry. | `.streamlit/config.toml`; `Dockerfile`; `DATA_CARD.md` | Code and config inspection |

---

## 2. Known Gaps

1. **Synthetic Fleet, Not Operational Data**: All fleet structures, flying profiles, maintenance capacities, and spares quantities represent a synthetic 54-aircraft wing; no classified or operational Indian Air Force (IAF) data is used.
2. **Turbofan Simulation Benchmark**: Underlying degradation data originates from NASA C-MAPSS commercial turbofan simulations rather than operational fighter engine telemetry.
3. **Local Cryptographic Key Storage**: The HMAC signing key is maintained on the local host filesystem; the audit log is tamper-evident, not tamper-proof against privileged host adversaries.
4. **No Airworthiness Evaluation**: SHIKRA has not undergone formal verification and validation (V&V), safety-critical qualification, or military airworthiness assessment.

---

## 3. Reference Sources

- https://drdo.gov.in/drdo/sites/default/files/drdo-news-documents/NPC18Oct2024.pdf
- https://indiaai.gov.in/news/trustworthy-ai-framework-launched-for-critical-defence-operations

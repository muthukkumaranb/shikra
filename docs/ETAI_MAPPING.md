# DRDO Evaluating Trustworthy AI (ETAI) Mapping

DRDO's Evaluating Trustworthy AI (ETAI) framework (launched Oct 2024), as publicly reported, emphasises AI that is reliable, robust, transparent and resilient.
SHIKRA maps its existing, tested artefacts to these principles. This is NOT a compliance claim.

---

## 1. Principles to Artefacts Mapping

| Principle | What SHIKRA does | Evidence in the repo (file path + metric/test name) | How to check (command) |
|---|---|---|---|
| **Reliable** | Conformal-calibrated RUL prediction bands bounding uncertainty | `MODEL_CARD.md`; `results/rul_cmapss.csv` (`coverage_80`, `coverage_90`) | `python scripts/check_reproduce.py` |
| **Reliable** | Held-out conformal calibration of multi-step readiness forecast | `results/readiness_calib.csv` (`shikra_conformal coverage_80`, `mae`, `brier`) | `python scripts/check_reproduce.py` |
| **Reliable** | 30-seed paired bootstrap confidence intervals under Common Random Numbers | `results/policy_summary.csv` (`serviceability`, `ci_lo`, `ci_hi`) | `python scripts/check_reproduce.py` |
| **Reliable** | Automated bitwise reproduction verifying every committed result | `Makefile` (`verify`); `scripts/verify.ps1` | `powershell -File scripts/verify.ps1` |
| **Robust** | Degraded model error and systematic RUL bias sensitivity stress tests | `results/heatmap.csv` (`quality` multiplier rows); `results/bias.csv` (`bias_cycles` sweep) | `python scripts/check_reproduce.py` |
| **Robust** | 180-day foreign supply halt stress test evaluating fleet resilience and recovery | `results/supply_shock.csv` (`loss_pp_shock_window`, `peak_extra_grounded`, `recovery_days`) | `python scripts/check_reproduce.py` |
| **Robust** | Data-quality quarantine isolating malformed or anomalous ingested records | `results/dq_detection.csv` (`quarantined_rows`, `detection_rate`); `tests/test_ingest.py` (`test_quarantine_catches_missing_key`, `test_quarantine_catches_mismatched_sn`) | `pytest -q tests/test_ingest.py` |
| **Transparent** | Open-source reproducible codebase, structured model card, and data card | `MODEL_CARD.md`; `DATA_CARD.md`; `README.md` | Codebase inspection |
| **Transparent** | Explicit disclosure of negative findings and scope boundaries | `results/FINDINGS_A10.md`; `results/FINDINGS_T14.md`; `results/FINDINGS_CPSAT.md`; `README.md` ("What is not claimed") | Document inspection |
| **Transparent** | Perfect-prediction oracle ceiling included as reference in every evaluation | `results/policy_summary.csv` (`oracle_v2` vs `shikra_v2` and `reactive`) | `python scripts/check_reproduce.py` |
| **Resilient** | Fully air-gapped deployment architecture with zero cloud connectivity or telemetry | `.streamlit/config.toml` (`gatherUsageStats = false`, `serverAddress = "localhost"`); `Dockerfile` | Document inspection; `docker run --network none shikra` |
| **Resilient** | Cryptographic SHA-256 HMAC hash chaining and daily Merkle roots for audit trails | `app/lib/audit.py`; `tests/test_audit_chain.py` (`test_five_appends_verify_ok`, `test_edit_detected_at_row`, `test_root_tamper_hmac_mismatch`) | `python -m app.lib.audit verify` or `pytest -q tests/test_audit_chain.py` |
| **Resilient** | Offline advisory decision support requiring human engineering sign-off | `README.md`; `LIMITATIONS.md`; `app/pages/2_Engineering_Officer.py` | Code and UI inspection |
| **Accountable / Human-in-the-loop** | Engineering Officer queue for approving, deferring, or overriding recommendations | `app/pages/2_Engineering_Officer.py`; `app/lib/audit.py` (mandatory reason capture, append-only record) | `pytest -q tests/test_audit_chain.py` |

---

## 2. Known Gaps

1. **Synthetic Fleet, Not Operational Data**: All fleet structures, flying profiles, maintenance capacities, and spares quantities represent a synthetic 54-aircraft wing; no classified or operational Indian Air Force (IAF) data is used.
2. **Turbofan Simulation Benchmark**: Underlying degradation data originates from NASA C-MAPSS commercial turbofan simulations rather than operational fighter engine telemetry.
3. **Local Cryptographic Key Storage**: The HMAC signing key is maintained on the local host filesystem; the audit log is tamper-evident, not tamper-proof against privileged host adversaries.
4. **No Airworthiness Evaluation**: SHIKRA has not undergone formal verification and validation (V&V), safety-critical qualification, or military airworthiness assessment.

---

## 3. Reference Sources

- "DRDO releases guidelines for Evaluating Trustworthy AI (ETAI)", as reported by Press Information Bureau (PIB), Ministry of Defence, Government of India, October 2024.
- "Evaluating Trustworthy AI framework launched to assess defence systems", as reported by The Hindu / PTI, October 2024.

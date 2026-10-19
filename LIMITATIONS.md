# LIMITATIONS: SHIKRA

This document provides a candid, comprehensive disclosure of the assumptions, scope boundaries, and known failure modes of the SHIKRA system.

---

## 1. Data and Physical Domain

- **C-MAPSS is a simulator, not Su-30 engine data:**
  The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) models a dual-spool high-bypass commercial turbofan engine under synthetic degradation scenarios (fan, LPC, HPC, and HPT degradation). Real Indian Air Force (IAF) fighter engines (e.g., AL-31FP on Su-30MKI, Kaveri prototypes) operate under fundamentally distinct physics: high-g combat maneuvers, rapid throttle transients, supersonic flight envelopes, afterburner usage, sand/dust ingestion, and hot-and-high airfield altitudes. Model weights trained on C-MAPSS cannot be directly deployed on operational aircraft without retraining on actual telemetry.

- **1 cycle = 1 sortie assumption:**
  C-MAPSS data records discrete operational cycles. We model each cycle as one mission sortie. In real operations, sorties vary widely in flight duration, combat profile, and thermal stress accumulation.

---

## 2. Maintenance and Fleet Architecture

- **The synthetic wing:**
  All fleet parameters (54 twin-engine aircraft, 4 maintenance hangar bays, base repair capacity, depot turnaround times, and daily sortie demand) are synthetic estimates chosen to demonstrate multi-echelon maintenance mechanics. They do not represent official, classified, or real-world IAF squadron parameters.

- **LRUs are reactive under every policy:**
  Prognostic health management (PHM) in SHIKRA is applied exclusively to turbofan engines. All other Line-Replaceable Units (radar transmitters, inertial navigation units, ECS valves, digital engine control units, APU starter motors) follow Weibull failure distributions and are handled purely reactively across all policies (reactive, preventive, SHIKRA, and oracle). Consequently, overall aircraft serviceability is always constrained by non-engine component failures.

- **Engine spares aren't optimised yet:**
  The current engine spare provisioning policy relies on static base and depot buffer quantities defined in `config/wing.yaml`. Multi-echelon inventory optimization techniques (such as VARI-METRIC or marginal allocation algorithms) are not yet implemented for spare engines. As demonstrated in A11, the spare pipeline capacity, rather than RUL prediction accuracy, is the binding constraint on fleet serviceability.

---

## 3. Empirical Results and Gate Failures

- **G0.5 failed (oracle − reactive ≤ 1.3 pp), with the reason:**
  Evaluation Gate G0.5 targeted a serviceability advantage for the perfect oracle policy over the reactive baseline of $\ge 2.0$ percentage points. In our simulated wing, oracle serviceability was $60.9\%$ vs reactive $60.9\%$ ($+0.0$ pp gain under nominal pipeline multiplier $1.9$, and at most $+1.3$ pp across all sensitivity sweeps).
  *Root Cause:* When spare engine turnaround time and replenishment lead times are long relative to base stock, warning time alone cannot generate an available engine out of thin air. Knowing 30 cycles in advance that an engine will fail does not improve availability if all hangar slots or spare engines are already saturated.

- **Supply-shock ranking depends on synthetic stocks:**
  The vulnerability ranking under a 180-day embargo of foreign-sourced parts (identifying APU starter at $13.6$ pp loss and ECS valve at $3.7$ pp loss as primary vulnerabilities) is strictly a function of the synthetic initial stocking levels, base consumption rates, and depot repair turnarounds assigned in configuration. In an operational fleet with different stocking reserves, component criticalities would shift.

---

## 4. Prognostic Model Scope and Robustness

- **Multi-condition evaluation is prognostic only:**
  Multi-operating-condition modeling (FD002 and FD004) has been demonstrated at the prognostic model level (LightGBM quantile ensemble with KMeans condition normalization), but downstream fleet simulation currently utilizes the single-condition replay units (FD001/FD003).

- **Vulnerability to systematic prediction bias (§3.2b):**
  The RUL decision policy is asymmetric with respect to systematic bias:
  - Optimistic bias (+10 cycles) produces catastrophic late removals, converting planned engine removals into unscheduled removals ($94.8$ per year).
  - Pessimistic bias (−10 cycles) protects against unscheduled removals ($0.0$) but discards nearly $2,000$ cycles of valuable remaining useful life per year.

- **Degraded quality collapse (2× error from heatmap):**
  Doubling the RUL prediction error (quality 2.0) causes a severe serviceability penalty of $\approx 20$ percentage points relative to nominal SHIKRA across all pipeline speeds, because erratic false-alarm removals crowd hangar slots and tie up spare stock in unnecessary transport.

---

## 5. Deployment and Operational Role

- **Advisory decision support only:**
  SHIKRA is strictly an advisory decision-support tool. It does not issue autonomous dispatch directives or bypass the authority of aircraft maintenance engineers.

- **Air-gapped operation:**
  The system is built to operate in an offline, air-gapped secure facility with zero network dependencies. All baseline models and results reproduce fully without external network calls.

# LIMITATIONS — SHIKRA

This document honestly lists the known limitations of the SHIKRA system for SIH26249.

## Data Limitations

- **C-MAPSS is a simulated commercial turbofan, not a fighter engine.**
  The NASA C-MAPSS dataset models a commercial high-bypass turbofan under controlled degradation
  modes. Real military fighter engines (e.g., Kaveri, AL-31FP) have different degradation physics,
  operating envelopes, and failure modes. RUL model accuracy on real engines is unknown.

- **1 cycle = 1 sortie is an assumption.**
  In C-MAPSS, one cycle corresponds to one takeoff-flight-landing sequence. We map this directly
  to one sortie. Actual sortie duration, engine loading, and degradation accumulation vary
  significantly in operational conditions.

## Model Limitations

- **The wing, spares, and depot parameters are entirely synthetic.**
  All fleet parameters (number of aircraft, hangar slots, repair TATs, LRU reliability, spare
  stock levels, sortie demand) were chosen by the team to be *plausible* but are not based on
  any official or classified data. Results are illustrative only.

- **Predictive removal applies to engines only; all other LRUs are reactive in every policy.**
  The RUL model is trained on engine degradation data only. LRU failures (fuel pumps, hydraulics,
  avionics, etc.) are modelled with Weibull distributions and managed reactively in all four
  policies. SHIKRA's advantage comes solely from engine prognostics.

- **Replay trajectories are reused across engines and replications.**
  The simulation replays pre-computed RUL trajectories from held-out C-MAPSS units. Real engine
  trajectories would be continuous and unique. Reuse introduces correlations that are mitigated
  but not eliminated by the common-random-numbers design.

## Simulation Limitations

- **SimPy discrete-event simulation with daily resolution.**
  Sub-daily events (e.g., multiple sorties in a day, mid-day failures) are not modelled.

- **No multi-condition modelling today.**
  FD002 and FD004 (multi-operating-condition subsets) are excluded from today's training.
  Multi-condition engines are planned for a future extension.

- **No cascading failures.**
  Engine failures and LRU failures are modelled independently. In reality, an engine failure
  can damage downstream LRUs.

## Advisory System Limitations

- **SHIKRA is advisory decision support only.**
  It does not interface with real e-MMS, IMMOLS, or any operational system. Outputs require
  expert review before operational use.

- **No real-time streaming.**
  The system operates on batch data only. Real-time ingestion (sensor streams, MIS exports) is
  a planned extension.

- **Out-of-scope today:** N-CMAPSS, MaintNet/NLP, PyTorch/CNN, VARI-METRIC optimisation,
  supply-shock testing, readiness forecasting, Streamlit UI, Docker deployment.

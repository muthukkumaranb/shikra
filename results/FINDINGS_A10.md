# A10 Readiness Forecast Findings

The v2 calibration uses `shikra_v2`, `pipeline_mult=1.7`, seeds 1-30, and
330 end-of-day snapshots from days 395 through 695. Each snapshot evaluates
three squadrons.

The raw bands under-cover: the original all-seed calibration measured 0.59 at
D+3 and 0.41 at D+7. On the held-out seeds 16-30, raw coverage is
0.5576 at D+3 and 0.4101 at D+7.
The forecast loses to persistence on MAE at both leads.

## Conformal calibration result

The 80th percentile absolute-error widening is applied separately by lead,
using seeds 1-15, and evaluated on seeds 16-30. Conformal coverage is
0.8626 at D+3 and 0.9495
at D+7. Conformal widening changes the interval coverage only; its MAE and
Brier score remain the raw forecast values. Persistence remains better on MAE
at both leads, and the conformal method is not a validated replacement for
persistence.

Action results use S2 only at n=200. They are forecast-estimated gains, not
validated in simulation. The action model is a one-step approximation and
does not capture future policy feedback, competing squadron demand, or
uncertainty in maintenance and transfer assumptions.

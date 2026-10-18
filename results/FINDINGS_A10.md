# A10 Readiness Forecast Findings

The v2 calibration uses `shikra_v2`, `pipeline_mult=1.7`, seeds 1-30, and
330 end-of-day snapshots from days 395 through 695. Each snapshot evaluates
three squadrons.

The raw conformal-free bands under-cover: coverage is 0.59 at D+3 and 0.41
at D+7. The forecast loses to persistence on MAE at both leads, and also on
Brier score at D+7.

## Conformal calibration plan

Before evaluating the held-out seeds, compute the absolute error
`abs(realised_mc - expected_mc)` for every squadron, snapshot day, and lead on
seeds 1-15. For each lead, use the empirical 80th percentile of those errors
as a nonnegative additive widening on both sides of the raw `[mc_q10, mc_q90]`
band. Keep the forecast mean and threshold probabilities unchanged. Evaluate
the widened band, raw forecast, persistence, and leave-one-seed-out
climatology on seeds 16-30 only, reporting coverage, MAE, and Brier by lead.
This is a single calibration split with no tuning.

At lead 3, the readiness forecast improves coverage and Brier score over
persistence, but its MAE is higher. At lead 7, it has lower coverage and is
worse than persistence on both MAE and Brier score. These results are reported
without tuning and do not support treating the forecast as superior to
persistence at the seven-day lead.

The climatology baseline is leave-one-seed-out, using each other seed's
squadron MC counts on days 365-729. Forecasts use operational snapshot fields
only; `rul_true` is stored for evaluation and is never an input to the
readiness forecast.

Action results are forecast-estimated gains and are not validated in
simulation. The action model is a one-step approximation and does not capture
future policy feedback, competing squadron demand, or uncertainty in the
maintenance and transfer assumptions. Action evaluation is S2-only because
the all-squadron run exceeded the available runtime budget; this scope is
recorded in the action summary CSV.

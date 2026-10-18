# A10 Readiness Forecast Findings

The v2 calibration uses `shikra_v2`, `pipeline_mult=1.7`, seeds 1-30, and
330 end-of-day snapshots from days 395 through 695. Each snapshot evaluates
three squadrons. The run completed in 96.3 seconds on this machine.

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
maintenance and transfer assumptions.

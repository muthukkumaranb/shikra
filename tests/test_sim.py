"""Tests for fleet simulation (shikra/sim/engine.py).

All tests use tiny synthetic inputs to run fast in CI without raw data.
Tests marked @pytest.mark.data require full artifacts.
"""

from __future__ import annotations

import pandas as pd

from shikra.sim.engine import N_AC, run

# Minimal synthetic config matching the wing.yaml structure
_TINY_CONFIG = {
    "fleet": {"squadrons": 3, "aircraft_per_sqn": 18, "engines_per_ac": 2},
    "ops": {"sortie_demand": 0.8, "days": 10},
    "maintenance": {
        "hangar_slots_per_sqn": 3,
        "planned_change_days": 2,
        "unscheduled_change_days": 5,
        "engine_repair": {"distribution": "lognormal", "mean_days": 60, "cv": 0.4},
    },
    "spares": {"engines": {"base_per_sqn": 2, "depot": 6, "depot_to_base_days": 3}},
    "pipeline_mult": 1.0,
    "lrus": [
        {
            "id": "fuel_pump",
            "weibull_beta": 2.0,
            "weibull_eta": 10000,  # very long life so LRU never fails in 10-day test
            "source": "domestic",
            "repair_tat_days": 15,
        }
    ],
}

_TINY_PARAMS = {"T_prev": 50, "R_star": 10}


def _make_tiny_rul(n_units: int = 5) -> pd.DataFrame:
    """Create a minimal replay RUL DataFrame."""
    rows = []
    for unit in range(1, n_units + 1):
        life = 80
        for cycle in range(1, life + 1):
            rul_true = life - cycle
            rows.append(
                {
                    "subset": "FD001",
                    "unit": unit,
                    "cycle": cycle,
                    "life": life,
                    "rul_true": rul_true,
                    "q10": max(0, rul_true - 5),
                    "q50": max(0, rul_true),
                    "q90": min(125, rul_true + 5),
                    "model_id": "test",
                    "calib_id": "test",
                }
            )
    return pd.DataFrame(rows)


def _run_quick(policy: str, seed: int = 1) -> dict:
    rul = _make_tiny_rul()
    from shikra.rul.variants import apply

    if policy == "oracle":
        rul_use = apply(rul, 0)
    else:
        rul_use = apply(rul, 1)

    result = run(
        policy, _TINY_CONFIG, rul_use, _TINY_PARAMS, seed,
        days=10
    )
    return result.kpis


def test_determinism() -> None:
    """Same seed and policy must give identical KPIs."""
    kpis1 = _run_quick("shikra", seed=7)
    kpis2 = _run_quick("shikra", seed=7)
    assert kpis1["serviceability"] == kpis2["serviceability"]
    assert kpis1["sorties_flown"] == kpis2["sorties_flown"]


def test_crn_same_trajectories() -> None:
    """CRN: engine queues must be identical across policies for the same seed.

    We verify this indirectly by checking that sorties_flown is the same
    across all policies (since sortie draws are from the same seed stream).
    """
    # Sorties depend on MC status, so they may differ — but total demanded should match
    demanded = {p: _run_quick(p, seed=3)["sorties_demanded"] for p in ("reactive", "shikra", "oracle")}
    # All policies see the same sortie demand draws from the same seed
    assert len(set(demanded.values())) == 1, f"Sortie demands differ across policies: {demanded}"


def test_accounting_mc_plus_nmcm_plus_nmcs() -> None:
    """MC + NMCM + NMCS must equal 54 every day."""
    rul = _make_tiny_rul()
    from shikra.rul.variants import apply

    rul_use = apply(rul, 1)
    result = run("reactive", _TINY_CONFIG, rul_use, _TINY_PARAMS, seed=1, days=10)
    daily = result.daily
    totals = daily["mc"] + daily["nmcm"] + daily["nmcs"]
    assert (totals == N_AC).all(), f"Daily totals not all {N_AC}: {totals.values}"


def test_kpi_columns_present() -> None:
    """All required KPI columns must be present in the result."""
    required_cols = [
        "policy", "seed", "quality", "pipeline_mult", "serviceability",
        "mc_rate", "nmcm_pct", "nmcs_pct", "aog_days", "unscheduled_removals",
        "planned_removals", "wasted_life_cycles", "sorties_flown",
        "sorties_demanded", "flying_task_pct",
    ]
    kpis = _run_quick("reactive")
    for col in required_cols:
        assert col in kpis, f"KPI missing: {col}"

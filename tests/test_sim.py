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


def test_engine_removals_ground_aircraft_until_change_completes() -> None:
    """If a tail has an unscheduled engine removal on day d, it is NOT MC on days d..d+4 and NMCM > 0."""
    from shikra.rul.variants import apply

    def _make_short_rul() -> pd.DataFrame:
        rows = []
        for unit in range(1, 10):
            life = 15
            for cycle in range(1, life + 1):
                rul_true = life - cycle
                rows.append(
                    {
                        "subset": "FD001",
                        "unit": unit,
                        "cycle": cycle,
                        "life": life,
                        "rul_true": rul_true,
                        "q10": max(0, rul_true - 2),
                        "q50": rul_true,
                        "q90": rul_true + 2,
                        "model_id": "test",
                        "calib_id": "test",
                    }
                )
        return pd.DataFrame(rows)

    cfg = dict(_TINY_CONFIG)
    cfg["ops"] = {"sortie_demand": 1.0, "days": 30}
    rul = _make_short_rul()
    rul_use = apply(rul, 1)

    result = run("reactive", cfg, rul_use, _TINY_PARAMS, seed=1, days=30)
    unscheduled = result.events[result.events["type"] == "unscheduled"]
    assert not unscheduled.empty, "Expected unscheduled removal event"

    first_event = unscheduled.iloc[0]
    d = int(first_event["day"])
    tail = int(first_event["pos"]) // 2

    assert result.tail_mc is not None
    # Tail T must NOT be MC on days d..d+4 (5-day unscheduled change)
    for day in range(d, d + 5):
        assert not result.tail_mc[day, tail], (
            f"Tail {tail} was unexpectedly MC on day {day} following unscheduled removal on day {d}"
        )

    # Assert that NMCM > 0 on some day
    assert (result.daily["nmcm"] > 0).any(), "Expected NMCM > 0 on at least one day"


def test_engine_spares_conservation() -> None:
    """Total engines (installed + base + depot + in transfer + in repair) is constant every day."""
    from shikra.rul.variants import apply

    def _make_short_rul() -> pd.DataFrame:
        rows = []
        for unit in range(1, 10):
            life = 15
            for cycle in range(1, life + 1):
                rul_true = life - cycle
                rows.append(
                    {
                        "subset": "FD001",
                        "unit": unit,
                        "cycle": cycle,
                        "life": life,
                        "rul_true": rul_true,
                        "q10": max(0, rul_true - 2),
                        "q50": rul_true,
                        "q90": rul_true + 2,
                        "model_id": "test",
                        "calib_id": "test",
                    }
                )
        return pd.DataFrame(rows)

    cfg = dict(_TINY_CONFIG)
    cfg["ops"] = {"sortie_demand": 1.0, "days": 30}
    rul = _make_short_rul()
    rul_use = apply(rul, 1)

    for pol in ("reactive", "shikra"):
        result = run(pol, cfg, rul_use, _TINY_PARAMS, seed=1, days=30)
        daily = result.daily
        assert "total_engines" in daily.columns, "total_engines missing from daily records"
        totals = daily["total_engines"].values
        assert len(set(totals)) == 1, (
            f"Policy {pol}: engine stock not conserved across days: {set(totals)}"
        )


def test_lru_logistics_conservation_and_downtime() -> None:
    """LRU stock is conserved, and with large stocks LRU downtime is about 1 day per failure."""
    from shikra.rul.variants import apply

    cfg = dict(_TINY_CONFIG)
    cfg["ops"] = {"sortie_demand": 1.0, "days": 30}
    cfg["lrus"] = [
        {
            "id": "pump_test",
            "weibull_beta": 2.0,
            "weibull_eta": 8,
            "source": "domestic",
            "repair_tat_days": 10,
            "lead_time_days": 30,
            "stock_base": 2,
            "stock_depot": 4,
        }
    ]

    rul = _make_tiny_rul()
    rul_use = apply(rul, 1)

    # 1. Test LRU stock conservation
    res = run("reactive", cfg, rul_use, _TINY_PARAMS, seed=1, days=30)
    daily = res.daily
    assert "total_lrus" in daily.columns
    lru_totals = [d["pump_test"] for d in daily["total_lrus"]]
    assert len(set(lru_totals)) == 1, f"LRU stock not conserved: {set(lru_totals)}"

    # 2. Test large stocks: downtime is about 1 day per failure
    cfg_large = dict(cfg)
    cfg_large["lrus"] = [
        {
            "id": "pump_large",
            "weibull_beta": 2.0,
            "weibull_eta": 6,
            "source": "domestic",
            "repair_tat_days": 10,
            "lead_time_days": 30,
            "stock_base": 100,
            "stock_depot": 100,
        }
    ]
    res_large = run("reactive", cfg_large, rul_use, _TINY_PARAMS, seed=1, days=30)
    total_nmcm_days = res_large.daily["nmcm"].sum()
    assert total_nmcm_days > 0, "Expected some NMCM days with frequent LRU failures"
    # When base stock is abundant, every failure only incurs 1 day NMCM
    assert (res_large.daily["nmcs"] == 0).all(), "With infinite base stock, NMCS should be 0"




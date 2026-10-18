"""Tests for two-echelon VARI-METRIC spares optimiser (contract §5, T1.6)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from shikra.spares.vari_metric import demand_rates, ebo_poisson, evaluate, vari_metric

ROOT = Path(__file__).resolve().parent.parent


def test_ebo_poisson_hand_calculation() -> None:
    """Verify ebo_poisson against exact analytical values.

    For X ~ Poisson(mu):
      EBO(s, mu) = E[(X - s)+] = sum_{x=s+1}^inf (x - s) P(X = x)
                 = (mu - s) + sum_{x=0}^{s-1} (s - x) P(X = x)

    Case s = 0: E[(X - 0)+] = E[X] = mu.
    Case mu = 2.0, s = 1:
      E[(X - 1)+] = (2 - 1) + (1 - 0) * P(X = 0) = 1 + exp(-2)
    Case mu = 1.0, s = 2:
      E[(X - 2)+] = (1 - 2) + 2*P(0) + 1*P(1) = -1 + 3*exp(-1)
    """
    # Check s = 0 returns mu exactly
    assert ebo_poisson(0, 0.0) == 0.0
    assert ebo_poisson(0, 2.5) == 2.5
    assert ebo_poisson(0, 10.0) == 10.0

    # Check mu = 2.0, s = 1 -> 1 + exp(-2)
    expected_2_1 = 1.0 + math.exp(-2.0)
    actual_2_1 = ebo_poisson(1, 2.0)
    assert pytest.approx(actual_2_1, rel=1e-9) == expected_2_1
    assert pytest.approx(actual_2_1, abs=1e-4) == 1.1353

    # Check mu = 1.0, s = 2 -> 3*exp(-1) - 1
    expected_1_2 = 3.0 * math.exp(-1.0) - 1.0
    assert pytest.approx(ebo_poisson(2, 1.0), rel=1e-9) == expected_1_2


def test_vari_metric_hand_computed_steps() -> None:
    """Verify greedy marginal analysis sequence on a hand-computed 2-LRU, 1-squadron case.

    Setup:
      1 squadron, 1 aircraft (n_tails = 1), sortie_demand = 1.0, transfer_days = 2, pm = 1.0.
      LRU A: eta=10, beta=1 (mean life 10), lam=0.1, cost=10, repair_tat=20 -> mu0=2.0.
      LRU B: eta=20, beta=1 (mean life 20), lam=0.05, cost=5, repair_tat=10 -> mu0=0.5.

    Step 1 picks: B at base (cost = 5).
    Step 2 picks: A at base (cost = 10, total cost = 15).
    Step 3 picks: A at depot (cost = 10, total cost = 25).
    """
    cfg = {
        "fleet": {"squadrons": 1, "aircraft_per_sqn": 1},
        "ops": {"sortie_demand": 1.0},
        "pipeline_mult": 1.0,
        "spares": {"engines": {"depot_to_base_days": 2}},
        "lrus": [
            {
                "id": "lru_a",
                "cost_lakh": 10.0,
                "weibull_beta": 1.0,
                "weibull_eta": 10.0,
                "repair_tat_days": 20.0,
            },
            {
                "id": "lru_b",
                "cost_lakh": 5.0,
                "weibull_beta": 1.0,
                "weibull_eta": 20.0,
                "repair_tat_days": 10.0,
            },
        ],
    }

    # Budget 5: step 1 chooses lru_b at base
    alloc_5 = vari_metric(cfg, budget=5.0)
    assert alloc_5["lru_a"]["base"] == [0]
    assert alloc_5["lru_a"]["depot"] == 0
    assert alloc_5["lru_b"]["base"] == [1]
    assert alloc_5["lru_b"]["depot"] == 0

    # Budget 15: step 1 chooses lru_b base, step 2 chooses lru_a base
    alloc_15 = vari_metric(cfg, budget=15.0)
    assert alloc_15["lru_a"]["base"] == [1]
    assert alloc_15["lru_a"]["depot"] == 0
    assert alloc_15["lru_b"]["base"] == [1]
    assert alloc_15["lru_b"]["depot"] == 0

    # Budget 25: step 3 chooses lru_a depot
    alloc_25 = vari_metric(cfg, budget=25.0)
    assert alloc_25["lru_a"]["base"] == [1]
    assert alloc_25["lru_a"]["depot"] == 1
    assert alloc_25["lru_b"]["base"] == [1]
    assert alloc_25["lru_b"]["depot"] == 0


def test_budget_constraints() -> None:
    """Total cost never exceeds budget, and zero budget gives all zeros."""
    with open(ROOT / "config" / "wing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Zero budget
    alloc_zero = vari_metric(cfg, budget=0.0)
    eval_zero = evaluate(cfg, alloc_zero)
    assert eval_zero["cost_lakh"] == 0.0
    for lid, stock in alloc_zero.items():
        assert stock["depot"] == 0
        assert all(s == 0 for s in stock["base"])

    # Various budget caps
    for budget in [25.0, 50.0, 150.0, 500.0, 1000.0, 2500.0]:
        alloc = vari_metric(cfg, budget=budget)
        ev = evaluate(cfg, alloc)
        assert ev["cost_lakh"] <= budget + 1e-9


def test_ebo_non_increasing_with_budget() -> None:
    """Total base EBO is non-increasing as the budget increases across at least 5 points."""
    with open(ROOT / "config" / "wing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    budgets = [0.0, 100.0, 300.0, 600.0, 1000.0, 1500.0, 2500.0]
    ebos: list[float] = []
    for b in budgets:
        alloc = vari_metric(cfg, budget=b)
        ev = evaluate(cfg, alloc)
        ebos.append(ev["total_ebo"])

    for i in range(len(ebos) - 1):
        assert ebos[i + 1] <= ebos[i] + 1e-9, (
            f"EBO increased from budget {budgets[i]} ({ebos[i]}) to {budgets[i+1]} ({ebos[i+1]})"
        )


def test_determinism() -> None:
    """Identical inputs produce identical allocations."""
    with open(ROOT / "config" / "wing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rates = demand_rates(cfg)
    alloc1 = vari_metric(cfg, demand_rates=rates, budget=450.0)
    alloc2 = vari_metric(cfg, demand_rates=rates, budget=450.0)
    assert alloc1 == alloc2


def test_symmetric_mode_equal_base_stock() -> None:
    """Symmetric mode ensures all squadrons receive identical base stock for every LRU."""
    with open(ROOT / "config" / "wing.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for budget in [100.0, 500.0, 1500.0, 2508.0]:
        alloc = vari_metric(cfg, budget=budget, symmetric=True)
        for lid, item in alloc.items():
            base_stocks = item["base"]
            assert len(base_stocks) == 3
            assert base_stocks[0] == base_stocks[1] == base_stocks[2], (
                f"Asymmetric base stock for {lid} at budget {budget}: {base_stocks}"
            )
            assert item["stock_base"] == base_stocks[0]

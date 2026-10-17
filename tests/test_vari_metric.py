"""Tests for two-echelon VARI-METRIC spares optimiser."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from shikra.spares.vari_metric import demand_rates, ebo_poisson, evaluate, vari_metric

ROOT = Path(__file__).parent.parent


def test_ebo_poisson_hand_calculation():
    """Verify ebo_poisson against exact analytical values.

    Arithmetic:
      For X ~ Poisson(mu):
        EBO(s, mu) = E[(X - s)+] = sum_{x=s+1}^inf (x - s) P(X = x)
                   = (mu - s) + sum_{x=0}^{s-1} (s - x) P(X = x)

      Case s = 0:
        E[(X - 0)+] = E[X] = mu.

      Case mu = 2.0, s = 1:
        E[(X - 1)+] = (2 - 1) + (1 - 0) * P(X = 0)
                    = 1 + exp(-2)
                    = 1 + 0.135335283...
                    approx 1.135335283.
    """
    # Check s = 0 returns mu exactly
    assert ebo_poisson(0, 0.0) == 0.0
    assert ebo_poisson(0, 2.5) == 2.5
    assert ebo_poisson(0, 10.0) == 10.0

    # Check mu = 2, s = 1
    expected_2_1 = 1.0 + math.exp(-2.0)
    actual_2_1 = ebo_poisson(1, 2.0)
    assert pytest.approx(actual_2_1, rel=1e-9) == expected_2_1
    assert pytest.approx(actual_2_1, abs=1e-4) == 1.1353

    # Check mu = 1, s = 2:
    # E[(X - 2)+] = (1 - 2) + 2*P(0) + 1*P(1) = -1 + 2*e^-1 + 1*e^-1 = 3*e^-1 - 1
    expected_1_2 = 3.0 * math.exp(-1.0) - 1.0
    assert pytest.approx(ebo_poisson(2, 1.0), rel=1e-9) == expected_1_2


def test_vari_metric_hand_computed_steps():
    """Verify greedy marginal analysis sequence on a hand-computed 2-LRU, 1-base case.

    Setup:
      1 squadron, 1 aircraft (n_tails = 1), sortie_demand = 1.0, transfer_days = 2, pm = 1.0.
      LRU A: eta=10, beta=1 (mean life 10), lam=0.1, cost=10, repair_tat=20 -> mu0=2.0.
      LRU B: eta=20, beta=1 (mean life 20), lam=0.05, cost=5, repair_tat=10 -> mu0=0.5.

    Initial state (cost 0):
      LRU A: s0=0, s1=0, delay=20, mu_base=2.2, base_ebo=2.2.
      LRU B: s0=0, s1=0, delay=10, mu_base=0.6, base_ebo=0.6.

    Step 1 evaluation (Delta EBO / cost):
      - A depot: new mu_base=1.3353, delta=0.8647, metric=0.08647
      - A base: delta = 2.2 - (1.2 + exp(-2.2)) = 0.8892, metric=0.08892
      - B depot: new mu_base=0.2065, delta=0.3935, metric=0.07869
      - B base: delta = 0.6 - (-0.4 + exp(-0.6)) = 0.4512, metric=0.09024  -> WINNER!
      Step 1 picks: B at base (cost = 5, total cost = 5).

    Step 2 evaluation:
      With B base=1, its base EBO is 0.1488.
      Next candidate for B base would be B base=2:
        EBO(2, 0.6) = (0.6 - 2) + 2*e^-0.6 + 1*0.6*e^-0.6 = -1.4 + 2.6*e^-0.6 = 0.0269.
        Delta = 0.1488 - 0.0269 = 0.1219, metric = 0.1219 / 5 = 0.02438.
      Meanwhile:
        A base: metric = 0.08892 (cost 10).
        A depot: metric = 0.08647 (cost 10).
        B depot: delta_base = 0.1488 - EBO(1, 0.2065) = 0.1488 - 0.0189 = 0.1299, metric = 0.02598.
      Step 2 picks: A at base (cost = 10, total cost = 15).

    Step 3 evaluation:
      With A base=1, A depot has delta = (2.2 - 1.3108) - (new EBO(1, 1.3353)) = 0.8892 - ...
      Comparing candidates for remaining budget:
      A depot gives higher reduction per cost than B base=2 or B depot.
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

    # Budget 5: step 1 should choose lru_b at base
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


def test_budget_constraints():
    """Total cost never exceeds budget, and zero budget gives all zeros."""
    with open(ROOT / "config" / "wing.yaml") as f:
        cfg = yaml.safe_load(f)

    # Zero budget
    alloc_zero = vari_metric(cfg, budget=0.0)
    eval_zero = evaluate(cfg, alloc_zero)
    assert eval_zero["cost_lakh"] == 0.0
    for lid, stock in alloc_zero.items():
        assert stock["depot"] == 0
        assert all(s == 0 for s in stock["base"])

    # Small and large budgets
    for budget in [25.0, 50.0, 150.0, 500.0, 1000.0]:
        alloc = vari_metric(cfg, budget=budget)
        ev = evaluate(cfg, alloc)
        assert ev["cost_lakh"] <= budget + 1e-9


def test_ebo_non_increasing_with_budget():
    """Total base EBO is non-increasing as the budget increases across at least 5 points."""
    with open(ROOT / "config" / "wing.yaml") as f:
        cfg = yaml.safe_load(f)

    budgets = [0.0, 100.0, 300.0, 600.0, 1000.0, 1500.0]
    ebos = []
    for b in budgets:
        alloc = vari_metric(cfg, budget=b)
        ev = evaluate(cfg, alloc)
        ebos.append(ev["total_ebo"])

    for i in range(len(ebos) - 1):
        assert ebos[i + 1] <= ebos[i] + 1e-9, f"EBO increased from budget {budgets[i]} to {budgets[i+1]}"


def test_determinism():
    """Identical inputs produce identical allocations."""
    with open(ROOT / "config" / "wing.yaml") as f:
        cfg = yaml.safe_load(f)

    rates = demand_rates(cfg)
    alloc1 = vari_metric(cfg, rates=rates, budget=450.0)
    alloc2 = vari_metric(cfg, rates=rates, budget=450.0)
    assert alloc1 == alloc2

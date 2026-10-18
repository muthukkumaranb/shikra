"""VARI-METRIC two-echelon spares allocation for SHIKRA synthetic fleet.

Mapping between Simulation Mechanics and Two-Echelon METRIC Quantities:
----------------------------------------------------------------------
1. Fleet and Basing Structure:
   - Simulation: 3 squadrons of 18 aircraft (54 tails total), 1 installed LRU
     per tail for each of the 10 LRU types.
   - METRIC: 2-echelon system with 1 central depot (echelon 0) and 3 operational
     bases (echelon 1, j in {1, 2, 3}). Demand is equally partitioned across
     the 3 squadrons (lambda_ij = lambda_i / 3).

2. Demand Generation:
   - Simulation: Daily sorties demanded per tail ~ Bernoulli(p=sortie_demand).
     LRU failures follow Weibull(beta, eta) in flown sorties.
   - METRIC: Stationary Poisson process approximation.
     Mean LRU life in sorties: E[Life] = eta * Gamma(1 + 1/beta).
     Daily fleet demand rate: lambda_i = n_tails * sortie_demand / E[Life].
     Per-squadron daily demand: lambda_ij = lambda_i / n_sqn.

3. Repair and Depot Pipeline:
   - Simulation: When an LRU fails, the unserviceable carcass is sent to depot
     repair. No base repair is performed for LRUs (all failed units repair at
     depot). Repair TAT is max(1, int(repair_tat_days * pipeline_mult)). Repaired
     carcasses satisfy waiting aircraft first-come-first-served (pop(0)), or replenish
     depot stock.
   - METRIC: Central depot repair pipeline:
     T_depot_i = repair_tat_days_i * pipeline_mult.
     Depot pipeline mean: mu0_i = lambda_i * T_depot_i.
     Expected backorders at depot with stock s0_i: EBO0_i(s0_i) = E[(X - s0_i)+],
     where X ~ Poisson(mu0_i).

4. Depot Delay and Base Resupply Pipeline:
   - Simulation: If base has stock, replacement is immediate (1-day change).
     If base is out of stock, spare is requisitioned from depot. If depot has
     stock, transfer takes max(1, int(depot_to_base_days * pipeline_mult)). If depot
     is also stocked out, requisition queues at depot (first-come-first-served)
     until a repaired unit completes, incurring waiting time before transfer.
   - METRIC: Average delay at depot per requisition by Little's law:
     Delay_i = EBO0_i(s0_i) / lambda_i.
     Effective base replenishment time:
     T_base_i = T_transfer + Delay_i, where T_transfer = depot_to_base_days * pipeline_mult.
     Base pipeline mean for squadron j:
     mu_ij = lambda_ij * T_base_i = lambda_ij * (T_transfer + EBO0_i(s0_i) / lambda_i).
     Notice: sum_j mu_ij = lambda_i * T_transfer + EBO0_i(s0_i).

5. Objective Function & Symmetric Mode:
   - Simulation: Minimise Not Mission Capable Supply (NMCS) aircraft. Each tail
     with a missing LRU without shelf replacement enters NMCS status.
   - METRIC: Base expected backorders EBO_ij(s_ij) = E[(X - s_ij)+] with
     X ~ Poisson(mu_ij).
     Total objective: minimise sum_{i} sum_{j=1}^3 EBO_ij(s_ij).
   - Symmetric Mode (Default): The three squadrons are identical in demand and
     configuration, so base stock is allocated in steps of one unit per base
     (3 units at a time across all squadrons), ensuring every base gets the same
     stock. This keeps the allocation expressible as a single stock_base integer
     per LRU, exactly matching the format accepted by the simulation engine
     without requiring any engine code modifications.
"""

from __future__ import annotations

import math
from typing import Any


def demand_rates(config: dict) -> dict[str, float]:
    """Calculate per-LRU fleet daily demand rate under Weibull failure process.

    Parameters
    ----------
    config : dict
        wing.yaml configuration dict.

    Returns
    -------
    dict[str, float]
        Mapping lru_id -> daily fleet failure rate.
        Fleet demand rate: lambda_i = n_tails * sortie_demand / (eta_i * Gamma(1 + 1/beta_i))
    """
    n_sqn = int(config["fleet"]["squadrons"])
    ac_per_sqn = int(config["fleet"]["aircraft_per_sqn"])
    n_tails = n_sqn * ac_per_sqn
    sortie_demand = float(config["ops"]["sortie_demand"])

    rates: dict[str, float] = {}
    for lru in config["lrus"]:
        lid = lru["id"]
        beta = float(lru["weibull_beta"])
        eta = float(lru["weibull_eta"])
        mean_life = eta * math.gamma(1.0 + 1.0 / beta)
        rates[lid] = (n_tails * sortie_demand) / mean_life
    return rates


def ebo_poisson(s: int, mu: float) -> float:
    """Exact Expected Backorders E[(X - s)+] for Poisson(mu).

    Parameters
    ----------
    s : non-negative integer stock level
    mu : Poisson mean parameter (>= 0.0)

    Returns
    -------
    float
        Expected backorders E[(X - s)+] = (mu - s) + sum_{x=0}^{s-1} (s - x) P(X = x).
    """
    if s < 0:
        raise ValueError(f"Stock s must be non-negative, got {s}")
    if mu <= 0.0:
        return 0.0
    if s == 0:
        return float(mu)

    # Exact iterative Poisson probability sum:
    # p_0 = exp(-mu), p_{x} = p_{x-1} * (mu / x)
    p_x = math.exp(-mu)
    correction = s * p_x
    for x in range(1, s):
        p_x *= mu / x
        correction += (s - x) * p_x

    ebo = (mu - s) + correction
    return max(0.0, float(ebo))


def evaluate(config: dict, alloc: dict) -> dict[str, Any]:
    """Evaluate two-echelon METRIC metrics for a given allocation.

    Parameters
    ----------
    config : wing.yaml dict
    alloc  : dict mapping lru_id -> {"base": [s1, s2, s3], "depot": s0}
             or {"base": int, "depot": int}

    Returns
    -------
    dict with:
      - "depot_ebo": {lru_id: float}
      - "base_ebo": {lru_id: [float, ...]}
      - "total_ebo": float
      - "cost_lakh": float
      - "expected_nmcs_per_sqn": [float, ...]
      - "expected_nmcs": float (total fleet expected NMCS aircraft)
    """
    pm = float(config.get("pipeline_mult", 1.0))
    n_sqn = int(config["fleet"]["squadrons"])
    transfer_days = float(config["spares"]["engines"]["depot_to_base_days"]) * pm

    rates = demand_rates(config)
    lru_map = {lru["id"]: lru for lru in config["lrus"]}

    depot_ebo: dict[str, float] = {}
    base_ebo: dict[str, list[float]] = {}
    total_cost_lakh = 0.0
    total_ebo = 0.0
    expected_nmcs_per_sqn = [0.0] * n_sqn

    for lid, stock_spec in alloc.items():
        lru = lru_map[lid]
        cost_unit = float(lru["cost_lakh"])
        s_depot = int(stock_spec.get("depot", 0))

        raw_base = stock_spec.get("base", [0] * n_sqn)
        if isinstance(raw_base, (int, float)):
            s_base = [int(raw_base)] * n_sqn
        else:
            s_base = [int(x) for x in raw_base]

        # Cost
        total_cost_lakh += cost_unit * (s_depot + sum(s_base))

        # Demand and depot pipeline
        lam_fleet = rates[lid]
        lam_sqn = lam_fleet / n_sqn
        t_depot = float(lru["repair_tat_days"]) * pm
        mu0 = lam_fleet * t_depot

        # Depot EBO & delay
        e0 = ebo_poisson(s_depot, mu0)
        depot_ebo[lid] = e0
        delay = e0 / lam_fleet if lam_fleet > 0 else 0.0

        # Base pipeline & EBO
        mu_base = lam_sqn * (transfer_days + delay)
        base_ebos = [ebo_poisson(s_j, mu_base) for s_j in s_base]
        base_ebo[lid] = base_ebos

        for j in range(n_sqn):
            expected_nmcs_per_sqn[j] += base_ebos[j]
            total_ebo += base_ebos[j]

    return {
        "depot_ebo": depot_ebo,
        "base_ebo": base_ebo,
        "total_ebo": total_ebo,
        "cost_lakh": total_cost_lakh,
        "expected_nmcs_per_sqn": expected_nmcs_per_sqn,
        "expected_nmcs": total_ebo,
    }


def vari_metric(
    config: dict,
    demand_rates: dict[str, float] | None = None,
    budget: float = 0.0,
    symmetric: bool = True,
) -> dict[str, dict]:
    """Greedy marginal analysis two-echelon VARI-METRIC spares allocation.

    Starts from all-zero stock; repeatedly allocates the candidate that maximizes:
        (reduction in total base EBO) / cost
    until no candidate fits in the remaining budget.

    In symmetric mode (default):
      Allocates base stock in steps of 1 unit per base simultaneously (n_sqn units
      at a time), ensuring all bases maintain identical stock levels.

    Deterministic tie-breaking:
      1. Largest (delta_ebo / delta_cost)
      2. Largest delta_ebo
      3. lru_id alphabetical order
      4. location ("depot" before "base")

    Parameters
    ----------
    config : wing.yaml dict
    demand_rates : optional dict of daily failure rates (computed if None)
    budget : maximum budget in lakh
    symmetric : bool, default True
        If True, allocate base stock symmetrically across all squadrons (1 unit each).

    Returns
    -------
    dict mapping lru_id -> {"base": [s1, s2, s3], "depot": s0, "stock_base": s_base}
    """
    rates = demand_rates if demand_rates is not None else globals()["demand_rates"](config)

    pm = float(config.get("pipeline_mult", 1.0))
    n_sqn = int(config["fleet"]["squadrons"])
    transfer_days = float(config["spares"]["engines"]["depot_to_base_days"]) * pm

    lrus = config["lrus"]
    lru_ids = [lru["id"] for lru in lrus]
    costs = {lru["id"]: float(lru["cost_lakh"]) for lru in lrus}
    t_depots = {lru["id"]: float(lru["repair_tat_days"]) * pm for lru in lrus}

    # Current allocation state
    alloc: dict[str, dict[str, Any]] = {
        lid: {"depot": 0, "base": [0] * n_sqn, "stock_base": 0} for lid in lru_ids
    }

    # Precompute initial depot and base EBOs
    mu0 = {lid: rates[lid] * t_depots[lid] for lid in lru_ids}

    # State per LRU: depot_ebo, delay, mu_base, base_ebos
    state: dict[str, dict[str, Any]] = {}
    for lid in lru_ids:
        lam_fleet = rates[lid]
        lam_sqn = lam_fleet / n_sqn
        e0 = ebo_poisson(0, mu0[lid])
        delay = e0 / lam_fleet if lam_fleet > 0 else 0.0
        mu_b = lam_sqn * (transfer_days + delay)
        b_ebos = [ebo_poisson(0, mu_b) for _ in range(n_sqn)]
        state[lid] = {
            "depot_ebo": e0,
            "delay": delay,
            "mu_base": mu_b,
            "base_ebos": b_ebos,
        }

    current_cost = 0.0

    while True:
        best_candidate = None
        best_metric = 0.0
        best_delta_ebo = 0.0

        for lid in sorted(lru_ids):
            unit_cost = costs[lid]
            lam_fleet = rates[lid]
            lam_sqn = lam_fleet / n_sqn
            s0 = alloc[lid]["depot"]
            s_bases = alloc[lid]["base"]
            cur_st = state[lid]

            # Candidate 1: Depot +1
            if current_cost + unit_cost <= budget + 1e-9:
                new_e0 = ebo_poisson(s0 + 1, mu0[lid])
                new_delay = new_e0 / lam_fleet if lam_fleet > 0 else 0.0
                new_mu_b = lam_sqn * (transfer_days + new_delay)
                delta_depot = sum(
                    cur_st["base_ebos"][j] - ebo_poisson(s_bases[j], new_mu_b)
                    for j in range(n_sqn)
                )
                if delta_depot > 1e-12:
                    metric = delta_depot / unit_cost
                    if (
                        metric > best_metric + 1e-12
                        or (
                            abs(metric - best_metric) <= 1e-12
                            and delta_depot > best_delta_ebo + 1e-12
                        )
                    ):
                        best_metric = metric
                        best_delta_ebo = delta_depot
                        best_candidate = (lid, "depot", None, delta_depot, unit_cost)

            # Candidate 2: Base allocation
            if symmetric:
                step_cost = n_sqn * unit_cost
                if current_cost + step_cost <= budget + 1e-9:
                    delta_base = sum(
                        cur_st["base_ebos"][j] - ebo_poisson(s_bases[j] + 1, cur_st["mu_base"])
                        for j in range(n_sqn)
                    )
                    if delta_base > 1e-12:
                        metric = delta_base / step_cost
                        if (
                            metric > best_metric + 1e-12
                            or (
                                abs(metric - best_metric) <= 1e-12
                                and delta_base > best_delta_ebo + 1e-12
                            )
                        ):
                            best_metric = metric
                            best_delta_ebo = delta_base
                            best_candidate = (lid, "base_sym", None, delta_base, step_cost)
            else:
                for j in range(n_sqn):
                    if current_cost + unit_cost <= budget + 1e-9:
                        cur_base_ebo = cur_st["base_ebos"][j]
                        new_base_ebo = ebo_poisson(s_bases[j] + 1, cur_st["mu_base"])
                        delta_base = cur_base_ebo - new_base_ebo
                        if delta_base > 1e-12:
                            metric = delta_base / unit_cost
                            if (
                                metric > best_metric + 1e-12
                                or (
                                    abs(metric - best_metric) <= 1e-12
                                    and delta_base > best_delta_ebo + 1e-12
                                )
                            ):
                                best_metric = metric
                                best_delta_ebo = delta_base
                                best_candidate = (lid, "base_asym", j, delta_base, unit_cost)

        if best_candidate is None or best_metric <= 0.0:
            break

        lid, loc_type, j_idx, _, cost_added = best_candidate
        current_cost += cost_added

        lam_fleet = rates[lid]
        lam_sqn = lam_fleet / n_sqn

        if loc_type == "depot":
            alloc[lid]["depot"] += 1
            s0 = alloc[lid]["depot"]
            new_e0 = ebo_poisson(s0, mu0[lid])
            new_delay = new_e0 / lam_fleet if lam_fleet > 0 else 0.0
            new_mu_b = lam_sqn * (transfer_days + new_delay)
            s_bases = alloc[lid]["base"]
            state[lid] = {
                "depot_ebo": new_e0,
                "delay": new_delay,
                "mu_base": new_mu_b,
                "base_ebos": [ebo_poisson(s_bases[j], new_mu_b) for j in range(n_sqn)],
            }
        elif loc_type == "base_sym":
            for j in range(n_sqn):
                alloc[lid]["base"][j] += 1
            alloc[lid]["stock_base"] = alloc[lid]["base"][0]
            s_bases = alloc[lid]["base"]
            cur_mu_b = state[lid]["mu_base"]
            state[lid]["base_ebos"] = [ebo_poisson(s_bases[j], cur_mu_b) for j in range(n_sqn)]
        else:
            assert j_idx is not None
            alloc[lid]["base"][j_idx] += 1
            alloc[lid]["stock_base"] = alloc[lid]["base"][0]
            s_bases = alloc[lid]["base"]
            cur_mu_b = state[lid]["mu_base"]
            state[lid]["base_ebos"][j_idx] = ebo_poisson(s_bases[j_idx], cur_mu_b)

    return alloc

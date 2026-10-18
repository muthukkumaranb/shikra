"""A10 Readiness Forecast & Action Recommendation engine (Contract §6, T1.7).

Provides:
- forecast: Monte Carlo mission-capable (MC) count distribution for lead days 1..7
  using only operational wing state (known returns, shelf stock, conformal RUL quantiles).
  Never uses rul_true.
- rank_actions: Ranks candidate logistical actions to answer the commander's question:
  "What is the probability that squadron S has at least N aircraft mission-capable
   on day D+3, and what single action raises it most?"

All data and parameters are for a synthetic wing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shikra.spares.demand import inverse_removal_prob

N_SQN = 3
AC_PER_SQN = 18
N_AC = N_SQN * AC_PER_SQN
N_ENG_PER_AC = 2


def snapshot_at(policy: str, seed: int, day: int) -> dict:
    """Run one simulation and return the requested end-of-day snapshot."""
    root = Path(__file__).resolve().parents[2]
    with (root / "config" / "wing.yaml").open(encoding="utf-8") as handle:
        import yaml

        config = yaml.safe_load(handle)
    with (root / "artifacts" / "policy_params.json").open(encoding="utf-8") as handle:
        params = json.load(handle)
    rul = pd.read_parquet(root / "artifacts" / "replay_rul.parquet")
    from shikra.rul.variants import apply
    from shikra.sim.engine import run

    config["pipeline_mult"] = 1.7
    result = run(
        policy,
        config,
        apply(rul, 1),
        params,
        seed=seed,
        days=day + 1,
        snapshot_days=[day],
    )
    if result.snapshots is None or day not in result.snapshots:
        raise ValueError(f"Simulation did not produce snapshot for day {day}")
    return result.snapshots[day]


def _weibull_conditional_survival_draw(
    rng: np.random.Generator,
    beta: float,
    eta: float,
    sorties_flown: float,
) -> float:
    """Sample remaining sorties until failure given survival to age sorties_flown.

    Exact inverse transform:
      P(T > a + S | T > a) = exp( (a/eta)^beta - ((a+S)/eta)^beta ) = 1 - U
      => S = eta * ( (a/eta)^beta + E )^(1/beta) - a,  where E ~ Exponential(1).
    """
    exp_draw = rng.exponential(1.0)
    a_norm = (sorties_flown / eta) ** beta
    total_sorties = eta * ((a_norm + exp_draw) ** (1.0 / beta))
    return max(0.0, float(total_sorties - sorties_flown))


def _simulate_mc_trajectory(
    snapshot: dict,
    config: dict,
    params: dict,
    policy: str,
    horizon_days: int,
    rng: np.random.Generator,
    action_override: dict | None = None,
) -> np.ndarray:
    """Simulate one Monte Carlo realization forward from snapshot day for horizon_days.

    Returns:
        np.ndarray of shape (horizon_days, N_SQN) containing MC count per squadron per lead day.
    """
    pm = float(config.get("pipeline_mult", 1.0))
    sortie_p = float(config["ops"]["sortie_demand"])
    maint_cfg = config["maintenance"]
    hangar_slots_base = int(maint_cfg["hangar_slots_per_sqn"])
    planned_change_days = int(maint_cfg["planned_change_days"])
    unscheduled_change_days = int(maint_cfg["unscheduled_change_days"])
    depot_to_base_days = float(config["spares"]["engines"]["depot_to_base_days"])
    transfer_days = max(1, int(depot_to_base_days * pm))
    R_star = float(params.get("R_star", 20.0))

    start_day = int(snapshot["day"])

    # 1. State extraction
    hangar_busy = list(snapshot.get("hangar_busy", [0] * N_SQN))
    hangar_capacity = [hangar_slots_base] * N_SQN

    # Stock
    base_engines = list(snapshot["stock"]["engines"]["base"])
    depot_engines = int(snapshot["stock"]["engines"]["depot"])
    base_lrus = {
        s: {lid: int(cnt) for lid, cnt in snapshot["stock"]["lrus"]["base"][s].items()}
        for s in range(N_SQN)
    }
    depot_lrus = {lid: int(cnt) for lid, cnt in snapshot["stock"]["lrus"]["depot"].items()}

    # Queues
    engine_repair_q = list(snapshot["queues"]["engine_repair"])
    engine_transfers_q = list(snapshot["queues"]["engine_transfers"])
    lru_repair_q = [dict(r) for r in snapshot["queues"]["lru_repair"]]
    lru_transfers_q = [dict(t) for t in snapshot["queues"]["lru_transfers"]]

    # Tail tracking: tail_grounded_until[t] is day when maintenance finishes
    tail_grounded_until = [0] * N_AC
    for t_info in snapshot["tails"]:
        tail_idx = int(t_info["tail"])
        if t_info["status"] != "MC":
            ret = t_info.get("return_day")
            tail_grounded_until[tail_idx] = (
                int(ret) if ret is not None else start_day + horizon_days + 10
            )

    # 2. Engine failure & planned removal sampling
    # eng_cycles_rem[pos]: remaining cycles until removal
    # eng_removal_type[pos]: 'planned' or 'unscheduled'
    eng_cycles_rem = [0.0] * (N_AC * N_ENG_PER_AC)
    eng_removal_type = ["unscheduled"] * (N_AC * N_ENG_PER_AC)

    for eng in snapshot["engines"]:
        pos = int(eng["pos"])
        # Check if installed and not in maintenance
        if eng.get("installed", True) and not eng.get("in_maintenance", False):
            # Sample remaining life from conformal predictive distribution
            u = rng.uniform(0.0, 1.0)
            rul_sample = inverse_removal_prob(eng["q10"], eng["q50"], eng["q90"], u)
            cycles_fail = max(0.0, rul_sample)

            if policy in ("shikra", "shikra_v2"):
                # Planned removal triggered when q10 drops to R_star
                # In each flight cycle, q10 degrades approximately by 1 cycle
                cycles_to_planned = max(0.0, float(eng["q10"]) - R_star)
                if cycles_to_planned <= cycles_fail:
                    eng_cycles_rem[pos] = cycles_to_planned
                    eng_removal_type[pos] = "planned"
                else:
                    eng_cycles_rem[pos] = cycles_fail
                    eng_removal_type[pos] = "unscheduled"
            else:
                eng_cycles_rem[pos] = cycles_fail
                eng_removal_type[pos] = "unscheduled"

    # 3. LRU remaining life sampling
    # lru_sorties_rem[tail][lid] = remaining sorties until failure
    lru_map = {item["id"]: item for item in config["lrus"]}
    lru_sorties_rem: dict[int, dict[str, float]] = {t: {} for t in range(N_AC)}

    for t in range(N_AC):
        t_lrus = snapshot["lrus"].get(t, {})
        for lid, l_cfg in lru_map.items():
            info = t_lrus.get(lid, {})
            if not info.get("missing", False):
                sorties_flown = float(info.get("sorties_flown", 0.0))
                rem = _weibull_conditional_survival_draw(
                    rng,
                    float(l_cfg["weibull_beta"]),
                    float(l_cfg["weibull_eta"]),
                    sorties_flown,
                )
                lru_sorties_rem[t][lid] = rem
            else:
                lru_sorties_rem[t][lid] = 0.0

    # 4. Apply Action Override (if testing an action)
    if action_override:
        atype = action_override.get("type")
        target_sqn = int(action_override.get("sqn", 0))

        if atype == "transfer_engine":
            if depot_engines > 0:
                depot_engines -= 1
                engine_transfers_q.append((start_day + transfer_days, target_sqn))

        elif atype == "transfer_lru":
            target_lid = str(action_override.get("lru_id", "apu_start"))
            if depot_lrus.get(target_lid, 0) > 0:
                depot_lrus[target_lid] -= 1
                lru_transfers_q.append({
                    "arrival_day": start_day + transfer_days,
                    "sqn": target_sqn,
                    "id": target_lid,
                    "tail": -1,
                })

        elif atype == "expedite_repair":
            # Expedite earliest engine repair for target_sqn by 50%
            if engine_repair_q:
                earliest_idx = int(np.argmin(engine_repair_q))
                orig_day = engine_repair_q[earliest_idx]
                rem_days = max(1, orig_day - start_day)
                engine_repair_q[earliest_idx] = start_day + max(1, rem_days // 2)

        elif atype == "defer_planned":
            # Defer planned removal for the first tail whose q10 is still above 0
            sqn_start = target_sqn * AC_PER_SQN
            sqn_end = sqn_start + AC_PER_SQN
            deferred_tail = None
            for pos in range(sqn_start * N_ENG_PER_AC, sqn_end * N_ENG_PER_AC):
                if eng_removal_type[pos] == "planned":
                    eng_snap = snapshot["engines"][pos]
                    if eng_snap["q10"] > 0:
                        deferred_tail = eng_snap["tail"]
                        break
            if deferred_tail is not None:
                for e_idx in range(N_ENG_PER_AC):
                    p = deferred_tail * N_ENG_PER_AC + e_idx
                    if eng_removal_type[p] == "planned":
                        eng_cycles_rem[p] += horizon_days * sortie_p

        elif atype == "add_hangar_slot":
            hangar_capacity[target_sqn] += 1

    # 5. Day-by-day simulation loop
    mc_counts = np.zeros((horizon_days, N_SQN), dtype=int)

    for step in range(1, horizon_days + 1):
        cur_day = start_day + step

        # A. Process repair returns -> depot
        new_depot_eng = sum(1 for r in engine_repair_q if r == cur_day)
        depot_engines += new_depot_eng

        for r in lru_repair_q:
            if r["return_day"] == cur_day:
                depot_lrus[r["id"]] = depot_lrus.get(r["id"], 0) + 1

        # B. Process inbound transfers -> base
        for arr_day, s in engine_transfers_q:
            if arr_day == cur_day:
                base_engines[s] += 1

        for tr in lru_transfers_q:
            if tr["arrival_day"] == cur_day:
                s = tr["sqn"]
                lid = tr["id"]
                base_lrus[s][lid] = base_lrus[s].get(lid, 0) + 1

        # C. Completed maintenance slots freeing up
        for t in range(N_AC):
            if tail_grounded_until[t] == cur_day:
                sqn = t // AC_PER_SQN
                hangar_busy[sqn] = max(0, hangar_busy[sqn] - 1)

        # D. Flown sorties and degradation
        for t in range(N_AC):
            sqn = t // AC_PER_SQN
            is_mc = tail_grounded_until[t] <= cur_day
            if not is_mc:
                continue

            demanded = rng.uniform(0.0, 1.0) < sortie_p
            if not demanded:
                continue

            # Tail flies 1 sortie:
            # 1. Advance engines
            engine_failed = False
            eng_maint_type = "unscheduled"
            for e_idx in range(N_ENG_PER_AC):
                pos = t * N_ENG_PER_AC + e_idx
                eng_cycles_rem[pos] -= 1.0
                if eng_cycles_rem[pos] <= 0:
                    engine_failed = True
                    if eng_removal_type[pos] == "planned":
                        eng_maint_type = "planned"

            # 2. Advance LRUs
            lru_failed = False
            failed_lid = None
            for lid in lru_map:
                lru_sorties_rem[t][lid] -= 1.0
                if lru_sorties_rem[t][lid] <= 0:
                    lru_failed = True
                    failed_lid = lid
                    break

            # 3. Handle Grounding
            if engine_failed:
                change_days = (
                    planned_change_days if eng_maint_type == "planned"
                    else unscheduled_change_days
                )
                if base_engines[sqn] > 0 and hangar_busy[sqn] < hangar_capacity[sqn]:
                    base_engines[sqn] -= 1
                    hangar_busy[sqn] += 1
                    tail_grounded_until[t] = cur_day + change_days
                elif base_engines[sqn] > 0:
                    tail_grounded_until[t] = cur_day + change_days + 1  # queue delay
                elif depot_engines > 0:
                    depot_engines -= 1
                    tail_grounded_until[t] = cur_day + transfer_days + change_days
                else:
                    tail_grounded_until[t] = cur_day + transfer_days + change_days + 5

            elif lru_failed and failed_lid:
                if base_lrus[sqn].get(failed_lid, 0) > 0:
                    base_lrus[sqn][failed_lid] -= 1
                    tail_grounded_until[t] = cur_day + 1
                elif depot_lrus.get(failed_lid, 0) > 0:
                    depot_lrus[failed_lid] -= 1
                    tail_grounded_until[t] = cur_day + transfer_days
                else:
                    tail_grounded_until[t] = cur_day + transfer_days + 3

        # E. Record daily MC count per squadron
        for s in range(N_SQN):
            s_start = s * AC_PER_SQN
            s_end = s_start + AC_PER_SQN
            mc_cnt = sum(1 for t in range(s_start, s_end) if tail_grounded_until[t] <= cur_day)
            mc_counts[step - 1, s] = mc_cnt

    return mc_counts


def forecast(
    snapshot: dict,
    config: dict,
    params: dict,
    policy: str = "shikra",
    horizon_days: int = 7,
    n: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Forecast mission-capable distribution for each squadron over lead days 1..7.

    Uses strictly operational wing knowledge (current tail status, conformal RUL quantiles,
    shelf stock, pending queues). Never reads or uses ``rul_true``.

    Parameters
    ----------
    snapshot     : dict captured by snapshot hook at end of day D
    config       : wing.yaml configuration dict
    params       : policy_params dict
    policy       : fleet policy ('shikra' or 'reactive')
    horizon_days : forecast horizon in days (default 7)
    n            : number of Monte Carlo replications (default 200)
    seed         : integer RNG seed

    Returns
    -------
    pd.DataFrame with columns:
        lead_days (1..7), sqn (1..3), expected_mc, mc_q10, mc_q90, p_ge_6 … p_ge_18
    """
    rng = np.random.default_rng(seed)

    # mc_reps: shape (n, horizon_days, N_SQN)
    mc_reps = np.zeros((n, horizon_days, N_SQN), dtype=int)
    for i in range(n):
        mc_reps[i] = _simulate_mc_trajectory(
            snapshot=snapshot,
            config=config,
            params=params,
            policy=policy,
            horizon_days=horizon_days,
            rng=rng,
        )

    rows = []
    thresholds = list(range(6, 19))

    for k in range(1, horizon_days + 1):
        for s in range(N_SQN):
            vals = mc_reps[:, k - 1, s]
            exp_mc = float(np.mean(vals))
            q10 = float(np.percentile(vals, 10))
            q90 = float(np.percentile(vals, 90))

            row: dict[str, Any] = {
                "lead_days": k,
                "sqn": s + 1,  # 1-indexed squadron S1, S2, S3
                "expected_mc": round(exp_mc, 3),
                "mc_q10": round(q10, 2),
                "mc_q90": round(q90, 2),
            }
            for n_thresh in thresholds:
                p_ge = float(np.mean(vals >= n_thresh))
                row[f"p_ge_{n_thresh}"] = round(p_ge, 4)

            rows.append(row)

    return pd.DataFrame(rows)


def rank_actions(
    snapshot: dict,
    config: dict,
    params: dict,
    sqn: int,
    lead_days: int = 3,
    n: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate and rank candidate operational interventions for a target squadron.

    Answers the commander's question:
    "What single action raises P(>=N MC) / expected MC most on day D+lead_days?"

    Uses Common Random Numbers (CRN) with the baseline forecast.

    Candidates evaluated:
    1. Transfer spare engine from depot to sqn
    2. Transfer scarcest LRU from depot to sqn
    3. Expedite earliest pending engine repair for sqn by 50%
    4. Defer one scheduled planned removal whose q10 is still > 0
    5. Add 1 additional hangar slot for 7 days

    Parameters
    ----------
    snapshot  : snapshot dict at day D
    config    : wing.yaml configuration dict
    params    : policy_params dict
    sqn       : target squadron (1, 2, or 3)
    lead_days : planning horizon lead day (default 3)
    n         : Monte Carlo samples (default 200)
    seed      : integer RNG seed

    Returns
    -------
    pd.DataFrame with top actions:
        rank, action, target, delta_expected_mc, delta_p_ge_N
    """
    s_idx = (sqn - 1) if sqn in (1, 2, 3) else sqn
    sqn_label = f"S{s_idx + 1}"

    # 1. Baseline forecast (no action)
    rng_base = np.random.default_rng(seed)
    base_mc = np.zeros(n)
    for i in range(n):
        traj = _simulate_mc_trajectory(
            snapshot, config, params, policy="shikra", horizon_days=lead_days, rng=rng_base
        )
        base_mc[i] = traj[lead_days - 1, s_idx]

    base_expected = float(np.mean(base_mc))
    # Threshold N chosen around baseline expected MC
    n_target = int(np.clip(round(base_expected), 6, 18))
    base_p_ge = float(np.mean(base_mc >= n_target))

    # Candidate actions
    candidates = [
        {
            "action": "Transfer spare engine depot to squadron",
            "target": f"{sqn_label} Engine Pool",
            "override": {"type": "transfer_engine", "sqn": s_idx},
        },
        {
            "action": "Transfer scarcest LRU (APU starter) depot to squadron",
            "target": f"{sqn_label} APU starter",
            "override": {"type": "transfer_lru", "sqn": s_idx, "lru_id": "apu_start"},
        },
        {
            "action": "Expedite earliest pending engine repair by 50%",
            "target": f"{sqn_label} Depot Repair Queue",
            "override": {"type": "expedite_repair", "sqn": s_idx},
        },
        {
            "action": "Defer planned removal within safety margin (q10 > 0)",
            "target": f"{sqn_label} Engine on Wing",
            "override": {"type": "defer_planned", "sqn": s_idx},
        },
        {
            "action": "Add 1 temporary hangar maintenance slot",
            "target": f"{sqn_label} Maintenance Hangar",
            "override": {"type": "add_hangar_slot", "sqn": s_idx},
        },
    ]

    action_results = []
    for cand in candidates:
        # Run with exact same seed (CRN)
        rng_cand = np.random.default_rng(seed)
        cand_mc = np.zeros(n)
        for i in range(n):
            traj = _simulate_mc_trajectory(
                snapshot,
                config,
                params,
                policy="shikra",
                horizon_days=lead_days,
                rng=rng_cand,
                action_override=cand["override"],
            )
            cand_mc[i] = traj[lead_days - 1, s_idx]

        cand_expected = float(np.mean(cand_mc))
        cand_p_ge = float(np.mean(cand_mc >= n_target))

        delta_mc = cand_expected - base_expected
        delta_p = cand_p_ge - base_p_ge

        action_results.append({
            "action": cand["action"],
            "target": cand["target"],
            "delta_expected_mc": round(delta_mc, 3),
            f"delta_p_ge_{n_target}": round(delta_p, 4),
        })

    # Sort descending by delta_expected_mc
    df_actions = pd.DataFrame(action_results).sort_values("delta_expected_mc", ascending=False).reset_index(drop=True)
    df_actions.insert(0, "rank", range(1, len(df_actions) + 1))
    df_actions.attrs["note"] = "forecast-estimated gain (not validated in simulation)"
    return df_actions

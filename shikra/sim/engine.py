"""SHIKRA fleet discrete-event simulation.

Simulates a 54-aircraft wing (3 squadrons × 18 aircraft) over a one-year
horizon using SimPy with daily resolution.  Four policies are supported:
  reactive    — remove only at failure
  preventive  — remove when installed cycles ≥ T_prev
  shikra      — remove when q10 ≤ R_star (or failure)
  oracle      — remove when rul_true ≤ R_star (or failure)

Common random numbers (CRN): for a given seed, engine trajectory queues
and LRU lifetime draws are identical across all four policies.

Public API
----------
run(policy, config, rul, params, seed, days, overrides) -> SimResult
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimResult:
    """Container for simulation outputs."""
    kpis: dict[str, Any]
    daily: pd.DataFrame
    events: pd.DataFrame
    tail_mc: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

N_SQN = 3
AC_PER_SQN = 18
N_AC = N_SQN * AC_PER_SQN           # 54
N_ENG_PER_AC = 2
N_ENG = N_AC * N_ENG_PER_AC         # 108 engine positions

POLICIES = ("reactive", "preventive", "shikra", "oracle")


def _lognormal_params(mean: float, cv: float) -> tuple[float, float]:
    """Return (mu, sigma) for a lognormal with given arithmetic mean and CV."""
    sigma2 = math.log(1 + cv**2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, math.sqrt(sigma2)


def _weibull_sample(rng: np.random.Generator, beta: float, eta: float) -> float:
    """Draw one Weibull(beta, eta) sample in cycles/sorties."""
    return float(eta * rng.weibull(beta) ** (1.0 / beta))


# ---------------------------------------------------------------------------
# Pre-draw engine trajectory queues
# ---------------------------------------------------------------------------

def _draw_engine_queues(
    rng: np.random.Generator,
    rul_df: pd.DataFrame,
    n_positions: int,
    queue_len: int = 30,
) -> list[list[tuple[str, int]]]:
    """Pre-draw engine trajectory assignment queues.

    Each position gets a list of (subset, unit) pairs drawn with replacement
    from the replay units.

    The initial engine for each position starts at a random cycle in [0, 0.6·life].
    Returns list of length n_positions, each a list of (subset, unit, start_cycle).
    """
    units = rul_df[["subset", "unit"]].drop_duplicates().values.tolist()
    n_units = len(units)
    queues = []
    for _ in range(n_positions):
        idxs = rng.integers(0, n_units, size=queue_len).tolist()
        queue = [tuple(units[i]) for i in idxs]
        queues.append(list(queue))
    return queues  # type: ignore[return-value]


def _draw_lru_queues(
    rng: np.random.Generator,
    lru_list: list[dict],
    n_tails: int,
    queue_len: int = 50,
) -> dict[str, list[list[float]]]:
    """Pre-draw LRU Weibull lifetime queues.

    Returns dict lru_id → list(n_tails lists of queue_len floats).
    """
    lru_queues: dict[str, list[list[float]]] = {}
    for lru in lru_list:
        lid = lru["id"]
        beta = lru["weibull_beta"]
        eta = lru["weibull_eta"]
        tail_queues = []
        for _ in range(n_tails):
            lifetimes = [_weibull_sample(rng, beta, eta) for _ in range(queue_len)]
            tail_queues.append(lifetimes)
        lru_queues[lid] = tail_queues
    return lru_queues


# ---------------------------------------------------------------------------
# Main simulation class
# ---------------------------------------------------------------------------

class FleetSim:
    """Discrete-time (daily) fleet simulation.

    Parameters
    ----------
    policy   : one of POLICIES
    config   : wing.yaml config dict
    rul      : replay RUL DataFrame (possibly oracle/degraded via variants.apply)
    params   : policy_params dict {T_prev, R_star, …}
    seed     : integer seed (1 … 30)
    days     : simulation horizon
    overrides: optional dict to override config values (e.g. pipeline_mult)
    """

    def __init__(
        self,
        policy: str,
        config: dict,
        rul: pd.DataFrame,
        params: dict,
        seed: int,
        days: int = 365,
        overrides: dict | None = None,
    ) -> None:
        assert policy in POLICIES, f"Unknown policy: {policy!r}"
        self.policy = policy
        self.cfg = config.copy()
        if overrides:
            self.cfg.update(overrides)
        self.rul = rul.copy()
        self.params = params
        self.seed = seed
        self.days = days
        self.pm = self.cfg.get("pipeline_mult", 1.0)

        # Build rul lookup: (subset, unit, cycle) → {q10, q50, q90, rul_true}
        self._rul_index = self.rul.set_index(["subset", "unit", "cycle"])

        # Wing parameters
        m = self.cfg["maintenance"]
        self.planned_days = int(m["planned_change_days"])
        self.unscheduled_days = int(m["unscheduled_change_days"])
        eng_r = m["engine_repair"]
        self.repair_mu, self.repair_sigma = _lognormal_params(
            eng_r["mean_days"] * self.pm, eng_r["cv"]
        )
        self.transfer_days = int(
            self.cfg["spares"]["engines"]["depot_to_base_days"] * self.pm
        )
        self.hangar_per_sqn = int(m["hangar_slots_per_sqn"])
        self.sortie_p = self.cfg["ops"]["sortie_demand"]  # 0.8

        self.T_prev = self.params["T_prev"]
        self.R_star = self.params["R_star"]

        self.lru_list = self.cfg["lrus"]

    # ── CRN draws (seed determines everything) ─────────────────────────────

    def _make_rng(self, stream: int) -> np.random.Generator:
        """Create a seeded RNG for a specific stream (avoids cross-contamination)."""
        from numpy.random import SeedSequence

        ss = SeedSequence(self.seed)
        child_seeds = ss.spawn(20)
        return np.random.default_rng(child_seeds[stream])

    # ── RUL lookup helper ──────────────────────────────────────────────────

    def _get_rul(
        self, subset: str, unit: int, cycle: int
    ) -> dict[str, float]:
        """Look up quantile RUL for a given (subset, unit, cycle).

        If the exact cycle is not in the table, uses the nearest available cycle
        (capped at the max cycle in the trajectory).
        """
        traj = self.rul[
            (self.rul["subset"] == subset) & (self.rul["unit"] == unit)
        ]
        if len(traj) == 0:
            return {"q10": 0.0, "q50": 0.0, "q90": 0.0, "rul_true": 0.0}
        max_cycle = traj["cycle"].max()
        c = min(cycle, max_cycle)
        row = traj[traj["cycle"] == c]
        if len(row) == 0:
            # Nearest cycle
            idx = (traj["cycle"] - c).abs().argmin()
            row = traj.iloc[[idx]]
        r = row.iloc[0]
        return {
            "q10": float(r["q10"]),
            "q50": float(r["q50"]),
            "q90": float(r["q90"]),
            "rul_true": float(r["rul_true"]),
        }

    # ── Engine state ───────────────────────────────────────────────────────

    def _should_remove(
        self,
        installed_cycles: int,
        q10: float,
        rul_true: float,
    ) -> str | None:
        """Return removal type ('planned' or None) based on policy."""
        if self.policy == "reactive":
            return None
        if self.policy == "preventive":
            return "planned" if installed_cycles >= self.T_prev else None
        if self.policy == "shikra":
            return "planned" if q10 <= self.R_star else None
        if self.policy == "oracle":
            return "planned" if rul_true <= self.R_star else None
        return None

    # ── Main run ───────────────────────────────────────────────────────────

    def run(self) -> SimResult:
        """Execute the simulation and return a SimResult."""
        rng_eng = self._make_rng(0)       # engine trajectory draws
        rng_lru = self._make_rng(1)       # LRU lifetime draws
        rng_sortie = self._make_rng(2)    # sortie probability draws
        rng_repair = self._make_rng(3)    # repair TAT draws
        rng_init = self._make_rng(4)      # initial cycle offset

        # ── Pre-draw sortie random numbers (tail × day) ──────────────────
        sortie_draws = rng_sortie.uniform(size=(N_AC, self.days))  # (54, 365)

        # ── Pre-draw engine trajectory queues ────────────────────────────
        eng_queues = _draw_engine_queues(rng_eng, self.rul, N_ENG, queue_len=60)

        # ── Pre-draw LRU lifetime queues ──────────────────────────────────
        lru_queues = _draw_lru_queues(rng_lru, self.lru_list, N_AC, queue_len=100)

        # ── Initial engine state ──────────────────────────────────────────
        # Each engine position: {subset, unit, cycle, life, queue_ptr}
        eng_state: list[dict] = []
        for pos in range(N_ENG):
            subset, unit = eng_queues[pos][0]
            traj = self.rul[(self.rul["subset"] == subset) & (self.rul["unit"] == unit)]
            life = int(traj["life"].iloc[0]) if len(traj) > 0 else 100
            start_cycle = int(rng_init.uniform(0, 0.6 * life))
            eng_state.append(
                {
                    "subset": subset,
                    "unit": unit,
                    "cycle": start_cycle,
                    "life": life,
                    "queue_ptr": 1,   # next trajectory to pull
                    "in_maintenance": False,
                    "maint_done_day": None,
                    "removal_type": None,
                    "wasted_life": 0,  # life - cycle at removal for planned
                }
            )

        # ── Spares pools ──────────────────────────────────────────────────
        # base_serviceable[sqn] = count of serviceable spare engines at base
        # depot_serviceable = count at depot
        base_serviceable = [
            int(self.cfg["spares"]["engines"]["base_per_sqn"]) for _ in range(N_SQN)
        ]
        depot_serviceable = int(self.cfg["spares"]["engines"]["depot"])
        # Inbound transfers: list of (arrival_day, sqn)
        inbound_transfers: list[tuple[int, int]] = []
        # Engines in repair: list of (return_day)  → go to depot
        in_repair: list[int] = []

        # ── LRU state ─────────────────────────────────────────────────────
        # lru_state[tail][lru_id] = {sorties_until_fail, missing, queue_ptr}
        lru_state: dict[int, dict[str, dict]] = {}
        for tail in range(N_AC):
            lru_state[tail] = {}
            for lru in self.lru_list:
                lid = lru["id"]
                lru_state[tail][lid] = {
                    "sorties_until_fail": lru_queues[lid][tail][0],
                    "sorties_flown": 0.0,
                    "missing": False,
                    "missing_until": -1,
                    "queue_ptr": 1,
                }

        # ── Hangar slot tracking ──────────────────────────────────────────
        # hangar_busy[sqn] = number of slots in use
        hangar_busy = [0] * N_SQN
        # maintenance queue (waiting for slot): list of (pos)
        maint_queue: list[list[int]] = [[] for _ in range(N_SQN)]

        # ── Output collectors ─────────────────────────────────────────────
        daily_records: list[dict] = []
        event_records: list[dict] = []
        tail_mc_history: list[list[bool]] = []

        planned_removals = 0
        unscheduled_removals = 0
        wasted_life_cycles = 0
        sorties_flown = 0
        sorties_demanded = 0
        aog_days = 0  # total tail-days with 0 sorties due to NMCS/NMCM

        # ── Day loop ─────────────────────────────────────────────────────

        for day in range(self.days):

            # --- Process returns from repair → depot ----------------------
            returned = [r for r in in_repair if r <= day]
            in_repair = [r for r in in_repair if r > day]
            depot_serviceable += len(returned)

            # --- Process inbound transfers → base -------------------------
            arrived = [(d, s) for (d, s) in inbound_transfers if d <= day]
            inbound_transfers = [(d, s) for (d, s) in inbound_transfers if d > day]
            for _, sqn in arrived:
                base_serviceable[sqn] += 1

            # --- Free completed maintenance slots -------------------------
            for pos in range(N_ENG):
                es = eng_state[pos]
                if es["in_maintenance"] and es["maint_done_day"] is not None and es["maint_done_day"] <= day:
                    sqn = pos // (N_ENG_PER_AC * AC_PER_SQN)
                    es["in_maintenance"] = False
                    es["maint_done_day"] = None
                    hangar_busy[sqn] = max(0, hangar_busy[sqn] - 1)
                    # Pull next trajectory from queue
                    qptr = es["queue_ptr"]
                    if qptr >= len(eng_queues[pos]):
                        eng_queues[pos] += eng_queues[pos]  # extend
                    subset, unit = eng_queues[pos][qptr]
                    es["queue_ptr"] = qptr + 1
                    traj = self.rul[
                        (self.rul["subset"] == subset) & (self.rul["unit"] == unit)
                    ]
                    life = int(traj["life"].iloc[0]) if len(traj) > 0 else 100
                    es["subset"] = subset
                    es["unit"] = unit
                    es["cycle"] = 0
                    es["life"] = life
                    es["removal_type"] = None

                    # Assign a spare engine from base
                    if base_serviceable[sqn] > 0:
                        base_serviceable[sqn] -= 1

            # --- Check engine failures and removal triggers ---------------
            for pos in range(N_ENG):
                es = eng_state[pos]
                if es["in_maintenance"]:
                    continue

                sqn = pos // (N_ENG_PER_AC * AC_PER_SQN)
                rul_info = self._get_rul(es["subset"], es["unit"], es["cycle"])

                # Check failure
                failed = es["cycle"] >= es["life"]
                if failed:
                    es["in_maintenance"] = True
                    es["maint_done_day"] = None
                    es["removal_type"] = "unscheduled"
                    unscheduled_removals += 1

                    # Send old engine to repair
                    repair_time = int(
                        np.exp(self.repair_mu + self.repair_sigma * rng_repair.standard_normal())
                    )
                    in_repair.append(day + repair_time)

                    event_records.append(
                        {"day": day, "pos": pos, "type": "unscheduled", "cycle": es["cycle"]}
                    )
                    maint_queue[sqn].append(pos)
                    continue

                # Check policy removal trigger
                removal_type = self._should_remove(
                    es["cycle"],
                    rul_info["q10"],
                    rul_info["rul_true"],
                )
                if removal_type:
                    wasted = max(0, es["life"] - es["cycle"])
                    es["in_maintenance"] = True
                    es["maint_done_day"] = None
                    es["removal_type"] = "planned"
                    planned_removals += 1
                    wasted_life_cycles += wasted

                    repair_time = int(
                        np.exp(self.repair_mu + self.repair_sigma * rng_repair.standard_normal())
                    )
                    in_repair.append(day + repair_time)

                    event_records.append(
                        {"day": day, "pos": pos, "type": "planned", "cycle": es["cycle"],
                         "wasted": wasted}
                    )
                    maint_queue[sqn].append(pos)

            # --- Process maintenance queue (assign available hangar slots) -
            for sqn in range(N_SQN):
                q = maint_queue[sqn]
                while q and hangar_busy[sqn] < self.hangar_per_sqn:
                    pos = q[0]
                    es = eng_state[pos]
                    if not es["in_maintenance"]:
                        q.pop(0)
                        continue
                    # Check spare availability
                    if base_serviceable[sqn] <= 0:
                        # Request from depot
                        if depot_serviceable > 0:
                            depot_serviceable -= 1
                            arrive = day + self.transfer_days
                            inbound_transfers.append((arrive, sqn))
                        # Engine remains grounded until spare arrives — keep at head of queue
                        break
                    q.pop(0)
                    hangar_busy[sqn] += 1
                    maint_type = es["removal_type"] or "unscheduled"
                    change_days = (
                        self.planned_days if maint_type == "planned"
                        else self.unscheduled_days
                    )
                    es["maint_done_day"] = day + change_days

            # --- Compute tail status -------------------------------------
            # tail_mc[tail] = True if MC this day
            tail_mc = []
            for tail in range(N_AC):
                sqn = tail // AC_PER_SQN
                pos0 = tail * N_ENG_PER_AC
                pos1 = pos0 + 1
                es0 = eng_state[pos0]
                es1 = eng_state[pos1]
                both_ok = not es0["in_maintenance"] and not es1["in_maintenance"]

                # Check LRU missing
                lru_ok = all(
                    not lru_state[tail][lru["id"]]["missing"]
                    for lru in self.lru_list
                )

                tail_mc.append(both_ok and lru_ok)
            tail_mc_history.append(list(tail_mc))

            # --- Sorties -------------------------------------------------
            sqn_sorties = [0] * N_SQN
            for tail in range(N_AC):
                sqn = tail // AC_PER_SQN
                demanded = 1 if sortie_draws[tail, day] < self.sortie_p else 0
                sorties_demanded += demanded
                if demanded and tail_mc[tail]:
                    sorties_flown += 1
                    sqn_sorties[sqn] += 1
                elif demanded and not tail_mc[tail]:
                    aog_days += 1

            # --- Advance engine cycles ----------------------------------
            for tail in range(N_AC):
                sqn = tail // AC_PER_SQN
                sortie_happened = (
                    sortie_draws[tail, day] < self.sortie_p and tail_mc[tail]
                )
                if sortie_happened:
                    for eng_off in range(N_ENG_PER_AC):
                        pos = tail * N_ENG_PER_AC + eng_off
                        es = eng_state[pos]
                        if not es["in_maintenance"]:
                            es["cycle"] += 1

                    # Advance LRU sorties
                    for lru in self.lru_list:
                        lid = lru["id"]
                        ls = lru_state[tail][lid]
                        ls["sorties_flown"] += 1.0

            # --- Pre-positioning (shikra and oracle only) -----------------
            if self.policy in ("shikra", "oracle"):
                self._preposition(
                    day, eng_state, base_serviceable, depot_serviceable,
                    inbound_transfers, rng_eng
                )

            # --- LRU failures and logistics --------------------------------
            for tail in range(N_AC):
                for lru in self.lru_list:
                    lid = lru["id"]
                    ls = lru_state[tail][lid]
                    if ls["missing"] and ls["missing_until"] <= day:
                        ls["missing"] = False

                    if not ls["missing"]:
                        if ls["sorties_flown"] >= ls["sorties_until_fail"]:
                            ls["missing"] = True
                            # Determine resupply lead time
                            lead = self._lru_lead_time(lru, day, rng_repair)
                            ls["missing_until"] = day + lead
                            # Advance to next lifetime draw
                            qptr = ls["queue_ptr"]
                            if qptr >= len(lru_queues[lid][tail]):
                                lru_queues[lid][tail] += lru_queues[lid][tail]
                            ls["sorties_until_fail"] = lru_queues[lid][tail][qptr]
                            ls["queue_ptr"] = qptr + 1
                            ls["sorties_flown"] = 0.0

            # --- Record daily status -------------------------------------
            mc = sum(tail_mc)
            nmcm = sum(
                1 for tail in range(N_AC)
                if not tail_mc[tail] and any(
                    eng_state[tail * N_ENG_PER_AC + e]["in_maintenance"]
                    for e in range(N_ENG_PER_AC)
                )
            )
            nmcs = N_AC - mc - nmcm
            daily_records.append(
                {"day": day, "mc": mc, "nmcm": nmcm, "nmcs": nmcs}
            )

        # ── Compute KPIs ─────────────────────────────────────────────────
        daily_df = pd.DataFrame(daily_records)
        events_df = pd.DataFrame(event_records) if event_records else pd.DataFrame(
            columns=["day", "pos", "type", "cycle"]
        )

        serviceability = float(daily_df["mc"].mean() / N_AC)
        mc_rate = serviceability
        nmcm_pct = float(daily_df["nmcm"].mean() / N_AC)
        nmcs_pct = float(daily_df["nmcs"].mean() / N_AC)

        kpis = {
            "policy": self.policy,
            "seed": self.seed,
            "quality": 0 if self.policy == "oracle" else 1,
            "pipeline_mult": self.pm,
            "serviceability": serviceability,
            "mc_rate": mc_rate,
            "nmcm_pct": nmcm_pct,
            "nmcs_pct": nmcs_pct,
            "aog_days": aog_days,
            "unscheduled_removals": unscheduled_removals,
            "planned_removals": planned_removals,
            "wasted_life_cycles": wasted_life_cycles,
            "sorties_flown": sorties_flown,
            "sorties_demanded": sorties_demanded,
            "flying_task_pct": (
                sorties_flown / sorties_demanded if sorties_demanded > 0 else 0.0
            ),
        }

        return SimResult(
            kpis=kpis,
            daily=daily_df,
            events=events_df,
            tail_mc=np.array(tail_mc_history, dtype=bool),
        )

    def _preposition(
        self,
        day: int,
        eng_state: list[dict],
        base_serviceable: list[int],
        depot_serviceable: int,
        inbound_transfers: list[tuple[int, int]],
        rng: np.random.Generator,
    ) -> None:
        """Pre-position spare engines from depot if expected demand exceeds supply."""
        from shikra.spares.demand import removal_prob

        h = self.transfer_days * self.sortie_p  # planning horizon in cycles

        for sqn in range(N_SQN):
            # Count expected removals over horizon h
            expected_removals = 0.0
            ac_start = sqn * AC_PER_SQN
            for tail in range(ac_start, ac_start + AC_PER_SQN):
                for eng_off in range(N_ENG_PER_AC):
                    pos = tail * N_ENG_PER_AC + eng_off
                    es = eng_state[pos]
                    if es["in_maintenance"]:
                        continue
                    rul_info = self._get_rul(es["subset"], es["unit"], es["cycle"])
                    p = removal_prob(
                        rul_info["q10"], rul_info["q50"], rul_info["q90"], h
                    )
                    expected_removals += p

            inbound_sqn = sum(1 for (_, s) in inbound_transfers if s == sqn)
            on_hand = base_serviceable[sqn] + inbound_sqn

            if expected_removals > on_hand and depot_serviceable > 0:
                depot_serviceable -= 1
                arrive = day + self.transfer_days
                inbound_transfers.append((arrive, sqn))

    def _lru_lead_time(
        self, lru: dict, day: int, rng: np.random.Generator
    ) -> int:
        """Determine resupply lead time for an LRU (in days)."""
        pm = self.pm
        # Check base stock (already consumed by the failure)
        # Simple policy: 1 day change (from base), else repair TAT or lead time
        repair_tat = int(lru["repair_tat_days"] * pm)
        return max(1, repair_tat)


# ---------------------------------------------------------------------------
# Public run function
# ---------------------------------------------------------------------------


def run(
    policy: str,
    config: dict,
    rul: pd.DataFrame,
    params: dict,
    seed: int,
    days: int = 365,
    overrides: dict | None = None,
) -> SimResult:
    """Run one simulation replication.

    Parameters
    ----------
    policy   : 'reactive', 'preventive', 'shikra', or 'oracle'
    config   : parsed wing.yaml dict
    rul      : replay RUL DataFrame (apply variants.apply before passing)
    params   : dict with T_prev, R_star
    seed     : integer seed for this replication
    days     : simulation horizon (default 365)
    overrides: optional dict of config overrides

    Returns
    -------
    SimResult with kpis, daily, and events DataFrames
    """
    sim = FleetSim(policy, config, rul, params, seed, days, overrides)
    return sim.run()

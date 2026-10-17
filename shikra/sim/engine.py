"""SHIKRA fleet discrete-event simulation.

Simulates a 54-aircraft wing (3 squadrons × 18 aircraft) over a one-year
horizon with daily resolution.  Four policies are supported:
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
    tail_status: list[list[str]] | None = None


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

N_SQN = 3
AC_PER_SQN = 18
N_AC = N_SQN * AC_PER_SQN           # 54
N_ENG_PER_AC = 2
N_ENG = N_AC * N_ENG_PER_AC         # 108 engine positions

POLICIES = ("reactive", "preventive", "shikra", "oracle", "shikra_v2", "oracle_v2")


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

        # Pre-build O(1) RUL lookup dictionary:
        # (subset, unit) -> (max_cycle, q10_arr, q50_arr, q90_arr, rul_true_arr)
        self._rul_cache: dict[tuple[str, int], tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._engine_lives: dict[tuple[str, int], int] = {}
        for (subset, unit), grp in self.rul.groupby(["subset", "unit"]):
            grp_sorted = grp.sort_values("cycle")
            max_c = int(grp_sorted["cycle"].max())
            life = int(grp_sorted["life"].iloc[0]) if "life" in grp_sorted.columns else max_c
            self._engine_lives[(str(subset), int(unit))] = life
            self._rul_cache[(str(subset), int(unit))] = (
                max_c,
                grp_sorted["q10"].values.astype(np.float64),
                grp_sorted["q50"].values.astype(np.float64),
                grp_sorted["q90"].values.astype(np.float64),
                grp_sorted["rul_true"].values.astype(np.float64),
            )

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

        self.shock = self.cfg.get("shock")
        if self.shock:
            self.shock_start = int(self.shock.get("start", 90))
            self.shock_duration = int(self.shock.get("duration", 180))
            self.shock_end = self.shock_start + self.shock_duration
            self.shock_lrus = set(self.shock.get("lrus", []))
        else:
            self.shock_start = -1
            self.shock_duration = 0
            self.shock_end = -1
            self.shock_lrus = set()

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
        """O(1) look up quantile RUL for a given (subset, unit, cycle)."""
        entry = self._rul_cache.get((str(subset), int(unit)))
        if entry is None:
            return {"q10": 0.0, "q50": 0.0, "q90": 0.0, "rul_true": 0.0}
        max_c, q10_arr, q50_arr, q90_arr, rul_true_arr = entry
        c = min(max(1, cycle), max_c)
        idx = c - 1
        return {
            "q10": float(q10_arr[idx]),
            "q50": float(q50_arr[idx]),
            "q90": float(q90_arr[idx]),
            "rul_true": float(rul_true_arr[idx]),
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
        if self.policy in ("shikra", "shikra_v2"):
            return "planned" if q10 <= self.R_star else None
        if self.policy in ("oracle", "oracle_v2"):
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
            life = self._engine_lives.get((str(subset), int(unit)), 100)
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
                    "installed": True,
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

        # ── LRU state and spares pools ─────────────────────────────────────
        # lru_base_stock[sqn][lru_id] = spare count at base
        # lru_depot_stock[lru_id] = spare count at depot
        lru_base_stock: dict[int, dict[str, int]] = {
            sqn: {lru["id"]: int(lru.get("stock_base", 1)) for lru in self.lru_list}
            for sqn in range(N_SQN)
        }
        lru_depot_stock: dict[str, int] = {
            lru["id"]: int(lru.get("stock_depot", 2)) for lru in self.lru_list
        }
        lru_in_repair: list[dict] = []
        lru_inbound_transfers: list[dict] = []
        lru_waiting: dict[str, list[tuple[int, int]]] = {lru["id"]: [] for lru in self.lru_list}
        lru_failures_shock_window: dict[str, int] = {lru["id"]: 0 for lru in self.lru_list}

        # lru_state[tail][lru_id] = {sorties_until_fail, missing, queue_ptr, has_lru, maint_type}
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
                    "has_lru": True,
                    "maint_type": None,
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
        tail_status_history: list[list[str]] = []

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

            # --- Process LRU returns from repair → depot -------------------
            if self.shock_lrus and self.shock_start <= day < self.shock_end:
                lru_returned = [
                    r for r in lru_in_repair
                    if r["return_day"] <= day and r["id"] not in self.shock_lrus
                ]
                lru_in_repair = [
                    r for r in lru_in_repair
                    if r["return_day"] > day or (r["return_day"] <= day and r["id"] in self.shock_lrus)
                ]
            else:
                lru_returned = [r for r in lru_in_repair if r["return_day"] <= day]
                lru_in_repair = [r for r in lru_in_repair if r["return_day"] > day]
            transfer_days = max(1, int(self.transfer_days * self.pm))
            for r in lru_returned:
                lid = r["id"]
                if lru_waiting[lid]:
                    wait_tail, wait_sqn = lru_waiting[lid].pop(0)
                    ls = lru_state[wait_tail][lid]
                    ls["missing_until"] = day + transfer_days
                    lru_inbound_transfers.append({
                        "arrival_day": day + transfer_days,
                        "sqn": wait_sqn,
                        "id": lid,
                        "tail": wait_tail,
                    })
                else:
                    lru_depot_stock[lid] += 1

            # --- Process inbound LRU transfers → base ----------------------
            lru_arrived = [t for t in lru_inbound_transfers if t["arrival_day"] < day]
            lru_inbound_transfers = [t for t in lru_inbound_transfers if t["arrival_day"] >= day]
            for t in lru_arrived:
                tail = t["tail"]
                lid = t["id"]
                if lru_state[tail][lid]["missing"]:
                    lru_state[tail][lid]["missing"] = False
                    lru_state[tail][lid]["has_lru"] = True
                    lru_state[tail][lid]["maint_type"] = None
                else:
                    lru_base_stock[t["sqn"]][lid] += 1

            # --- Check completed LRU base installation (1-day change) -----
            for tail in range(N_AC):
                for lru in self.lru_list:
                    lid = lru["id"]
                    ls = lru_state[tail][lid]
                    if ls["missing"] and ls["has_lru"] and ls["missing_until"] < day:
                        ls["missing"] = False
                        ls["maint_type"] = None

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
                    life = self._engine_lives.get((str(subset), int(unit)), 100)
                    es["subset"] = subset
                    es["unit"] = unit
                    es["cycle"] = 0
                    es["life"] = life
                    es["removal_type"] = None

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
                    es["installed"] = False
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
                    es["installed"] = False
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
                        inbound_sqn = sum(1 for (_, s) in inbound_transfers if s == sqn)
                        # Request from depot if needed and not already inbound
                        if inbound_sqn < len(q) and depot_serviceable > 0:
                            depot_serviceable -= 1
                            transfer_time = max(1, int(self.transfer_days * self.pm))
                            arrive = day + transfer_time
                            inbound_transfers.append((arrive, sqn))
                        # Engine remains grounded until spare arrives — keep at head of queue
                        break
                    q.pop(0)
                    hangar_busy[sqn] += 1
                    base_serviceable[sqn] -= 1  # Reserve spare when change starts
                    es["installed"] = True
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

            # --- Pre-positioning (shikra, oracle, shikra_v2, oracle_v2) ---
            if self.policy in ("shikra", "oracle", "shikra_v2", "oracle_v2"):
                depot_serviceable = self._preposition(
                    day, eng_state, base_serviceable, depot_serviceable,
                    inbound_transfers, rng_eng
                )

            # --- LRU failures and logistics --------------------------------
            for tail in range(N_AC):
                sqn = tail // AC_PER_SQN
                for lru in self.lru_list:
                    lid = lru["id"]
                    ls = lru_state[tail][lid]

                    if not ls["missing"]:
                        if ls["sorties_flown"] >= ls["sorties_until_fail"]:
                            ls["missing"] = True
                            ls["has_lru"] = False
                            if 90 <= day < 270:
                                lru_failures_shock_window[lid] += 1
                            # Failed unit goes to repair and returns to depot stock
                            repair_tat = max(1, int(lru["repair_tat_days"] * self.pm))
                            if self.shock_lrus and lid in self.shock_lrus and self.shock_start <= day < self.shock_end:
                                return_day = max(day + repair_tat, self.shock_end)
                            else:
                                return_day = day + repair_tat
                            lru_in_repair.append({"return_day": return_day, "id": lid})

                            if lru_base_stock[sqn][lid] > 0:
                                # Take base stock (1-day change, 1 day NMCM)
                                lru_base_stock[sqn][lid] -= 1
                                ls["has_lru"] = True
                                ls["maint_type"] = "nmcm"
                                ls["missing_until"] = day + 1
                            elif lru_depot_stock[lid] > 0:
                                # Transfer from depot (depot_to_base_days * pipeline_mult, NMCS)
                                lru_depot_stock[lid] -= 1
                                transfer_days = max(1, int(self.transfer_days * self.pm))
                                ls["has_lru"] = False
                                ls["maint_type"] = "nmcs"
                                ls["missing_until"] = day + transfer_days
                                lru_inbound_transfers.append({
                                    "arrival_day": day + transfer_days,
                                    "sqn": sqn,
                                    "id": lid,
                                    "tail": tail,
                                })
                            else:
                                # Wait for repair return or new buy (lead_time_days * pipeline_mult, by source)
                                ls["has_lru"] = False
                                ls["maint_type"] = "nmcs"
                                lru_waiting[lid].append((tail, sqn))
                                lead_days = max(1, int(lru["lead_time_days"] * self.pm))
                                transfer_days = max(1, int(self.transfer_days * self.pm))
                                repairs = [r["return_day"] for r in lru_in_repair if r["id"] == lid]
                                if self.shock_lrus and lid in self.shock_lrus and self.shock_start <= day < self.shock_end:
                                    available_repairs = [
                                        r["return_day"] for r in lru_in_repair
                                        if r["id"] == lid and r["return_day"] >= self.shock_end
                                    ]
                                    if available_repairs:
                                        wait_days = max(1, min(available_repairs) - day) + transfer_days
                                    else:
                                        wait_days = max(1, self.shock_end + lead_days - day) + transfer_days
                                else:
                                    if repairs:
                                        wait_days = max(1, min(repairs) - day) + transfer_days
                                    else:
                                        wait_days = lead_days + transfer_days
                                ls["missing_until"] = day + wait_days

                            # Advance to next lifetime draw
                            qptr = ls["queue_ptr"]
                            if qptr >= len(lru_queues[lid][tail]):
                                lru_queues[lid][tail] += lru_queues[lid][tail]
                            ls["sorties_until_fail"] = lru_queues[lid][tail][qptr]
                            ls["queue_ptr"] = qptr + 1
                            ls["sorties_flown"] = 0.0

            # --- Record daily status -------------------------------------
            mc = sum(tail_mc)
            daily_tail_status = []
            for tail in range(N_AC):
                if tail_mc[tail]:
                    daily_tail_status.append("MC")
                else:
                    # NMCM = actively in a hangar slot (change in progress) or LRU change in progress
                    is_nmcm = (
                        any(
                            eng_state[tail * N_ENG_PER_AC + e]["in_maintenance"]
                            and eng_state[tail * N_ENG_PER_AC + e]["maint_done_day"] is not None
                            for e in range(N_ENG_PER_AC)
                        )
                        or any(
                            lru_state[tail][lru_cfg["id"]]["missing"]
                            and lru_state[tail][lru_cfg["id"]].get("maint_type") == "nmcm"
                            for lru_cfg in self.lru_list
                        )
                    )
                    daily_tail_status.append("NMCM" if is_nmcm else "NMCS")
            tail_status_history.append(daily_tail_status)

            nmcm = sum(1 for s in daily_tail_status if s == "NMCM")
            nmcs = sum(1 for s in daily_tail_status if s == "NMCS")
            installed_engines = sum(1 for e in eng_state if e["installed"])
            total_engines = (
                installed_engines
                + sum(base_serviceable)
                + depot_serviceable
                + len(inbound_transfers)
                + len(in_repair)
            )
            total_lrus = {
                lru["id"]: (
                    sum(1 for t in range(N_AC) if lru_state[t][lru["id"]]["has_lru"])
                    + sum(lru_base_stock[s][lru["id"]] for s in range(N_SQN))
                    + lru_depot_stock[lru["id"]]
                    + sum(1 for r in lru_in_repair if r["id"] == lru["id"])
                    + sum(1 for tr in lru_inbound_transfers if tr["id"] == lru["id"])
                )
                for lru in self.lru_list
            }
            daily_records.append(
                {
                    "day": day,
                    "mc": mc,
                    "nmcm": nmcm,
                    "nmcs": nmcs,
                    "total_engines": total_engines,
                    "total_lrus": total_lrus,
                    "lru_missing": {
                        lru["id"]: sum(
                            1 for t in range(N_AC) if lru_state[t][lru["id"]]["missing"]
                        )
                        for lru in self.lru_list
                    },
                }
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
            "quality": 0 if self.policy in ("oracle", "oracle_v2") else 1,
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
            "lru_failures_shock_window": lru_failures_shock_window,
        }

        return SimResult(
            kpis=kpis,
            daily=daily_df,
            events=events_df,
            tail_mc=np.array(tail_mc_history, dtype=bool),
            tail_status=tail_status_history,
        )

    def _preposition(
        self,
        day: int,
        eng_state: list[dict],
        base_serviceable: list[int],
        depot_serviceable: int,
        inbound_transfers: list[tuple[int, int]],
        rng: np.random.Generator,
    ) -> int:
        """Pre-position spare engines from depot if expected demand exceeds supply."""
        from shikra.spares.demand import removal_prob

        if self.policy in ("shikra_v2", "oracle_v2"):
            # Lead-time-aware pre-positioning horizon:
            # h = (depot→base transfer days + planned change days) × pipeline_mult × sortie_demand, in cycles
            h = (self.transfer_days + self.planned_days) * self.pm * self.sortie_p
        else:
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
                transfer_time = max(1, int(self.transfer_days * self.pm))
                arrive = day + transfer_time
                inbound_transfers.append((arrive, sqn))

        return depot_serviceable

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

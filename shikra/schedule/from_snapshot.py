"""Build scheduling Problem instances from readiness simulation snapshots.

Converts operational snapshots of fleet state (engine maintenance queues,
in-progress engine changes, and spare arrival ETAs) into hangar scheduling problems
ready for CP-SAT optimization or baseline dispatch.

All scheduling models and synthetic wing assumptions represent a synthetic wing.
"""

from __future__ import annotations

from typing import Any

from shikra.schedule.cpsat import Job, Problem

AC_PER_SQN = 18
DEFAULT_HANGAR_SLOTS = 3
DEFAULT_PLANNED_DAYS = 2
DEFAULT_UNSCHEDULED_DAYS = 5


def problem_from_snapshot(
    snapshot: dict[str, Any],
    config: dict[str, Any] | None = None,
    sqn: int | None = None,
    capacity: int | None = None,
    horizon: int | None = None,
    due_lead_days: int = 7,
    name: str = "HangarMaintenance",
) -> Problem:
    """Build a hangar bay scheduling Problem from a simulation snapshot.

    Parameters
    ----------
    snapshot : dict
        Fleet state snapshot at day D, matching shikra/sim/engine.py:
        - "day": current day int
        - "engines": list of dicts with {"pos", "tail", "sqn", "in_maintenance", "maint_done_day", "removal_type"}
        - "queues": dict with "engine_maint_queue", "engine_transfers", etc.
        - "hangar_busy": list of current occupied slots per squadron [busy_0, busy_1, busy_2]
    config : dict, optional
        Configuration dict (e.g. from wing.yaml) containing maintenance parameters:
        - maintenance.hangar_slots_per_sqn
        - maintenance.planned_change_days
        - maintenance.unscheduled_change_days
    sqn : int, optional
        Squadron filter (0, 1, or 2). If provided, only includes jobs
        for engines/tails belonging to that squadron. If None, includes all squadrons.
    capacity : int, optional
        Bay capacity override. If None, derived from hangar_slots_per_sqn minus hangar_busy.
    due_lead_days : int, optional
        Target completion horizon in days relative to current day (default 7).
    name : str, optional
        Problem instance name.

    Returns
    -------
    Problem
        Problem instance with jobs and hangar bay capacity.
    """
    cur_day = int(snapshot.get("day", 0))

    if sqn is not None and sqn not in (0, 1, 2):
        raise ValueError(f"Squadron index must be 0, 1, or 2, got {sqn}")

    # Parse maintenance configs
    maint_cfg = (config or {}).get("maintenance", {})
    hangar_slots = int(maint_cfg.get("hangar_slots_per_sqn", DEFAULT_HANGAR_SLOTS))
    planned_change_days = int(maint_cfg.get("planned_change_days", DEFAULT_PLANNED_DAYS))
    unscheduled_change_days = int(maint_cfg.get("unscheduled_change_days", DEFAULT_UNSCHEDULED_DAYS))

    # Parse hangar_busy
    hangar_busy = snapshot.get("hangar_busy", [0, 0, 0])

    # Determine capacity (subtracting hangar_busy unless explicitly overridden)
    if capacity is not None:
        prob_capacity = max(1, int(capacity))
    elif sqn is not None:
        busy = hangar_busy[sqn] if sqn < len(hangar_busy) else 0
        prob_capacity = max(1, hangar_slots - busy)
    else:
        # Whole wing: 3 squadrons
        total_busy = sum(hangar_busy) if hangar_busy else 0
        prob_capacity = max(1, hangar_slots * 3 - total_busy)

    # Parse inbound transfers per squadron: engine_transfers are (eta_day, sqn) tuples
    queues = snapshot.get("queues", {})
    engine_transfers = queues.get("engine_transfers", [])
    sqn_transfers: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for tr in engine_transfers:
        if isinstance(tr, (tuple, list)) and len(tr) >= 2:
            eta, tr_sqn = int(tr[0]), int(tr[1])
            if tr_sqn in sqn_transfers and eta >= cur_day:
                sqn_transfers[tr_sqn].append(eta)
        elif isinstance(tr, dict):
            eta = int(tr.get("eta", tr.get("eta_day", 0)))
            tr_sqn = int(tr.get("sqn", 0))
            if tr_sqn in sqn_transfers and eta >= cur_day:
                sqn_transfers[tr_sqn].append(eta)
    for s in sqn_transfers:
        sqn_transfers[s].sort()

    # Parse maintenance queues: engine_maint_queue holds engine positions pos
    maint_queues = queues.get("engine_maint_queue", [[], [], []])
    queued_positions_by_sqn: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for s_idx, q in enumerate(maint_queues):
        if s_idx in queued_positions_by_sqn and isinstance(q, list):
            for item in q:
                if isinstance(item, dict):
                    pos = int(item["pos"]) if "pos" in item else int(item["tail"]) * 2
                else:
                    pos = int(item)
                queued_positions_by_sqn[s_idx].append(pos)

    all_queued_positions = {p for q in queued_positions_by_sqn.values() for p in q}

    jobs: list[Job] = []
    seen_pos: set[int] = set()

    # 1. Build jobs from snapshot["engines"] where in_maintenance=True
    engines_list = snapshot.get("engines", [])
    for eng in engines_list:
        if not eng.get("in_maintenance", False):
            continue

        pos = int(eng["pos"])
        t_idx = int(eng.get("tail", pos // 2))
        t_sqn = int(eng.get("sqn", pos // 36))

        if sqn is not None and t_sqn != sqn:
            continue

        seen_pos.add(pos)
        maint_type = eng.get("removal_type", "unscheduled")
        base_duration = planned_change_days if maint_type == "planned" else unscheduled_change_days

        # Check if already underway in hangar or waiting in queue
        if pos in all_queued_positions:
            k = queued_positions_by_sqn[t_sqn].index(pos)
            release = max(0, sqn_transfers[t_sqn][k] - cur_day) if k < len(sqn_transfers[t_sqn]) else 0
            duration = base_duration
        else:
            # Underway in bay
            release = 0
            ret_day = eng.get("maint_done_day")
            if ret_day is not None and ret_day > cur_day:
                duration = max(1, int(ret_day) - cur_day)
            else:
                duration = base_duration

        due = max(release + duration, due_lead_days)
        jobs.append(
            Job(
                job_id=f"pos_{pos:03d}",
                tail=t_idx,
                duration=duration,
                release_date=release,
                due_date=due,
                weight=1.0,
                sqn=t_sqn,
                job_type="engine_change",
            )
        )

    # 2. Build jobs from engine_maint_queue for any positions not in engines_list
    for s_idx, q in queued_positions_by_sqn.items():
        if sqn is not None and s_idx != sqn:
            continue
        for k, pos in enumerate(q):
            if pos in seen_pos:
                continue
            seen_pos.add(pos)
            t_idx = pos // 2
            t_sqn = pos // 36
            release = max(0, sqn_transfers[s_idx][k] - cur_day) if k < len(sqn_transfers[s_idx]) else 0
            duration = unscheduled_change_days
            due = max(release + duration, due_lead_days)

            jobs.append(
                Job(
                    job_id=f"pos_{pos:03d}",
                    tail=t_idx,
                    duration=duration,
                    release_date=release,
                    due_date=due,
                    weight=1.0,
                    sqn=t_sqn,
                    job_type="engine_change",
                )
            )

    # Sort jobs deterministically by tail ID, then job ID
    jobs.sort(key=lambda j: (j.tail, j.job_id))

    prob_name = (
        f"{name}_Sqn{sqn}_Day{cur_day}"
        if sqn is not None
        else f"{name}_Wing_Day{cur_day}"
    )

    # Planning horizon
    if horizon is not None:
        prob_horizon = max(1, int(horizon))
    else:
        max_job_span = max((j.release_date + j.duration for j in jobs), default=0)
        total_duration = sum(j.duration for j in jobs)
        prob_horizon = max(due_lead_days, max_job_span + total_duration + 14)

    return Problem(name=prob_name, jobs=jobs, capacity=prob_capacity, horizon=prob_horizon)


# Alias for backward/forward compatibility
build_problem_from_snapshot = problem_from_snapshot

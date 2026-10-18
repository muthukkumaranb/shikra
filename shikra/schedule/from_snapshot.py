"""Build scheduling Problem instances from readiness simulation snapshots.

Converts operational snapshots of fleet state (grounded aircraft, maintenance queues,
in-progress engine/LRU changes, and spare arrival ETAs) into hangar scheduling problems
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
        Fleet state snapshot at day D, containing:
        - "day": current day int
        - "tails": list of dicts with {"tail": int, "status": str, "return_day": int | None}
        - "queues": dict with "engine_maint_queue", "engine_transfers", etc.
        - "engines": (optional) list of dicts with engine status
        - "hangar_busy": (optional) list of current occupied slots per squadron
    config : dict, optional
        Configuration dict (e.g. from wing.yaml) containing maintenance parameters:
        - maintenance.hangar_slots_per_sqn
        - maintenance.planned_change_days
        - maintenance.unscheduled_change_days
    sqn : int, optional
        Squadron filter. If provided (1, 2, or 3, or 0, 1, 2), only includes jobs
        for tails belonging to that squadron. If None, includes all squadrons.
    capacity : int, optional
        Bay capacity override. If None, derived from config or default (3 slots/sqn).
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

    # Parse maintenance configs
    maint_cfg = (config or {}).get("maintenance", {})
    hangar_slots = int(maint_cfg.get("hangar_slots_per_sqn", DEFAULT_HANGAR_SLOTS))
    planned_change_days = int(maint_cfg.get("planned_change_days", DEFAULT_PLANNED_DAYS))
    unscheduled_change_days = int(maint_cfg.get("unscheduled_change_days", DEFAULT_UNSCHEDULED_DAYS))

    # Resolve target squadron if specified (normalize to 1-indexed for display)
    target_sqn_1indexed = None
    if sqn is not None:
        target_sqn_1indexed = sqn if sqn in (1, 2, 3) else (sqn + 1)

    # Determine capacity
    if capacity is not None:
        prob_capacity = max(1, int(capacity))
    elif target_sqn_1indexed is not None:
        prob_capacity = hangar_slots
    else:
        # Whole wing: 3 squadrons
        prob_capacity = hangar_slots * 3

    # Map tails to their squadron index (0, 1, 2)
    def tail_sqn_1indexed(tail_idx: int) -> int:
        return (tail_idx // AC_PER_SQN) + 1

    # Extract pending inbound spare ETAs if available
    tail_eta: dict[int, int] = {}
    queues = snapshot.get("queues", {})
    for transfer in queues.get("engine_transfers", []):
        if isinstance(transfer, dict) and "tail" in transfer and "eta" in transfer:
            tail_eta[int(transfer["tail"])] = int(transfer["eta"])

    jobs: list[Job] = []
    seen_tails: set[int] = set()

    # 1. Inspect tails in snapshot
    tails_list = snapshot.get("tails", [])
    for t_info in tails_list:
        t_idx = int(t_info["tail"])
        t_sqn = tail_sqn_1indexed(t_idx)

        if target_sqn_1indexed is not None and t_sqn != target_sqn_1indexed:
            continue

        status = t_info.get("status", "MC")
        ret_day = t_info.get("return_day")

        if status in ("NMCM", "NMCS") or ret_day is not None:
            seen_tails.add(t_idx)

            # Release date: 0 if already at base, or ETA if waiting for transfer
            release = 0
            if t_idx in tail_eta:
                release = max(0, tail_eta[t_idx] - cur_day)

            # Duration: remaining if in progress, else change days
            if ret_day is not None and ret_day > cur_day:
                duration = max(1, int(ret_day) - cur_day)
            elif status == "NMCM":
                duration = planned_change_days
            else:
                duration = unscheduled_change_days

            due = max(release + duration, due_lead_days)

            jobs.append(
                Job(
                    job_id=f"tail_{t_idx:02d}",
                    tail=t_idx,
                    duration=duration,
                    release_date=release,
                    due_date=due,
                    weight=1.0,
                    sqn=t_sqn,
                    job_type="engine_change",
                )
            )

    # 2. Inspect engine maintenance queues for any waiting tails not yet in tails_list
    maint_queues = queues.get("engine_maint_queue", [])
    for s_idx, sqn_q in enumerate(maint_queues):
        sqn_num = s_idx + 1
        if target_sqn_1indexed is not None and sqn_num != target_sqn_1indexed:
            continue

        if isinstance(sqn_q, list):
            for item in sqn_q:
                t_idx = int(item["tail"]) if isinstance(item, dict) and "tail" in item else int(item)
                if t_idx not in seen_tails:
                    seen_tails.add(t_idx)
                    duration = (
                        planned_change_days
                        if isinstance(item, dict) and item.get("type") == "planned"
                        else unscheduled_change_days
                    )
                    release = 0
                    if t_idx in tail_eta:
                        release = max(0, tail_eta[t_idx] - cur_day)
                    due = max(release + duration, due_lead_days)

                    jobs.append(
                        Job(
                            job_id=f"tail_{t_idx:02d}",
                            tail=t_idx,
                            duration=duration,
                            release_date=release,
                            due_date=due,
                            weight=1.0,
                            sqn=sqn_num,
                            job_type="engine_change",
                        )
                    )

    # Sort jobs deterministically by tail ID
    jobs.sort(key=lambda j: j.tail)

    prob_name = (
        f"{name}_Sqn{target_sqn_1indexed}_Day{cur_day}"
        if target_sqn_1indexed is not None
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

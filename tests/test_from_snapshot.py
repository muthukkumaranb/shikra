"""Unit tests for shikra/schedule/from_snapshot.py."""

from __future__ import annotations

import pytest

from shikra.schedule.cpsat import edf_baseline, evaluate, solve
from shikra.schedule.from_snapshot import problem_from_snapshot


def test_empty_snapshot_produces_empty_problem() -> None:
    snapshot = {
        "day": 100,
        "hangar_busy": [0, 0, 0],
        "engines": [{"pos": p, "tail": p // 2, "sqn": p // 36, "in_maintenance": False} for p in range(108)],
        "queues": {"engine_maint_queue": [[], [], []], "engine_transfers": []},
    }
    prob = problem_from_snapshot(snapshot, sqn=0)
    assert len(prob.jobs) == 0
    assert prob.capacity == 3

    res = solve(prob)
    assert res.makespan == 0
    assert res.total_weighted_tardiness == 0.0


def test_real_snapshot_format_and_capacity_subtraction() -> None:
    # Day 50
    # Pos 4 (tail 2, sqn 0): in maintenance in hangar, maint_done_day=53 -> duration=3, release=0
    # Pos 10 (tail 5, sqn 0): queued in engine_maint_queue[0], unscheduled -> duration=5
    # Pos 14 (tail 7, sqn 0): queued in engine_maint_queue[0], planned -> duration=2
    # Pos 38 (tail 19, sqn 1): in maintenance in Sqn 1
    # Tail 12 (sqn 0): NMCS waiting ONLY for an LRU, NO engine maintenance -> NOT a bay job!
    snapshot = {
        "day": 50,
        "hangar_busy": [1, 1, 0],
        "tails": [
            {"tail": 12, "sqn": 0, "status": "NMCS", "return_day": 55},
        ],
        "engines": [
            {"pos": 4, "tail": 2, "sqn": 0, "in_maintenance": True, "maint_done_day": 53, "removal_type": "planned"},
            {"pos": 10, "tail": 5, "sqn": 0, "in_maintenance": True, "maint_done_day": None, "removal_type": "unscheduled"},
            {"pos": 14, "tail": 7, "sqn": 0, "in_maintenance": True, "maint_done_day": None, "removal_type": "planned"},
            {"pos": 38, "tail": 19, "sqn": 1, "in_maintenance": True, "maint_done_day": 54, "removal_type": "unscheduled"},
        ],
        "queues": {
            "engine_maint_queue": [[10, 14], [], []],
            "engine_transfers": [(52, 0)],  # Sqn 0 inbound transfer at day 52 -> release = 52 - 50 = 2
        },
    }

    # Test Sqn 0 only
    prob_s0 = problem_from_snapshot(snapshot, sqn=0)
    # Default hangar_slots_per_sqn = 3, hangar_busy[0] = 1 -> capacity = 2
    assert prob_s0.capacity == 2

    # Should include tail 2 (pos 4), tail 5 (pos 10), and tail 7 (pos 14).
    # Tail 12 is NMCS waiting only for an LRU and must NOT be a bay job.
    job_tails = {j.tail: j for j in prob_s0.jobs}
    assert set(job_tails.keys()) == {2, 5, 7}
    assert 12 not in job_tails

    # Pos 4 (tail 2)
    assert job_tails[2].duration == 3
    assert job_tails[2].release_date == 0
    assert job_tails[2].sqn == 0

    # Pos 10 (tail 5): 1st in queue, matches (52, 0) -> release = 2
    assert job_tails[5].duration == 5
    assert job_tails[5].release_date == 2
    assert job_tails[5].sqn == 0

    # Pos 14 (tail 7): 2nd in queue, no remaining inbound transfer -> release = 0
    assert job_tails[7].duration == 2
    assert job_tails[7].release_date == 0
    assert job_tails[7].sqn == 0

    # Test solving
    res_opt = solve(prob_s0)
    eval_opt = evaluate(prob_s0, res_opt)
    assert eval_opt["is_valid"] is True

    res_edf = edf_baseline(prob_s0)
    eval_edf = evaluate(prob_s0, res_edf)
    assert eval_edf["is_valid"] is True


def test_invalid_sqn_index_raises() -> None:
    snapshot = {
        "day": 0,
        "hangar_busy": [0, 0, 0],
        "engines": [],
        "queues": {"engine_maint_queue": [[], [], []], "engine_transfers": []},
    }
    with pytest.raises(ValueError, match="Squadron index must be 0, 1, or 2"):
        problem_from_snapshot(snapshot, sqn=3)

    with pytest.raises(ValueError, match="Squadron index must be 0, 1, or 2"):
        problem_from_snapshot(snapshot, sqn=-1)


def test_wing_wide_snapshot() -> None:
    snapshot = {
        "day": 20,
        "hangar_busy": [1, 2, 0],
        "engines": [
            {"pos": 2, "tail": 1, "sqn": 0, "in_maintenance": True, "maint_done_day": 22},
            {"pos": 40, "tail": 20, "sqn": 1, "in_maintenance": True, "maint_done_day": 23},
            {"pos": 80, "tail": 40, "sqn": 2, "in_maintenance": True, "maint_done_day": None},
        ],
        "queues": {"engine_maint_queue": [[], [], [80]], "engine_transfers": []},
    }
    prob = problem_from_snapshot(snapshot, sqn=None)
    assert len(prob.jobs) == 3
    # 3 * 3 - (1 + 2 + 0) = 6
    assert prob.capacity == 6

    res = solve(prob)
    eval_res = evaluate(prob, res)
    assert eval_res["is_valid"] is True


def test_integration_readiness_snapshot_at() -> None:
    try:
        from shikra.strategy.readiness import snapshot_at
    except ImportError:
        pytest.skip("shikra.strategy.readiness.snapshot_at is not importable on this branch")

    snapshot = snapshot_at("shikra_v2", seed=1, day=485)
    prob = problem_from_snapshot(snapshot, sqn=None)

    # Assert every job tail is in 0..53
    for job in prob.jobs:
        assert 0 <= job.tail <= 53

    # Assert job count equals the number of engines in maintenance or queued
    engines_in_maint = [e for e in snapshot.get("engines", []) if e.get("in_maintenance")]
    queued_pos = [p for q in snapshot.get("queues", {}).get("engine_maint_queue", []) for p in q]
    expected_count = len({e["pos"] for e in engines_in_maint} | set(queued_pos))
    assert len(prob.jobs) == expected_count

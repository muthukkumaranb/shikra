"""Unit tests for shikra/schedule/from_snapshot.py."""

from __future__ import annotations

from shikra.schedule.cpsat import edf_baseline, evaluate, solve
from shikra.schedule.from_snapshot import problem_from_snapshot


def test_empty_snapshot_produces_empty_problem() -> None:
    snapshot = {
        "day": 100,
        "tails": [{"tail": t, "status": "MC", "return_day": None} for t in range(54)],
        "queues": {"engine_maint_queue": [[], [], []], "engine_transfers": []},
    }
    prob = problem_from_snapshot(snapshot, sqn=1)
    assert len(prob.jobs) == 0
    assert prob.capacity == 3

    res = solve(prob)
    assert res.makespan == 0
    assert res.total_weighted_tardiness == 0.0


def test_grounded_tails_parsed_correctly() -> None:
    snapshot = {
        "day": 50,
        "tails": [
            {"tail": 2, "status": "NMCM", "return_day": 53},  # remaining = 3
            {"tail": 5, "status": "NMCS", "return_day": None},  # unscheduled = 5
            {"tail": 19, "status": "NMCM", "return_day": None},  # Sqn 2
        ],
        "queues": {
            "engine_maint_queue": [[{"tail": 7, "type": "planned"}], [], []],
            "engine_transfers": [{"tail": 5, "eta": 52}],  # arrives at day 52 -> release = 2
        },
    }

    # Test Sqn 1 only (tails 0..17)
    prob_s1 = problem_from_snapshot(snapshot, sqn=1, capacity=2)
    assert prob_s1.capacity == 2
    # Should include tail 2, tail 5, and tail 7 (from queue)
    job_tails = {j.tail: j for j in prob_s1.jobs}
    assert set(job_tails.keys()) == {2, 5, 7}

    assert job_tails[2].duration == 3
    assert job_tails[2].release_date == 0
    assert job_tails[2].sqn == 1

    assert job_tails[5].duration == 5
    assert job_tails[5].release_date == 2
    assert job_tails[5].sqn == 1

    assert job_tails[7].duration == 2
    assert job_tails[7].release_date == 0

    # Test solving this extracted problem
    res_opt = solve(prob_s1)
    eval_opt = evaluate(prob_s1, res_opt)
    assert eval_opt["is_valid"] is True

    res_edf = edf_baseline(prob_s1)
    eval_edf = evaluate(prob_s1, res_edf)
    assert eval_edf["is_valid"] is True


def test_wing_wide_snapshot() -> None:
    snapshot = {
        "day": 20,
        "tails": [
            {"tail": 1, "status": "NMCM", "return_day": 22},
            {"tail": 20, "status": "NMCM", "return_day": 23},
            {"tail": 40, "status": "NMCS", "return_day": None},
        ],
        "queues": {"engine_maint_queue": [[], [], []], "engine_transfers": []},
    }
    prob = problem_from_snapshot(snapshot, sqn=None)
    assert len(prob.jobs) == 3
    assert prob.capacity == 9  # 3 sqns * 3 slots

    res = solve(prob)
    eval_res = evaluate(prob, res)
    assert eval_res["is_valid"] is True

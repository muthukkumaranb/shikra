"""Unit tests for CP-SAT hangar maintenance scheduling."""

from __future__ import annotations

import random

import pytest

from shikra.schedule.cpsat import Job, Problem, edf_baseline, evaluate, solve


def test_hand_built_known_optimum_single_bay() -> None:
    """Test where CP-SAT finds known optimum beating EDF on a 1-bay instance.

    Arithmetic:
      Hangar capacity: 1 bay.
      Job A: duration = 4, release = 0, due_date = 4, weight = 1.0
      Job B: duration = 2, release = 0, due_date = 5, weight = 10.0

      EDF sorts by due_date:
        A has due_date=4, B has due_date=5 -> schedules A first:
        A runs [0, 4) -> completion 4 <= due 4 -> tardiness = 0
        B runs [4, 6) -> completion 6 > due 5  -> tardiness = 1, weighted = 10.0
        Total weighted tardiness under EDF = 10.0.

      CP-SAT schedules B first:
        B runs [0, 2) -> completion 2 <= due 5 -> tardiness = 0
        A runs [2, 6) -> completion 6 > due 4  -> tardiness = 2, weighted = 2.0
        Total weighted tardiness under CP-SAT = 2.0.

    Optimum is strictly 2.0, which CP-SAT must achieve.
    """
    job_a = Job(job_id="A", tail=101, duration=4, release_date=0, due_date=4, weight=1.0)
    job_b = Job(job_id="B", tail=102, duration=2, release_date=0, due_date=5, weight=10.0)

    problem = Problem(jobs=[job_a, job_b], capacity=1, horizon=10, name="hand_built_opt")

    # 1. EDF baseline
    edf_res = edf_baseline(problem)
    assert edf_res.total_weighted_tardiness == pytest.approx(10.0)
    assert edf_res.jobs[0].job_id == "A"
    assert edf_res.jobs[1].job_id == "B"

    # 2. CP-SAT solver
    cpsat_res = solve(problem, time_limit_s=5.0)
    assert cpsat_res.status == "OPTIMAL"
    assert cpsat_res.method == "cpsat"
    assert cpsat_res.metadata.get("solver") == "ortools_cpsat"
    assert cpsat_res.total_weighted_tardiness == pytest.approx(2.0)

    # Job B must be scheduled first
    jobs_by_id = {j.job_id: j for j in cpsat_res.jobs}
    assert jobs_by_id["B"].assigned_start == 0
    assert jobs_by_id["B"].assigned_end == 2
    assert jobs_by_id["A"].assigned_start == 2
    assert jobs_by_id["A"].assigned_end == 6

    # Validation
    eval_res = evaluate(problem, cpsat_res)
    assert eval_res["is_valid"] is True


def test_ortools_path_on_hand_built_optimum() -> None:
    """Explicit test verifying the OR-Tools solver path finds known optimum 2.0 with method='cpsat'."""
    job_a = Job(job_id="A", tail=101, duration=4, release_date=0, due_date=4, weight=1.0)
    job_b = Job(job_id="B", tail=102, duration=2, release_date=0, due_date=5, weight=10.0)
    problem = Problem(jobs=[job_a, job_b], capacity=1, horizon=10, name="hand_built_opt")

    result = solve(problem, max_deterministic_time=5.0)
    assert result.status == "OPTIMAL"
    assert result.method == "cpsat"
    assert result.metadata.get("solver") == "ortools_cpsat"
    assert result.total_weighted_tardiness == pytest.approx(2.0)


def test_capacity_limits() -> None:
    """Test that parallel hangar bay capacity constraints are strictly respected.

    With capacity=2 bays and 3 jobs of duration 3 released at 0:
      At most 2 jobs can be in hangar concurrently.
      Makespan must be at least 6.
    """
    jobs = [
        Job(job_id=f"J{i}", tail=100 + i, duration=3, release_date=0, due_date=3)
        for i in range(3)
    ]
    problem = Problem(jobs=jobs, capacity=2, horizon=10, name="capacity_test")

    res = solve(problem)
    assert res.status == "OPTIMAL"
    assert res.makespan >= 6

    eval_res = evaluate(problem, res)
    assert eval_res["is_valid"] is True

    # Check concurrent usage at each day
    for day in range(res.makespan):
        active = sum(
            1 for j in res.jobs
            if (j.assigned_start is not None and j.assigned_end is not None
                and j.assigned_start <= day < j.assigned_end)
        )
        assert active <= problem.capacity


def test_release_dates() -> None:
    """Test that jobs are never scheduled before their release date."""
    jobs = [
        Job(job_id="J1", tail=201, duration=2, release_date=5, due_date=10),
        Job(job_id="J2", tail=202, duration=3, release_date=8, due_date=15),
    ]
    problem = Problem(jobs=jobs, capacity=1, horizon=20, name="release_test")

    res = solve(problem)
    assert res.status == "OPTIMAL"

    for j in res.jobs:
        assert j.assigned_start is not None
        assert j.assigned_start >= j.release_date

    eval_res = evaluate(problem, res)
    assert eval_res["is_valid"] is True


def test_determinism() -> None:
    """Test that identical problems yield byte-identical schedules across repeated solves."""
    rng = random.Random(2026)
    jobs = [
        Job(
            job_id=f"J{i}",
            tail=300 + i,
            duration=rng.randint(1, 4),
            release_date=rng.randint(0, 5),
            due_date=rng.randint(4, 12),
            weight=round(rng.uniform(1.0, 5.0), 1),
        )
        for i in range(8)
    ]
    problem = Problem(jobs=jobs, capacity=2, horizon=25, name="determinism_test")

    res1 = solve(problem, deterministic=True)
    res2 = solve(problem, deterministic=True)

    assert res1.status == res2.status
    assert res1.makespan == res2.makespan
    assert res1.total_weighted_tardiness == pytest.approx(res2.total_weighted_tardiness)

    for j1, j2 in zip(res1.jobs, res2.jobs):
        assert j1.assigned_start == j2.assigned_start
        assert j1.assigned_bay == j2.assigned_bay


def test_empty_problem() -> None:
    """Test handling of empty job list."""
    problem = Problem(jobs=[], capacity=2, horizon=10)
    res = solve(problem)
    assert res.status == "OPTIMAL"
    assert res.makespan == 0
    assert len(res.jobs) == 0

    edf = edf_baseline(problem)
    assert edf.status == "OPTIMAL"
    assert edf.makespan == 0


def test_fallback_behavior() -> None:
    """Test that solver gracefully falls back to EDF if given extreme timeout."""
    jobs = [
        Job(job_id=f"J{i}", tail=400 + i, duration=2, release_date=0, due_date=4)
        for i in range(4)
    ]
    problem = Problem(jobs=jobs, capacity=1, horizon=15)

    # With extreme time limit, should produce valid feasible schedule or fallback
    res = solve(problem, time_limit_s=0.0001, fallback_to_baseline=True)
    assert res.status in ("OPTIMAL", "FEASIBLE", "FALLBACK")
    eval_res = evaluate(problem, res)
    assert eval_res["is_valid"] is True

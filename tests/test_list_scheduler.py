"""Tests for precedence-aware insertion list scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from solvers.list_scheduler import (
    LIST_RULES,
    earliest_insertion_start,
    list_schedule,
    schedule_best_rule,
)
from solvers.milp import FJSPInstance, validate_milp_schedule


def test_list_rules_are_lrpt_and_cp():
    assert LIST_RULES == ("LRPT", "CP")


def test_insertion_fits_idle_gap():
    intervals = [(0.0, 2.0), (5.0, 8.0)]
    assert earliest_insertion_start(intervals, ready=2.0, proc=2.0) == pytest.approx(2.0)
    assert earliest_insertion_start(intervals, ready=3.0, proc=3.0) == pytest.approx(8.0)
    assert earliest_insertion_start([], ready=1.5, proc=4.0) == pytest.approx(1.5)
    assert earliest_insertion_start([(0.0, 4.0)], ready=0.0, proc=1.0) == pytest.approx(4.0)


def test_list_schedule_schedules_each_op_once():
    instance = FJSPInstance(
        n_operations=3,
        n_machines=1,
        proc_times=np.array([[2.0], [3.0], [4.0]], dtype=np.float64),
        eligibility=np.array([[True], [True], [True]]),
        precedences=((0, 2),),
    )
    result = list_schedule(instance, ((0, 0), (1, 0), (2, 0)), rule="LRPT")
    assert result.status == "Feasible"
    assert result.assignment is not None
    assert {op for op, _ in result.assignment} == {0, 1, 2}
    assert validate_milp_schedule(instance, result) == []
    assert result.starts is not None
    assert result.starts[2] >= result.starts[0] + 2.0 - 1e-9
    assert result.makespan == pytest.approx(max(s + p for s, p in zip(result.starts, [2.0, 3.0, 4.0])))


def test_cp_rule_inserts_into_machine_gap():
    """CP schedules the long chain first; the independent op fills the gap at t=0."""
    instance = FJSPInstance(
        n_operations=3,
        n_machines=2,
        proc_times=np.array(
            [
                [2.0, 99.0],
                [2.0, 99.0],
                [99.0, 5.0],
            ],
            dtype=np.float64,
        ),
        eligibility=np.array(
            [
                [True, False],
                [True, False],
                [False, True],
            ]
        ),
        precedences=((2, 0),),
    )
    assignment = ((0, 0), (1, 0), (2, 1))
    result = list_schedule(instance, assignment, rule="CP")
    assert result.status == "Feasible"
    assert validate_milp_schedule(instance, result) == []
    assert result.starts is not None
    # op2 [0,5] on m1, op0 [5,7] on m0, op1 inserts [0,2] on m0
    assert result.starts[2] == pytest.approx(0.0)
    assert result.starts[0] == pytest.approx(5.0)
    assert result.starts[1] == pytest.approx(0.0)
    assert result.makespan == pytest.approx(7.0)


def test_unknown_rule_raises():
    instance = FJSPInstance(
        n_operations=1,
        n_machines=1,
        proc_times=np.array([[1.0]]),
        eligibility=np.array([[True]]),
        precedences=(),
    )
    with pytest.raises(ValueError, match="unknown list rule"):
        list_schedule(instance, ((0, 0),), rule="EFT")


def test_ineligible_assignment_is_incomplete():
    instance = FJSPInstance(
        n_operations=1,
        n_machines=2,
        proc_times=np.array([[1.0, 9.0]]),
        eligibility=np.array([[True, False]]),
        precedences=(),
    )
    result = list_schedule(instance, ((0, 1),), rule="LRPT")
    assert result.status == "Incomplete"
    assert result.makespan == float("inf")


def test_schedule_best_rule_keeps_lowest_cmax():
    instance = FJSPInstance(
        n_operations=2,
        n_machines=1,
        proc_times=np.array([[2.0], [3.0]], dtype=np.float64),
        eligibility=np.array([[True], [True]]),
        precedences=(),
    )
    result, rule = schedule_best_rule(instance, ((0, 0), (1, 0)))
    assert rule in LIST_RULES
    assert result.makespan == pytest.approx(5.0)
    assert validate_milp_schedule(instance, result) == []


def test_multiple_predecessors():
    instance = FJSPInstance(
        n_operations=3,
        n_machines=2,
        proc_times=np.array([[2.0, 99.0], [3.0, 99.0], [99.0, 4.0]], dtype=np.float64),
        eligibility=np.array([[True, False], [True, False], [False, True]]),
        precedences=((0, 2), (1, 2)),
    )
    result = list_schedule(instance, ((0, 0), (1, 0), (2, 1)), rule="LRPT")
    assert validate_milp_schedule(instance, result) == []
    assert result.starts is not None
    assert result.starts[2] >= 5.0 - 1e-9
    assert result.makespan == pytest.approx(9.0)


def test_every_list_rule_builds_a_feasible_schedule():
    instance = FJSPInstance(
        n_operations=3,
        n_machines=2,
        proc_times=np.array([[2.0, 4.0], [3.0, 1.0], [5.0, 5.0]], dtype=np.float64),
        eligibility=np.array([[True, True], [True, True], [True, True]]),
        precedences=((0, 2),),
    )
    assignment = ((0, 0), (1, 1), (2, 0))
    for rule in LIST_RULES:
        result = list_schedule(instance, assignment, rule=rule)
        assert result.status == "Feasible", rule
        assert validate_milp_schedule(instance, result) == [], rule

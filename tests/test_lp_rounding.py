"""Tests for the FJSP LP relaxation + rounding baseline."""

from __future__ import annotations

import numpy as np
import pytest

from baseline_lp import parse_args
from config import get_default_eval_config
from solvers.lp_rounding import (
    LP_TOL,
    generate_rounded_assignments,
    largest_fraction_assignment,
    sample_rounded_assignment,
    solve_lp_relaxation,
    solve_lp_rounding,
)
from solvers.milp import (
    FJSPInstance,
    extract_fjsp_instance,
    solve_makespan,
    validate_milp_schedule,
)
from training.eval_cli import apply_shared_eval_args, build_eval_train_config
from training.evaluate import evaluate_lp_rounding_fjsp
from training.make_env import make_vec_env


def _chain_instance() -> FJSPInstance:
    return FJSPInstance(
        n_operations=2,
        n_machines=2,
        proc_times=np.array([[2.0, 99.0], [99.0, 3.0]], dtype=np.float64),
        eligibility=np.array([[True, False], [False, True]]),
        precedences=((0, 1),),
    )


def _serial_machine_instance() -> FJSPInstance:
    return FJSPInstance(
        n_operations=2,
        n_machines=1,
        proc_times=np.array([[2.0], [3.0]], dtype=np.float64),
        eligibility=np.array([[True], [True]]),
        precedences=(),
    )


def _jsp_2x2_instance() -> FJSPInstance:
    """Same-route 2x2 job shop: load/CP = 4, OPT = 6."""
    return FJSPInstance(
        n_operations=4,
        n_machines=2,
        proc_times=np.array(
            [
                [2.0, 99.0],
                [99.0, 2.0],
                [2.0, 99.0],
                [99.0, 2.0],
            ],
            dtype=np.float64,
        ),
        eligibility=np.array(
            [
                [True, False],
                [False, True],
                [True, False],
                [False, True],
            ]
        ),
        precedences=((0, 1), (2, 3)),
    )


def test_lp_assignment_and_eligibility():
    instance = FJSPInstance(
        n_operations=2,
        n_machines=2,
        proc_times=np.array([[2.0, 4.0], [5.0, 3.0]], dtype=np.float64),
        eligibility=np.array([[True, True], [False, True]]),
        precedences=(),
    )
    lp = solve_lp_relaxation(instance)
    assert lp.status == "Optimal"
    for i in range(instance.n_operations):
        row = lp.fractional_assignment[i]
        for m, val in row.items():
            assert bool(instance.eligibility[i, m])
            assert -LP_TOL <= val <= 1.0 + LP_TOL
        assert abs(sum(row.values()) - 1.0) <= 1e-5
    assert lp.assignment_sum_violation <= 1e-5
    assert lp.max_constraint_violation <= 1e-5


def test_lp_precedence_and_objective_chain():
    instance = _chain_instance()
    lp = solve_lp_relaxation(instance)
    assert lp.status == "Optimal"
    assert lp.starts is not None
    p0 = 2.0
    assert lp.starts[1] + 1e-6 >= lp.starts[0] + p0
    assert lp.lp_lower_bound == pytest.approx(5.0)
    assert lp.lp_lower_bound == pytest.approx(
        max(lp.starts[i] + [2.0, 3.0][i] for i in range(2))
    )


def test_lp_lower_bound_matches_opt_on_serial_and_chain():
    for instance, opt in ((_serial_machine_instance(), 5.0), (_chain_instance(), 5.0)):
        lp = solve_lp_relaxation(instance)
        milp = solve_makespan(instance)
        assert milp.status == "Optimal"
        assert lp.status == "Optimal"
        assert lp.lp_lower_bound <= milp.makespan + 1e-5
        assert lp.lp_lower_bound == pytest.approx(opt)
        assert milp.makespan == pytest.approx(opt)


def test_lp_lower_bound_strictly_below_opt_on_2x2_jsp():
    instance = _jsp_2x2_instance()
    lp = solve_lp_relaxation(instance)
    milp = solve_makespan(instance)
    assert lp.status == "Optimal"
    assert milp.status == "Optimal"
    assert milp.makespan == pytest.approx(6.0)
    assert lp.lp_lower_bound == pytest.approx(4.0)
    assert lp.lp_lower_bound <= milp.makespan - 1e-6


def test_lp_does_not_silently_round_fractional_assignment():
    """Three equal ops on two machines: load bound 3 forces a split assignment."""
    instance = FJSPInstance(
        n_operations=3,
        n_machines=2,
        proc_times=np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float64),
        eligibility=np.array([[True, True], [True, True], [True, True]]),
        precedences=(),
    )
    lp = solve_lp_relaxation(instance)
    assert lp.status == "Optimal"
    assert lp.lp_lower_bound == pytest.approx(3.0)
    assert lp.n_fractional >= 1
    raw = [v for row in lp.fractional_assignment.values() for v in row.values()]
    assert any(min(v, 1.0 - v) > 1e-6 for v in raw)


def test_largest_fraction_rounding_is_deterministic_and_eligible():
    frac = {
        0: {0: 0.72, 1: 0.28},
        1: {1: 0.4, 2: 0.4, 0: 0.2},
    }
    a = largest_fraction_assignment(frac, 2)
    b = largest_fraction_assignment(frac, 2)
    assert a == b
    assert a == ((0, 0), (1, 1))
    for op, machine in a:
        assert machine in frac[op]


def test_randomized_rounding_respects_seed_and_eligibility():
    frac = {
        0: {0: 0.55, 1: 0.45},
        1: {0: 0.1, 1: 0.9},
    }
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    rng_c = np.random.default_rng(999)
    a = sample_rounded_assignment(frac, 2, rng_a)
    b = sample_rounded_assignment(frac, 2, rng_b)
    c = sample_rounded_assignment(frac, 2, rng_c)
    assert a == b
    for op, machine in a:
        assert machine in frac[op]
    for op, machine in c:
        assert machine in frac[op]
    unique = generate_rounded_assignments(frac, 2, n_trials=20, seed=7)
    assert unique[0] == largest_fraction_assignment(frac, 2)
    assert 1 <= len(unique) <= 20


def test_rounding_never_assigns_ineligible_from_lp():
    instance = FJSPInstance(
        n_operations=2,
        n_machines=2,
        proc_times=np.array([[2.0, 4.0], [9.0, 3.0]], dtype=np.float64),
        eligibility=np.array([[True, False], [False, True]]),
        precedences=(),
    )
    lp = solve_lp_relaxation(instance)
    assignment = largest_fraction_assignment(lp.fractional_assignment, 2)
    assert assignment == ((0, 0), (1, 1))


def test_end_to_end_rounding_feasible_and_above_lb():
    instance = _serial_machine_instance()
    result = solve_lp_rounding(instance, seed=42, rounding_trials=5)
    assert result.status == "Feasible"
    assert result.lp_status == "Optimal"
    assert validate_milp_schedule(instance, result.as_schedule()) == []
    assert result.makespan == pytest.approx(5.0)
    assert result.lp_lower_bound <= result.makespan + 1e-6
    by_rule = dict(result.rule_makespans)
    assert set(by_rule) == {"LRPT", "CP"}
    assert min(by_rule.values()) == pytest.approx(result.makespan)


def test_end_to_end_2x2_jsp_sandwich():
    instance = _jsp_2x2_instance()
    result = solve_lp_rounding(instance, seed=0, rounding_trials=8)
    milp = solve_makespan(instance)
    assert result.status == "Feasible"
    assert validate_milp_schedule(instance, result.as_schedule()) == []
    assert result.lp_lower_bound <= milp.makespan + 1e-5
    assert milp.makespan <= result.makespan + 1e-5


def _tiny_eval_cfg(seed: int = 0, n_episodes: int = 1):
    cfg = get_default_eval_config()
    cfg.seed = seed
    cfg.n_episodes = n_episodes
    cfg.env.n_machines = 2
    cfg.env.n_jobs = 2
    cfg.env.avg_operations_per_job = 2
    cfg.env.cross_job_dep_prob = 0.0
    cfg.env.shared_dep_prob = 0.0
    cfg.validate()
    return cfg


def test_lp_bound_vs_milp_on_generated_instances():
    cfg = _tiny_eval_cfg(seed=3, n_episodes=3)
    train_cfg = build_eval_train_config(cfg)
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        for ep in range(cfg.n_episodes):
            env.seed(cfg.seed + ep)
            env.reset()
            instance = extract_fjsp_instance(env.envs[0].unwrapped)
            lp = solve_lp_relaxation(instance)
            milp = solve_makespan(instance, time_limit=60.0)
            rounded = solve_lp_rounding(instance, seed=cfg.seed + ep, rounding_trials=8)
            assert lp.status == "Optimal"
            assert milp.status == "Optimal"
            assert rounded.status == "Feasible"
            assert lp.lp_lower_bound <= milp.makespan + 1e-4
            assert milp.makespan <= rounded.makespan + 1e-4
            assert validate_milp_schedule(instance, rounded.as_schedule()) == []
            assert lp.max_constraint_violation <= 1e-5
    finally:
        env.close()


def test_lp_eval_seed_suite_deterministic():
    cfg = _tiny_eval_cfg(seed=7, n_episodes=2)
    train_cfg = build_eval_train_config(cfg)

    def _run():
        env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
        try:
            return evaluate_lp_rounding_fjsp(
                env,
                n_episodes=cfg.n_episodes,
                seed=cfg.seed,
                rounding_trials=5,
            )
        finally:
            env.close()

    a_result, a_rows = _run()
    b_result, b_rows = _run()
    assert a_result.success_rate == pytest.approx(1.0)
    assert b_result.mean_makespan == pytest.approx(a_result.mean_makespan)
    assert [r.makespan for r in a_rows] == [r.makespan for r in b_rows]
    assert [r.lp_lower_bound for r in a_rows] == [r.lp_lower_bound for r in b_rows]


def test_cli_flags():
    with pytest.raises(SystemExit):
        parse_args(["--rounding-trials", "nope"])
    args = parse_args(
        ["--rounding-trials", "8", "--verbose", "--compare-milp", "--n-episodes", "2"]
    )
    assert args.rounding_trials == 8
    assert args.verbose is True
    assert args.compare_milp is True
    assert args.n_episodes == 2
    assert parse_args(["--full-scale"]).full_scale is True


def test_eval_cli_full_scale_includes_lp_baseline():
    args = parse_args(["--full-scale", "--rounding-trials", "3"])
    cfg = apply_shared_eval_args(get_default_eval_config(), args)
    assert cfg.env.n_machines == 25
    assert cfg.env.n_jobs == 15
    assert args.rounding_trials == 3

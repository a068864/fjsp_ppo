"""Tests for the exact FJSP MILP makespan solver."""

from __future__ import annotations

import numpy as np
import pytest

from config import get_default_eval_config
from solvers.milp import (
    FJSPInstance,
    extract_fjsp_instance,
    milp_episode_metrics,
    solve_makespan,
)
from training.eval_cli import build_eval_train_config
from training.evaluate import evaluate_milp_fjsp
from training.make_env import make_vec_env


def test_handcrafted_precedence_optimum():
    """Op0 (p=2 on m0) precedes Op1 (p=3 on m1) → Cmax = 5."""
    instance = FJSPInstance(
        n_operations=2,
        n_machines=2,
        proc_times=np.array([[2.0, 99.0], [99.0, 3.0]], dtype=np.float64),
        eligibility=np.array([[True, False], [False, True]]),
        precedences=((0, 1),),
    )
    result = solve_makespan(instance)
    assert result.status == "Optimal"
    assert result.makespan == pytest.approx(5.0)
    assert result.assignment == ((0, 0), (1, 1))


def test_handcrafted_single_machine_serial():
    """Both ops only on m0 with p=2,3 → Cmax = 5."""
    instance = FJSPInstance(
        n_operations=2,
        n_machines=1,
        proc_times=np.array([[2.0], [3.0]], dtype=np.float64),
        eligibility=np.array([[True], [True]]),
        precedences=(),
    )
    result = solve_makespan(instance)
    assert result.status == "Optimal"
    assert result.makespan == pytest.approx(5.0)


def test_non_optimal_maps_to_failure_metrics():
    instance = FJSPInstance(
        n_operations=1,
        n_machines=1,
        proc_times=np.array([[1.0]]),
        eligibility=np.array([[True]]),
        precedences=(),
        time_penalty=-0.1,
        time_step=1.0,
    )
    from solvers.milp import MilpResult

    ep = milp_episode_metrics(
        instance,
        MilpResult(status="Not Solved", makespan=float("inf"), solve_time_s=0.0),
    )
    assert ep["success"] is False
    assert ep["makespan"] == float("inf")
    assert ep["truncated"] is True


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


def test_env_extract_and_solve_smoke():
    cfg = _tiny_eval_cfg(seed=0, n_episodes=1)
    train_cfg = build_eval_train_config(cfg)
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(cfg.seed)
        env.reset()
        instance = extract_fjsp_instance(env.envs[0].unwrapped)
        result = solve_makespan(instance, time_limit=60.0)
        assert result.status == "Optimal"
        assert np.isfinite(result.makespan)
        assert result.makespan > 0.0
    finally:
        env.close()


def test_milp_eval_seed_suite_deterministic():
    cfg = _tiny_eval_cfg(seed=7, n_episodes=2)
    train_cfg = build_eval_train_config(cfg)

    def _run():
        env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
        try:
            return evaluate_milp_fjsp(
                env, n_episodes=cfg.n_episodes, seed=cfg.seed, time_limit=60.0
            )
        finally:
            env.close()

    a = _run()
    b = _run()
    assert a.success_rate == pytest.approx(1.0)
    assert b.mean_makespan == pytest.approx(a.mean_makespan)
    assert b.mean_reward == pytest.approx(a.mean_reward)

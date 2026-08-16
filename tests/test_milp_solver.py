"""Tests for the exact FJSP MILP makespan solver."""

from __future__ import annotations

from itertools import product
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from config import get_default_eval_config
from solvers.milp import (
    FJSPInstance,
    decode_assignment_schedule,
    extract_fjsp_instance,
    milp_episode_metrics,
    solve_makespan,
    validate_milp_schedule,
)
from training.eval_cli import build_eval_train_config
from training.evaluate import evaluate_milp_fjsp, evaluate_policy_fjsp
from training.make_env import make_vec_env


def _topo_orders(n: int, precedences: Sequence[Tuple[int, int]]) -> List[Tuple[int, ...]]:
    succ = {i: [] for i in range(n)}
    indeg = [0] * n
    seen = set()
    for a, b in precedences:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        succ[a].append(b)
        indeg[b] += 1
    orders: List[Tuple[int, ...]] = []

    def rec(prefix: List[int], remaining_indeg: List[int]) -> None:
        if len(prefix) == n:
            orders.append(tuple(prefix))
            return
        ready = [i for i in range(n) if remaining_indeg[i] == 0 and i not in prefix]
        for i in ready:
            nxt = remaining_indeg[:]
            nxt[i] = -1
            for j in succ[i]:
                nxt[j] -= 1
            rec(prefix + [i], nxt)

    rec([], indeg)
    return orders


def brute_classic_makespan(instance: FJSPInstance) -> float:
    """Earliest-start Cmax over all precedence-feasible orders and assignments."""
    n = instance.n_operations
    elig = [np.flatnonzero(instance.eligibility[i]).tolist() for i in range(n)]
    best = float("inf")
    for order in _topo_orders(n, instance.precedences):
        for machines in product(*elig):
            start = [0.0] * n
            free = [0.0] * instance.n_machines
            for i in order:
                m = int(machines[i])
                s = free[m]
                for pred, succ in instance.precedences:
                    if succ != i:
                        continue
                    pm = int(machines[pred])
                    s = max(s, start[pred] + float(instance.proc_times[pred, pm]))
                start[i] = s
                p = float(instance.proc_times[i, m])
                free[m] = s + p
            cmax = max(
                start[i] + float(instance.proc_times[i, int(machines[i])])
                for i in range(n)
            )
            if cmax < best:
                best = cmax
    return best


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
    assert result.starts is not None
    assert result.starts[0] == pytest.approx(0.0)
    assert result.starts[1] == pytest.approx(2.0)
    assert validate_milp_schedule(instance, result) == []
    assert brute_classic_makespan(instance) == pytest.approx(result.makespan)


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
    assert validate_milp_schedule(instance, result) == []
    assert brute_classic_makespan(instance) == pytest.approx(result.makespan)


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
        assert validate_milp_schedule(instance, result) == []
        assert brute_classic_makespan(instance) == pytest.approx(result.makespan, abs=1e-4)
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


def test_two_independent_ops_milp_is_classic_not_env_makespan():
    """Classic FJSP starts both machines at t=0; the env assigns one op per tick."""
    instance = FJSPInstance(
        n_operations=2,
        n_machines=2,
        proc_times=np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64),
        eligibility=np.array([[True, True], [True, True]]),
        precedences=(),
        time_step=1.0,
    )
    result = solve_makespan(instance)
    assert validate_milp_schedule(instance, result) == []
    assert result.makespan == pytest.approx(2.0)
    assert brute_classic_makespan(instance) == pytest.approx(2.0)

    decoded = decode_assignment_schedule(instance, ((0, 0), (1, 1)))
    assert decoded.status == "Feasible"
    assert decoded.makespan == pytest.approx(2.0)
    assert decoded.starts == pytest.approx((0.0, 0.0))
    assert validate_milp_schedule(instance, decoded) == []


def test_decode_respects_precedence_and_machine_order():
    instance = FJSPInstance(
        n_operations=3,
        n_machines=1,
        proc_times=np.array([[2.0], [3.0], [4.0]], dtype=np.float64),
        eligibility=np.array([[True], [True], [True]]),
        precedences=((0, 2),),
    )
    decoded = decode_assignment_schedule(instance, ((1, 0), (0, 0), (2, 0)))
    assert decoded.status == "Feasible"
    assert decoded.makespan == pytest.approx(9.0)
    assert decoded.starts == pytest.approx((3.0, 0.0, 5.0))
    assert validate_milp_schedule(instance, decoded) == []
    incomplete = decode_assignment_schedule(instance, ((0, 0),))
    assert incomplete.status == "Incomplete"
    assert incomplete.makespan == float("inf")


class _FirstValidPolicy:
    def predict(self, observations, deterministic=True):
        mask = np.asarray(observations["action_mask"], dtype=np.float32)
        if mask.ndim == 1:
            mask = mask.reshape(1, -1)
        actions = np.array(
            [int(np.flatnonzero(row > 0.5)[0]) for row in mask],
            dtype=np.int64,
        )
        return actions, None


def test_policy_eval_scores_classic_instance_cmax():
    """Eval reports decoded FJSP Cmax, which cannot beat the MILP optimum."""
    cfg = _tiny_eval_cfg(seed=11, n_episodes=3)
    train_cfg = build_eval_train_config(cfg)
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(cfg.seed)
        milp = evaluate_milp_fjsp(env, n_episodes=cfg.n_episodes, seed=cfg.seed)
        env.seed(cfg.seed)
        policy = evaluate_policy_fjsp(
            _FirstValidPolicy(), env, n_episodes=cfg.n_episodes, deterministic=True
        )
    finally:
        env.close()
    assert milp.success_rate == pytest.approx(1.0)
    assert policy.success_rate == pytest.approx(1.0)
    assert np.isfinite(policy.mean_makespan)
    assert policy.mean_makespan + 1e-4 >= milp.mean_makespan


def test_policy_eval_uses_env_logged_cmax_without_decode(monkeypatch):
    """Eval reads classic Cmax from episode info; it must not replay assignments."""

    def _no_eval_replay(*_args, **_kwargs):
        raise AssertionError("evaluate must not call decode_assignment_schedule")

    monkeypatch.setattr("solvers.milp.decode_assignment_schedule", _no_eval_replay)

    import training.evaluate as evaluate_mod

    assert not hasattr(evaluate_mod, "decode_assignment_schedule")

    cfg = _tiny_eval_cfg(seed=11, n_episodes=2)
    train_cfg = build_eval_train_config(cfg)
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(cfg.seed)
        policy = evaluate_policy_fjsp(
            _FirstValidPolicy(), env, n_episodes=cfg.n_episodes, deterministic=True
        )
    finally:
        env.close()
    assert policy.success_rate == pytest.approx(1.0)
    assert np.isfinite(policy.mean_makespan)
    assert policy.n_episodes == 2

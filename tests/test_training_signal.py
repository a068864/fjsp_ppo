"""Training signal: makespan-aligned reward, failure cost, ECT action prior."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from config import get_debug_train_config
from envs.fjsp_env import (
    OP_DURATION,
    OP_FINISHED,
    OP_REMAINING,
    OP_SEQ_DEPS,
    OP_CP_REMAINING,
    FJSPEnv,
)
from models.actor_critic import GraphActorCritic


def test_debug_ppo_uses_full_horizon_gae_and_low_entropy():
    cfg = get_debug_train_config()
    assert cfg.ppo.gae_lambda == pytest.approx(1.0)
    assert cfg.ppo.ent_coef == pytest.approx(0.01)
    assert cfg.ppo.gamma == pytest.approx(1.0)
    assert cfg.ppo.n_epochs == 6
    assert cfg.ppo.vf_coef == pytest.approx(0.1)
    assert cfg.ppo.max_grad_norm == pytest.approx(5.0)
    assert cfg.ppo.target_kl == pytest.approx(0.02)
    assert cfg.lr_end_fraction == pytest.approx(0.8)


def test_nonterminal_step_reward_tracks_classic_cmax():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    before = float(env._classic_cmax)
    action = int(np.flatnonzero(env.get_action_mask())[0])
    _obs, reward, terminated, truncated, info = env.step(action)
    after = float(env._classic_cmax)
    expected = float(env.time_penalty) * (after - before)
    if terminated or truncated:
        if not info.get("success"):
            expected += env.failure_penalty()
        else:
            env.close()
            pytest.skip("first step finished the instance")
    assert float(reward) == pytest.approx(expected, abs=1e-5)
    env.close()


def test_failure_penalty_worse_than_serial_success_bound():
    env = FJSPEnv(
        n_machines=2,
        n_jobs=2,
        avg_operations_per_job=2,
        seed=0,
        device="cpu",
    )
    worst_success = (
        float(env.time_penalty)
        * float(env.n_operations)
        * float(env.max_operation_duration)
        * 1.5
    )
    assert env.failure_penalty() < worst_success
    env.close()


def test_gridlock_step_applies_failure_penalty():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {(1, 2): "sequential", (2, 0): "sequential"}
    env.state["operation", "precede", "operation"].edge_index = torch.tensor(
        [[1, 2], [2, 0]], dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, OP_SEQ_DEPS] = 0
    env.state["operation"].x[2, OP_SEQ_DEPS] = 1
    env.state["operation"].x[0, OP_SEQ_DEPS] = 1
    env.state["operation"].x[:, OP_DURATION] = 2.0
    env.state["operation"].x[:, OP_REMAINING] = 2.0
    env.state["operation"].x[:, OP_FINISHED] = 0
    env.state["machine"].x[:] = 0
    env.schedule_operation(0, 0)
    env.schedule_operation(0, 1)
    action = 1 * env.n_operations + 2
    _obs, reward, terminated, _truncated, info = env.step(action)
    assert terminated
    assert info.get("is_gridlock")
    assert float(reward) <= env.failure_penalty() + 1e-5
    env.close()


def test_dispatch_score_prefers_lower_ect():
    graph = HeteroData()
    graph["operation"].x = torch.zeros((1, 10), dtype=torch.float32)
    graph["operation"].x[0, OP_DURATION] = 4.0
    graph["machine"].x = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=torch.float32
    )
    even = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    scores = GraphActorCritic.dispatch_score_matrix(graph, even, torch.device("cpu"))
    assert float(scores[0, 0]) > float(scores[0, 1])

    graph["machine"].x[:] = 0
    faster_first = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    scores_fast = GraphActorCritic.dispatch_score_matrix(
        graph, faster_first, torch.device("cpu")
    )
    assert float(scores_fast[0, 0]) > float(scores_fast[0, 1])


def test_reset_estimated_completion_uses_critical_path():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    env.reset(seed=0)
    op_x = env.state["operation"].x
    unfinished = op_x[:, OP_FINISHED] < 0.5
    cp = float(op_x[unfinished, OP_CP_REMAINING].max().item())
    work_lb = float(op_x[unfinished, OP_REMAINING].sum().item()) / float(env.n_machines)
    assert env.estimated_completion() == pytest.approx(max(cp, work_lb), abs=1e-4)
    env.close()


def test_successful_episode_logs_classic_makespan_not_clock(monkeypatch):
    """Logged makespan is earliest-start Cmax; PPO reward is the Cmax delta."""
    from solvers.milp import decode_assignment_schedule as decode_oracle
    from solvers.milp import extract_fjsp_instance

    def _no_end_replay(*_args, **_kwargs):
        raise AssertionError("do not replay the assignment list to log Cmax")

    monkeypatch.setattr("solvers.milp.decode_assignment_schedule", _no_end_replay)

    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    instance = extract_fjsp_instance(env)
    before = float(env._classic_cmax)
    action = int(np.flatnonzero(obs["action_mask"])[0])
    obs, first_reward, terminated, truncated, _info = env.step(action)
    after = float(env._classic_cmax)
    if not (terminated or truncated):
        assert float(first_reward) == pytest.approx(
            float(env.time_penalty) * (after - before), abs=1e-5
        )
    done = bool(terminated or truncated)
    info = _info
    while not done:
        mask = np.asarray(obs["action_mask"])
        valid = np.flatnonzero(mask > 0.5)
        if valid.size == 0:
            env.close()
            pytest.skip("empty mask before success")
        obs, _r, terminated, truncated, info = env.step(int(valid[0]))
        done = bool(terminated or truncated)
    if not info.get("success"):
        env.close()
        pytest.skip("episode did not succeed")
    decoded = decode_oracle(instance, env._assignment_order)
    assert info["makespan"] == pytest.approx(decoded.makespan)
    assert decoded.makespan <= env.current_time + 1e-6
    env.close()


def test_logged_makespan_is_running_classic_cmax(monkeypatch):
    """Two independent ops: Cmax is 2, maintained during assign, not a final replay."""
    from solvers.milp import decode_assignment_schedule as decode_oracle
    from solvers.milp import extract_fjsp_instance

    def _no_end_replay(*_args, **_kwargs):
        raise AssertionError("do not replay the assignment list to log Cmax")

    monkeypatch.setattr("solvers.milp.decode_assignment_schedule", _no_end_replay)

    env = FJSPEnv(
        n_machines=2,
        n_jobs=1,
        avg_operations_per_job=2,
        seed=0,
        device="cpu",
    )
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.state["operation", "precede", "operation"].edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, OP_DURATION] = 2.0
    env.state["operation"].x[:, OP_REMAINING] = 2.0
    env._mark_lookahead_stale()
    env._cached_action_mask = None
    instance = extract_fjsp_instance(env)

    info = {}
    done = False
    ep_return = 0.0
    for op in range(2):
        action = op * env.n_operations + op
        _obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        done = bool(terminated or truncated)
    assert done
    assert info.get("success")
    decoded = decode_oracle(instance, env._assignment_order)
    assert info["makespan"] == pytest.approx(2.0)
    assert info["makespan"] == pytest.approx(decoded.makespan)
    assert decoded.makespan <= env.current_time + 1e-6
    assert ep_return == pytest.approx(float(env.time_penalty) * 2.0)
    env.close()

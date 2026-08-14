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
    FJSPEnv,
)
from models.actor_critic import GraphActorCritic


def test_debug_ppo_uses_full_horizon_gae_and_low_entropy():
    cfg = get_debug_train_config()
    assert cfg.ppo.gae_lambda == pytest.approx(1.0)
    assert cfg.ppo.ent_coef == pytest.approx(0.01)
    assert cfg.ppo.gamma == pytest.approx(1.0)
    assert cfg.ppo.n_epochs == 4
    assert cfg.ppo.vf_coef == pytest.approx(0.25)
    assert cfg.ppo.max_grad_norm == pytest.approx(2.0)
    assert cfg.ppo.target_kl == pytest.approx(0.02)
    assert cfg.lr_end_fraction == pytest.approx(0.5)


def test_nonterminal_step_reward_tracks_completion_bound():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    before = env.estimated_completion()
    action = int(np.flatnonzero(env.get_action_mask())[0])
    _obs, reward, terminated, truncated, info = env.step(action)
    after = env.estimated_completion()
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
        time_step=4.0,
        seed=0,
        device="cpu",
    )
    worst_success = (
        float(env.time_penalty)
        * float(env.n_operations)
        * float(env.max_operation_duration)
        * float(env.time_step)
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

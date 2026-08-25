"""Contracts for audit remediation (buffer clone, metrics, edges)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch_geometric.data import HeteroData

from config import get_debug_train_config
from envs.fjsp_env import FJSPEnv, make_sb3_graph_observation_space
from models.actor_critic import GraphActorCritic
from models.graph_encoder import MESSAGE_EDGE_TYPES, GraphEncoder
from models.graph_ppo import GraphPPO
from training.evaluate import _aggregate_eval
from training.graph_buffer import GraphDictRolloutBuffer, slim_graph_for_policy
from training.make_env import make_vec_env


def test_buffer_add_does_not_clone_already_slim_graph():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    buf = GraphDictRolloutBuffer(
        buffer_size=2,
        observation_space=make_sb3_graph_observation_space(),
        action_space=spaces.Discrete(2),
        device="cpu",
        n_envs=1,
        gae_lambda=0.95,
        gamma=0.99,
    )
    buf.reset()
    ptr = obs["graph"]["operation"].x.data_ptr()
    graphs = np.empty((1,), dtype=object)
    graphs[0] = obs["graph"]
    buf.add(
        {
            "dummy": np.asarray(obs["dummy"], dtype=np.float32).reshape(1, -1),
            "action_mask": np.asarray(obs["action_mask"], dtype=np.float32).reshape(1, -1),
            "graph": graphs,
        },
        np.array([[0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        torch.zeros(1),
        torch.zeros(1),
    )
    stored = buf.observations["graph"][0, 0]
    assert stored["operation"].x.data_ptr() == ptr
    env.close()


def test_slim_graph_clone_false_shares_storage():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    slim = slim_graph_for_policy(obs["graph"], clone=False)
    assert slim["operation"].x.data_ptr() == obs["graph"]["operation"].x.data_ptr()
    env.close()


def test_graph_ppo_collect_rollouts_delegates_to_sb3():
    src = inspect.getsource(GraphPPO.collect_rollouts)
    assert "super().collect_rollouts" in src
    assert "graph_obs_as_tensor" in src


def test_encoder_convs_expose_configured_num_heads():
    enc = GraphEncoder(hidden_dim=32, num_layers=1, num_heads=4)
    conv = next(iter(enc.layers[0].conv.convs.values()))
    assert getattr(conv, "heads", None) == 4


def test_env_has_no_write_only_tracking_fields():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    assert not hasattr(env, "machine_last_idle_time")
    assert not hasattr(env, "assignment_history")
    env.close()


def test_aggregate_eval_does_not_count_timeouts_as_failures():
    result = _aggregate_eval(
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.01],
    )
    assert result.n_success == 1
    assert result.n_timeout == 1
    assert result.n_failure == 2


def test_efficiency_matrix_zero_for_missing_compatible_edges():
    g = HeteroData()
    g["operation"].x = torch.zeros((2, 10), dtype=torch.float32)
    g["machine"].x = torch.zeros((2, 3), dtype=torch.float32)
    g["operation", "compatible", "machine"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long
    )
    g["operation", "compatible", "machine"].edge_attr = torch.tensor(
        [[0.7]], dtype=torch.float32
    )
    mat = GraphActorCritic.efficiency_matrix(g, torch.device("cpu"))
    assert float(mat[0, 0]) == pytest.approx(0.7)
    assert float(mat[0, 1]) == pytest.approx(0.0)
    assert float(mat[1, 0]) == pytest.approx(0.0)


def test_dummy_vec_render_does_not_raise():
    cfg = get_debug_train_config()
    cfg.n_envs = 1
    env = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=True)
    env.reset()
    env.render()
    env.close()


def test_schedule_adds_reverse_processing_edge():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.schedule_operation(0, 0)
    reverse = ("operation", "processed_by", "machine")
    assert reverse in MESSAGE_EDGE_TYPES
    assert reverse in env.state.edge_types
    ei = env.state[reverse].edge_index
    assert ei.shape[1] == 1
    assert int(ei[0, 0]) == 0 and int(ei[1, 0]) == 0
    env.close()

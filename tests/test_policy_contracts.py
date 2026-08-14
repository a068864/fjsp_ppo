"""PPO / policy contract checks from the audit remediation plan."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch_geometric.data import HeteroData

from config import ModelConfig
from envs.fjsp_env import FJSPEnv, OP_FEATURE_DIM
from models.actor_critic import GraphActorCritic
from models.sb3_policy import GraphActorCriticPolicy


def _tiny_graph(n_machines: int = 2, n_ops: int = 2, efficiency: float = 1.0) -> HeteroData:
    g = HeteroData()
    g["operation"].x = torch.zeros((n_ops, OP_FEATURE_DIM), dtype=torch.float32)
    g["operation"].x[:, 0] = 4.0
    g["machine"].x = torch.zeros((n_machines, 3), dtype=torch.float32)
    # One compatible edge per (op, machine) with efficiency attr.
    edges = []
    attrs = []
    for op in range(n_ops):
        for m in range(n_machines):
            edges.append([op, m])
            attrs.append([efficiency if (op + m) % 2 == 0 else efficiency * 0.5])
    g["operation", "compatible", "machine"].edge_index = torch.tensor(
        edges, dtype=torch.long
    ).t()
    g["operation", "compatible", "machine"].edge_attr = torch.tensor(
        attrs, dtype=torch.float32
    )
    g["operation", "precede", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
    g["operation", "next", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
    g["machine", "processing", "operation"].edge_index = torch.empty((2, 0), dtype=torch.long)
    return g


def _pack_obs(graph: HeteroData, action_mask: np.ndarray) -> dict:
    graphs = np.empty((1,), dtype=object)
    graphs[0] = graph
    mask = np.asarray(action_mask, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    return {
        "dummy": np.zeros((1, 1), dtype=np.float32),
        "action_mask": mask,
        "graph": graphs,
    }


def test_model_config_rejects_nonzero_dropout():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=0.1)


def test_swapped_efficiencies_change_logits():
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.eval()
    g_hi = _tiny_graph(efficiency=1.0)
    g_lo = _tiny_graph(efficiency=0.1)
    # Swap only compatible edge attrs between copies of the same topology.
    g_swapped = g_hi.clone()
    g_swapped["operation", "compatible", "machine"].edge_attr = (
        g_lo["operation", "compatible", "machine"].edge_attr.clone()
    )
    with torch.no_grad():
        logits_a, _ = ac.forward_single(g_hi)
        logits_b, _ = ac.forward_single(g_swapped)
    assert not torch.allclose(logits_a, logits_b), (
        "Efficiency edge attributes must affect pair scores"
    )


def test_encode_does_not_move_caller_owned_graph_to_cuda_device():
    if not torch.cuda.is_available():
        # Still assert CPU path does not mutate storage device unexpectedly.
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    ).to(device)
    ac.eval()
    g = _tiny_graph()
    assert g["operation"].x.device.type == "cpu"
    with torch.no_grad():
        ac.forward_single(g)
    assert g["operation"].x.device.type == "cpu"
    assert g["machine"].x.device.type == "cpu"


def test_get_value_skips_actor_path(monkeypatch):
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.eval()
    called = {"actor": False}

    original = ac.actor.forward

    def _boom(*args, **kwargs):
        called["actor"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(ac.actor, "forward", _boom)
    g = _tiny_graph()
    with torch.no_grad():
        value = ac.get_value(g)
    assert value.ndim == 0 or value.numel() == 1
    assert called["actor"] is False


def test_policy_rejects_all_zero_masks_for_action_distribution():
    cfg = ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    n_actions = 4
    policy = GraphActorCriticPolicy(
        observation_space=spaces.Dict(
            {
                "dummy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(n_actions,), dtype=np.float32),
                "graph": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        ),
        action_space=spaces.Discrete(n_actions),
        lr_schedule=lambda _: 1e-3,
        model_config=cfg,
    )
    policy.eval()
    g = _tiny_graph()
    obs = _pack_obs(g, np.zeros((n_actions,), dtype=np.float32))
    with pytest.raises(ValueError, match="mask"):
        policy.forward(obs, deterministic=True)


def test_policy_value_only_allows_terminal_bootstrap_with_empty_mask():
    cfg = ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    n_actions = 4
    policy = GraphActorCriticPolicy(
        observation_space=spaces.Dict(
            {
                "dummy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(n_actions,), dtype=np.float32),
                "graph": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        ),
        action_space=spaces.Discrete(n_actions),
        lr_schedule=lambda _: 1e-3,
        model_config=cfg,
    )
    policy.eval()
    g = _tiny_graph()
    obs = _pack_obs(g, np.zeros((n_actions,), dtype=np.float32))
    values = policy.predict_values(obs)
    assert values.shape[0] == 1
    assert torch.isfinite(values).all()


def test_old_new_log_probabilities_match_before_optimization():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    cfg = ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    policy = GraphActorCriticPolicy(
        observation_space=spaces.Dict(
            {
                "dummy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "action_mask": spaces.Box(
                    0.0, 1.0, shape=(env.action_space.n,), dtype=np.float32
                ),
                "graph": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        ),
        action_space=env.action_space,
        lr_schedule=lambda _: 1e-3,
        model_config=cfg,
    )
    policy.eval()
    packed = _pack_obs(obs["graph"], obs["action_mask"])
    with torch.no_grad():
        actions, _, log_prob = policy.forward(packed, deterministic=True)
        _, log_prob2, _ = policy.evaluate_actions(packed, actions)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    env.close()


def test_batched_outputs_match_scalar_path():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    graphs = []
    masks = []
    for seed in (0, 1):
        obs, _ = env.reset(seed=seed)
        graphs.append(obs["graph"])
        masks.append(obs["action_mask"])
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.eval()
    with torch.no_grad():
        batch_logits, batch_values = ac(graphs, action_mask=masks)
        single = [
            ac.forward_single(g, torch.as_tensor(m))
            for g, m in zip(graphs, masks)
        ]
    for i, (logits_i, value_i) in enumerate(single):
        assert torch.allclose(batch_logits[i], logits_i, atol=1e-5)
        assert torch.allclose(batch_values[i], value_i, atol=1e-5)
    env.close()

"""Batch encode / actor-critic batch forward shape checks."""

from __future__ import annotations

import torch

from config import ModelConfig
from envs.fjsp_env import FJSPEnv
from models.actor_critic import GraphActorCritic
from models.graph_encoder import GraphEncoder


def test_encode_batch_same_size_shapes():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    g0 = env.reset(seed=0)[0]["graph"]
    g1 = env.reset(seed=1)[0]["graph"]
    enc = GraphEncoder(hidden_dim=32, num_layers=2, num_heads=2)
    enc.eval()
    m_list, o_list, g_batch = enc.encode_batch([g0, g1])
    assert len(m_list) == 2 and len(o_list) == 2
    assert g_batch.shape == (2, 32)
    assert m_list[0].shape[0] == env.n_machines
    assert o_list[0].shape[0] == env.n_operations
    env.close()


def test_actor_critic_batch_forward():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    graphs = []
    masks = []
    for i in range(2):
        obs, _ = env.reset(seed=i)
        graphs.append(obs["graph"])
        masks.append(obs["action_mask"])
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=32, num_layers=2, num_heads=2, critic_hidden_dim=64)
    )
    ac.eval()
    logits, values = ac(graphs, action_mask=masks)
    assert logits.shape == (2, env.n_machines * env.n_operations)
    assert values.shape == (2,)
    env.close()


def test_slim_obs_retains_efficiency_edge_attr():
    from training.graph_buffer import slim_graph_for_policy

    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    slim = slim_graph_for_policy(obs["graph"])
    key = ("operation", "compatible", "machine")
    assert key in slim.edge_types
    assert slim[key].edge_attr is not None
    assert slim[key].edge_attr.numel() > 0
    env.close()


def test_batched_actor_gradient_parity_with_scalar():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    graphs = []
    for seed in (0, 1):
        obs, _ = env.reset(seed=seed)
        graphs.append(obs["graph"])
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.train()
    logits_b, values_b = ac(graphs)
    loss_b = logits_b.float().sum() + values_b.sum()
    loss_b.backward()
    grad_b = {
        n: p.grad.detach().clone()
        for n, p in ac.named_parameters()
        if p.grad is not None
    }
    ac.zero_grad()
    loss_s = 0.0
    for g in graphs:
        logits_i, value_i = ac.forward_single(g)
        loss_s = loss_s + logits_i.sum() + value_i
    loss_s.backward()
    assert torch.allclose(loss_b.detach(), loss_s.detach(), atol=1e-5)
    for name, g_b in grad_b.items():
        g_s = dict(ac.named_parameters())[name].grad
        assert torch.allclose(g_b, g_s, atol=1e-5), name
    env.close()

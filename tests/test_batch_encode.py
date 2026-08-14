"""Batch encode / actor-critic batch forward shape checks."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.nn import TransformerConv

from config import ModelConfig, get_debug_train_config
from envs.fjsp_env import FJSPEnv
from models.actor_critic import GraphActorCritic
from models.edge_predictor import EdgePredictor
from models.graph_encoder import GraphEncoder, REVERSE_EDGE_TYPES


def _mixed_edge_rollout_graphs(n_graphs: int = 8):
    """Reset + mid-episode snapshots so some graphs lack next/processing edges."""
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    rng = np.random.default_rng(0)
    graphs = []
    masks = []
    obs, _ = env.reset(seed=0)
    for _ in range(n_graphs):
        graphs.append(obs["graph"])
        masks.append(obs["action_mask"])
        valid = np.flatnonzero(np.asarray(obs["action_mask"]) >= 0.5)
        if valid.size == 0:
            obs, _ = env.reset()
            continue
        obs, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return graphs, masks


def test_encode_batch_matches_independent_forward():
    """Same-size collation must not change embeddings (PPO log-prob invariance).

    Reset-only graphs all share the same edge types and hide the bug:
    TransformerConv add_self_loops on a collated batch fires for graphs that
    omitted that type.
    """
    graphs, masks = _mixed_edge_rollout_graphs(8)
    enc = GraphEncoder(hidden_dim=32, num_layers=2, num_heads=2)
    enc.eval()
    with torch.no_grad():
        m_list, o_list, g_batch = enc.encode_batch(graphs)
        for i, graph in enumerate(graphs):
            m_i, o_i, g_i = enc.forward(graph)
            assert torch.allclose(m_list[i], m_i, atol=1e-5), f"machine emb {i}"
            assert torch.allclose(o_list[i], o_i, atol=1e-5), f"operation emb {i}"
            assert torch.allclose(g_batch[i], g_i, atol=1e-5), f"graph emb {i}"

    ac = GraphActorCritic(
        ModelConfig(hidden_dim=32, num_layers=2, num_heads=2, critic_hidden_dim=64)
    )
    ac.eval()
    with torch.no_grad():
        logits_b, values_b = ac(graphs, action_mask=masks)
        for i, graph in enumerate(graphs):
            logits_i, value_i = ac.forward_single(graph, torch.as_tensor(masks[i]))
            assert torch.allclose(logits_b[i], logits_i, atol=1e-5), f"logits {i}"
            assert torch.allclose(values_b[i], value_i, atol=1e-5), f"value {i}"


def test_ppo_pre_update_kl_near_zero(tmp_path):
    """Collect-time log-probs must match batched evaluate_actions before SGD.

    SB3 measures approx_kl before the optimizer step; a mismatch here aborts
    every PPO epoch at step 0 (target_kl).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    from train import build_ppo
    from training.make_env import make_vec_env

    cfg = get_debug_train_config()
    cfg.n_envs = 1
    cfg.device = "cpu"
    cfg.tensorboard_log = str(tmp_path)
    cfg.ppo.n_steps = 32
    cfg.ppo.batch_size = 32
    cfg.ppo.n_epochs = 1
    env = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=False)
    model = build_ppo(cfg, env)

    class Nop(BaseCallback):
        def _on_step(self):
            return True

    callback = Nop()
    callback.init_callback(model)
    model._setup_learn(
        total_timesteps=32,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="PPO",
        progress_bar=False,
    )
    model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=32)
    batch = next(model.rollout_buffer.get(32))
    model.policy.eval()
    with torch.no_grad():
        _values, log_prob, _entropy = model.policy.evaluate_actions(
            batch.observations, batch.actions.long().flatten()
        )
        log_ratio = log_prob - batch.old_log_prob
        approx_kl = float(torch.mean((torch.exp(log_ratio) - 1) - log_ratio))
    env.close()
    assert approx_kl < 1e-4, f"pre-update approx_kl={approx_kl:.4f} (collect vs batched train)"


def test_actor_critic_batch_forward_uses_encode_batch(monkeypatch):
    """Train-path batch forward must use encode_batch, not a Python loop of encoder.forward."""
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    graphs = [env.reset(seed=i)[0]["graph"] for i in range(2)]
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.eval()
    calls = {"batch": 0}
    orig = GraphEncoder.encode_batch

    def _counting_batch(self, graphs_in):
        calls["batch"] += 1
        return orig(self, graphs_in)

    monkeypatch.setattr(GraphEncoder, "encode_batch", _counting_batch)
    with torch.no_grad():
        ac(graphs)
    assert calls["batch"] == 1
    env.close()


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


def test_encoder_uses_multihead_transformer_and_reverse_relations():
    enc = GraphEncoder(hidden_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    convs = enc.layers[0].conv.convs
    assert all(isinstance(conv, TransformerConv) for conv in convs.values())
    assert all(conv.heads == 4 for conv in convs.values())
    assert all(conv.edge_dim == 1 for conv in convs.values())
    assert all(conv.root_weight is False for conv in convs.values())
    assert all(reverse in convs for reverse in REVERSE_EDGE_TYPES.values())


def test_compatibility_edge_attributes_reach_operation_embeddings():
    env = FJSPEnv(
        n_machines=3,
        n_jobs=2,
        avg_operations_per_job=2,
        seed=0,
        device="cpu",
    )
    graph = env.reset(seed=0)[0]["graph"]
    changed = graph.clone()
    key = ("operation", "compatible", "machine")
    changed[key].edge_attr = changed[key].edge_attr * 0.5
    enc = GraphEncoder(hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        machines_a, operations_a, graph_a = enc(graph)
        machines_b, operations_b, graph_b = enc(changed)
    # Compatible edges update machines; reverse edges / pooling carry the change.
    assert not torch.allclose(machines_a, machines_b)
    assert not torch.allclose(graph_a, graph_b)
    env.close()


def test_predictor_logits_are_pairwise():
    torch.manual_seed(0)
    machines = torch.randn(2, 8)
    operations = torch.randn(3, 8)
    for predictor_type in ("dot_product", "bilinear"):
        predictor = EdgePredictor(hidden_dim=8, predictor_type=predictor_type)
        predictor.norm_m = torch.nn.Identity()
        predictor.norm_o = torch.nn.Identity()
        predictor.eval()
        with torch.no_grad():
            base = predictor(machines, operations).view(2, 3)
            ops2 = operations.clone()
            ops2[1] += 1.0
            changed = predictor(machines, ops2).view(2, 3)
        assert not torch.allclose(base[:, 1], changed[:, 1])
        assert torch.allclose(base[:, 0], changed[:, 0])
        assert torch.allclose(base[:, 2], changed[:, 2])


def test_bilinear_predictor_uses_efficiency_bias():
    torch.manual_seed(0)
    predictor = EdgePredictor(hidden_dim=8, predictor_type="bilinear")
    predictor.eval()
    machines = torch.randn(2, 8)
    operations = torch.randn(3, 8)
    with torch.no_grad():
        without_eff = predictor(machines, operations, None)
        with_eff = predictor(machines, operations, torch.ones(3, 2))
    assert without_eff.shape == (6,)
    assert not torch.allclose(without_eff, with_eff)


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
        assert g_s is not None, name
        assert torch.isfinite(g_b).all() and torch.isfinite(g_s).all(), name
    env.close()

"""Size-agnostic Gym spaces, policy forward, and fixed train/eval instance size."""

from __future__ import annotations

import numpy as np
import torch

from config import ModelConfig, get_debug_train_config
from envs.fjsp_env import (
    FJSPEnv,
    GraphObsSpace,
    make_sb3_action_space,
    make_sb3_graph_observation_space,
)
from models.actor_critic import MASK_LOGIT, GraphActorCritic
from models.sb3_policy import GraphActorCriticPolicy
from training.make_env import make_vec_env


def _pack_obs(graph, action_mask: np.ndarray) -> dict:
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


def test_gym_spaces_independent_of_instance_size():
    a = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    b = FJSPEnv(n_machines=4, n_jobs=2, avg_operations_per_job=3, seed=0, device="cpu")
    assert a.action_space == b.action_space == make_sb3_action_space()
    assert make_sb3_graph_observation_space() == make_sb3_graph_observation_space(99)
    obs_a, _ = a.reset(seed=0)
    obs_b, _ = b.reset(seed=0)
    space = GraphObsSpace()
    assert space.contains(obs_a)
    assert space.contains(obs_b)
    assert obs_a["action_mask"].shape != obs_b["action_mask"].shape
    a.close()
    b.close()


def test_same_policy_forward_on_two_instance_sizes():
    policy = GraphActorCriticPolicy(
        observation_space=make_sb3_graph_observation_space(),
        action_space=make_sb3_action_space(),
        lr_schedule=lambda _: 1e-3,
        model_config=ModelConfig(
            hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32
        ),
    )
    policy.eval()
    env_small = FJSPEnv(
        n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu"
    )
    env_large = FJSPEnv(
        n_machines=4, n_jobs=2, avg_operations_per_job=3, seed=0, device="cpu"
    )
    obs_s, _ = env_small.reset(seed=0)
    obs_l, _ = env_large.reset(seed=0)
    with torch.no_grad():
        actions_s, values_s, logp_s = policy.forward(
            _pack_obs(obs_s["graph"], obs_s["action_mask"]), deterministic=True
        )
        actions_l, values_l, logp_l = policy.forward(
            _pack_obs(obs_l["graph"], obs_l["action_mask"]), deterministic=True
        )
    assert int(actions_s.item()) < int(obs_s["action_mask"].size)
    assert int(actions_l.item()) < int(obs_l["action_mask"].size)
    assert torch.isfinite(values_s).all() and torch.isfinite(values_l).all()
    assert torch.isfinite(logp_s).all() and torch.isfinite(logp_l).all()
    env_small.close()
    env_large.close()


def test_mixed_size_forward_pads_logits():
    env_a = FJSPEnv(
        n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu"
    )
    env_b = FJSPEnv(
        n_machines=4, n_jobs=2, avg_operations_per_job=3, seed=0, device="cpu"
    )
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    ac = GraphActorCritic(
        ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, critic_hidden_dim=32)
    )
    ac.eval()
    a_small = int(obs_a["action_mask"].size)
    a_large = int(obs_b["action_mask"].size)
    with torch.no_grad():
        logits, values = ac(
            [obs_a["graph"], obs_b["graph"]],
            action_mask=[obs_a["action_mask"], obs_b["action_mask"]],
        )
        logits_a, _ = ac.forward_single(
            obs_a["graph"], torch.as_tensor(obs_a["action_mask"])
        )
    assert logits.shape == (2, a_large)
    assert values.shape == (2,)
    assert torch.allclose(logits[0, :a_small], logits_a, atol=1e-5)
    assert torch.all(logits[0, a_small:] == MASK_LOGIT)
    env_a.close()
    env_b.close()


def test_reset_keeps_configured_instance_size():
    env = FJSPEnv(
        n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu"
    )
    for _ in range(3):
        obs, _ = env.reset()
        assert env.n_machines == 5
        assert env.n_jobs == 3
        assert env.n_operations == 12
        assert obs["action_mask"].shape == (60,)
    env.close()

    cfg = get_debug_train_config()
    cfg.n_envs = 1
    vec = make_vec_env(cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        assert vec.action_space == make_sb3_action_space()
        assert tuple(vec.observation_space.spaces["action_mask"].shape) == (0,)
        inner = vec.envs[0].unwrapped
        for _ in range(2):
            vec.reset()
            assert inner.n_machines == cfg.env.n_machines
            assert inner.n_jobs == cfg.env.n_jobs
            assert inner.n_operations == cfg.env.n_operations
    finally:
        vec.close()


def test_eligibility_clamps_keep_k_to_n_machines():
    env = FJSPEnv(
        n_machines=1,
        n_jobs=1,
        avg_operations_per_job=2,
        min_eligible_machines=2,
        seed=0,
        device="cpu",
    )
    env.reset(seed=0)
    assert tuple(env.eligibility_matrix.shape) == (2, 1)
    assert bool(env.eligibility_matrix.all().item())
    env.close()

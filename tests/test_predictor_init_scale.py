"""Init-scale / entropy checks for edge predictors (PPO stability)."""

from __future__ import annotations

import torch

from config import ModelConfig
from envs.fjsp_env import FJSPEnv
from models.actor_critic import GraphActorCritic


def test_bilinear_init_entropy_not_collapsed():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    mask = torch.as_tensor(obs["action_mask"])
    ents = []
    maxps = []
    for seed in range(8):
        torch.manual_seed(seed)
        ac = GraphActorCritic(
            ModelConfig(
                hidden_dim=64,
                num_layers=2,
                num_heads=2,
                critic_hidden_dim=128,
                predictor_type="bilinear",
            )
        )
        ac.eval()
        with torch.no_grad():
            logits, _ = ac.forward_single(obs["graph"], mask)
            probs = torch.softmax(logits, dim=-1)
            ents.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
            maxps.append(float(probs.max()))
    # Uniform over ~8 valid actions has H≈2.0; collapsed one-hot is ~0.
    assert min(ents) > 0.5, f"entropy collapsed at init: {ents}"
    assert sum(ents) / len(ents) > 1.0, f"mean entropy too low: {ents}"
    assert max(maxps) < 0.9, f"max prob too peaked at init: {maxps}"
    env.close()

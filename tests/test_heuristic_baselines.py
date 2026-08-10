"""Unit and smoke tests for dispatching-rule heuristic baselines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from baseline_heuristic import parse_args
from envs.fjsp_env import (
    OP_DURATION,
    OP_ELIGIBLE_COUNT,
    OP_FINISHED,
    OP_SCHEDULED,
    FJSPEnv,
)
from heuristics import RULES, select_heuristic_action
from training.evaluate import evaluate_heuristic_fjsp
from training.make_env import make_vec_env
from config import get_default_eval_config
from training.eval_cli import build_eval_train_config


class _FakeEnv:
    """Minimal stand-in for SPT scoring tests."""

    def __init__(self):
        self.n_operations = 2
        self.n_machines = 2
        self.job_sequences = [[0], [1]]
        # Actions: 0=(m0,o0), 1=(m0,o1), 2=(m1,o0), 3=(m1,o1)
        # Valid: (m0,o0) eff=2 → 10, (m1,o0) eff=0.5 → 2.5  → SPT picks 2
        self.state = {
            "operation": type("X", (), {})(),
            "machine": type("X", (), {})(),
        }
        self.state["operation"].x = torch.tensor(
            [
                [5.0, 0, 0, 0, 0, 0, 0, 0, 5.0, 2.0],
                [9.0, 0, 0, 0, 0, 0, 0, 0, 9.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.state["machine"].x = torch.zeros(2, 3)
        self.efficiency_modifiers = torch.tensor(
            [[2.0, 0.5], [1.0, 1.0]],
            dtype=torch.float32,
        )
        self._mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def get_action_mask(self):
        return self._mask.copy()


def test_spt_prefers_shorter_effective_time():
    env = _FakeEnv()
    assert select_heuristic_action(env, "SPT") == 2
    assert select_heuristic_action(env, "LPT") == 0


def test_empty_mask_raises():
    env = _FakeEnv()
    env._mask[:] = 0.0
    with pytest.raises(ValueError, match="empty"):
        select_heuristic_action(env, "SPT")


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown rule"):
        select_heuristic_action(_FakeEnv(), "NOT_A_RULE")


def test_cli_rule_validation():
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--rule", "NOPE"])
    args = parse_args(["--rule", "spt", "--n-episodes", "1"])
    assert args.rule == "SPT"
    assert "SPT" in RULES


def test_tie_breaks_lowest_action_index():
    env = _FakeEnv()
    env.efficiency_modifiers[:] = 1.0
    env.state["operation"].x[0, OP_DURATION] = 5.0
    # Both valid actions for op0 have same effective time → pick lower index 0.
    assert select_heuristic_action(env, "SPT") == 0


def test_heuristic_smoke_rollout():
    cfg = get_default_eval_config()
    cfg.n_episodes = 1
    cfg.env.n_machines = 2
    cfg.env.n_jobs = 2
    cfg.env.avg_operations_per_job = 2
    train_cfg = build_eval_train_config(cfg)
    env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
    try:
        env.seed(cfg.seed)
        result = evaluate_heuristic_fjsp(env, rule="SPT", n_episodes=1)
        assert result.n_episodes == 1
        assert np.isfinite(result.mean_reward)
    finally:
        env.close()


def test_all_heuristics_repeatable_on_held_out_seed():
    cfg = get_default_eval_config()
    cfg.seed = 23
    cfg.n_episodes = 2
    cfg.env.n_machines = 2
    cfg.env.n_jobs = 2
    cfg.env.avg_operations_per_job = 2
    train_cfg = build_eval_train_config(cfg)
    assert train_cfg.eval_seed == cfg.seed

    for rule in sorted(RULES):
        def _run(_rule=rule):
            env = make_vec_env(train_cfg, n_envs=1, use_subprocess=False, for_eval=True)
            try:
                env.seed(cfg.seed)
                return evaluate_heuristic_fjsp(env, rule=_rule, n_episodes=cfg.n_episodes)
            finally:
                env.close()

        a = _run()
        b = _run()
        assert a.mean_reward == pytest.approx(b.mean_reward), rule
        assert a.mean_ep_length == pytest.approx(b.mean_ep_length), rule



def test_fifo_on_live_env_picks_lowest_op_index():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {}
    env.state["operation", "precede", "operation"].edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, OP_SCHEDULED] = 0
    env.state["operation"].x[:, OP_FINISHED] = 0
    env.state["operation"].x[:, OP_ELIGIBLE_COUNT] = 2
    env._cached_action_mask = None
    action = select_heuristic_action(env, "FIFO")
    _, op = divmod(action, env.n_operations)
    assert op == 0
    env.close()

"""Lookahead operation features: CP remaining, job remaining, ready flag."""

from __future__ import annotations

import numpy as np
import torch

from envs.fjsp_env import (
    FJSPEnv,
    OP_CP_REMAINING,
    OP_FEATURE_DIM,
    OP_FINISHED,
    OP_JOB_REMAINING_OPS,
    OP_JOB_REMAINING_WORK,
    OP_READY,
    OP_REMAINING,
    OP_SCHEDULED,
)
from utils import unflatten_action


def test_reset_obs_has_lookahead_columns():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    op_x = obs["graph"]["operation"].x
    assert op_x.shape == (env.n_operations, OP_FEATURE_DIM)
    mask = np.asarray(obs["action_mask"]).reshape(env.n_machines, env.n_operations)
    ready_from_mask = mask.max(axis=0) > 0.5
    ready_feat = op_x[:, OP_READY].cpu().numpy() > 0.5
    assert np.array_equal(ready_feat, ready_from_mask)
    unfinished = op_x[:, OP_FINISHED] < 0.5
    assert bool(torch.all(op_x[unfinished, OP_CP_REMAINING] >= op_x[unfinished, OP_REMAINING]).item())
    env.close()


def test_critical_path_on_chain():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=3, seed=0, device="cpu")
    env.reset(seed=0)
    assert env.n_operations == 3
    env.state["operation", "precede", "operation"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, OP_REMAINING] = 0
    env.state["operation"].x[0, OP_REMAINING] = 3.0
    env.state["operation"].x[1, OP_REMAINING] = 2.0
    env.state["operation"].x[2, OP_REMAINING] = 1.0
    env.state["operation"].x[:, OP_FINISHED] = 0
    env._refresh_lookahead_features()
    cp = env.state["operation"].x[:, OP_CP_REMAINING]
    assert float(cp[2]) == 1.0
    assert float(cp[1]) == 3.0
    assert float(cp[0]) == 6.0
    env.close()


def test_job_remaining_tracks_unfinished_members():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    seq = env.job_sequences[0]
    op_x = env.state["operation"].x
    expected_work = float(op_x[seq, OP_REMAINING].sum().item())
    expected_ops = float(len(seq))
    for op in seq:
        assert float(op_x[op, OP_JOB_REMAINING_WORK]) == expected_work
        assert float(op_x[op, OP_JOB_REMAINING_OPS]) == expected_ops
    env.state["operation"].x[seq[0], OP_FINISHED] = 1
    env.state["operation"].x[seq[0], OP_REMAINING] = 0
    env._refresh_lookahead_features()
    op_x = env.state["operation"].x
    rest = seq[1:]
    expected_work = float(op_x[rest, OP_REMAINING].sum().item())
    for op in seq:
        assert float(op_x[op, OP_JOB_REMAINING_WORK]) == expected_work
        assert float(op_x[op, OP_JOB_REMAINING_OPS]) == float(len(rest))
    env.close()


def test_ready_clears_after_schedule():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    action = int(np.flatnonzero(obs["action_mask"])[0])
    _, op = unflatten_action(action, env.n_operations)
    assert float(obs["graph"]["operation"].x[op, OP_READY]) == 1.0
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        env.close()
        return
    assert float(obs["graph"]["operation"].x[op, OP_READY]) == 0.0
    assert float(env.state["operation"].x[op, OP_SCHEDULED]) == 1.0
    env.close()


def test_nonterminal_step_refreshes_lookahead_once(monkeypatch):
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    calls = {"n": 0}
    orig = FJSPEnv._refresh_lookahead_features

    def _counting(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(FJSPEnv, "_refresh_lookahead_features", _counting)
    action = int(np.flatnonzero(obs["action_mask"])[0])
    _obs, _reward, terminated, truncated, _info = env.step(action)
    if terminated or truncated:
        env.close()
        return
    assert calls["n"] == 1
    env.close()

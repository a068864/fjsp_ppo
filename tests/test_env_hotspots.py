"""Hotspot refactor checks: action-mask semantics and DAG invariants."""

from __future__ import annotations

import numpy as np
import torch

from envs.fjsp_env import FJSPEnv, OP_FINISHED, OP_SCHEDULED


def _reference_valid_actions(env: FJSPEnv) -> set[int]:
    """Mirror pre-refactor semantics: prereq met if any of scheduled/processing/finished."""
    state = env.state
    op_x = state["operation"].x
    unscheduled = torch.where((op_x[:, OP_SCHEDULED] == 0) & (op_x[:, OP_FINISHED] == 0))[0]
    dep = state["operation", "precede", "operation"].edge_index
    valid: set[int] = set()
    for op in unscheduled.tolist():
        ok = True
        if dep.numel() > 0:
            prereqs = dep[0][dep[1] == op]
            for p in prereqs.tolist():
                if float(op_x[p, 5:8].sum()) == 0.0:
                    ok = False
                    break
        if ok:
            for m in torch.where(env.eligibility_matrix[op])[0].tolist():
                valid.add(int(m * env.n_operations + op))
    return valid


def test_action_mask_matches_reference_semantics():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    env.reset(seed=0)
    for _ in range(5):
        mask = env.get_action_mask()
        got = set(np.flatnonzero(mask).tolist())
        assert got == _reference_valid_actions(env)
        valid = list(got)
        if not valid:
            break
        _, _, term, trunc, _ = env.step(int(valid[0]))
        if term or trunc:
            break
    env.close()


def test_dependencies_acyclic_and_include_sequential():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=1, device="cpu")
    env.reset(seed=1)
    deps = list(env.dependency_types.keys())
    all_ops = set(range(env.n_operations))
    assert not env._has_cycle(deps, all_ops)
    for seq in env.job_sequences:
        for a, b in zip(seq, seq[1:]):
            assert env.dependency_types[(a, b)] == "sequential"
    env.close()

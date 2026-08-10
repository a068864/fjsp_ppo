"""Environment invariants: discrete ticks, deps, masks, reset, CUDA/MPS."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from envs.fjsp_env import (
    FJSPEnv,
    GraphObsSpace,
    OP_CROSS_DEPS,
    OP_FINISHED,
    OP_PAR_DEPS,
    OP_REMAINING,
    OP_SEQ_DEPS,
    make_sb3_graph_observation_space,
)
from gymnasium import spaces
from torch_geometric.data import HeteroData


def _sum_machine_workload(env: FJSPEnv) -> float:
    return float(env.state["machine"].x[:, 1].sum().item())


def _sum_queued_remaining(env: FJSPEnv) -> float:
    proc = env.state["machine", "processing", "operation"].edge_index
    if proc.numel() == 0:
        return 0.0
    ops = proc[1].unique()
    return float(env.state["operation"].x[ops, OP_REMAINING].sum().item())


def test_graph_obs_space_rejects_nan_and_oob_masks():
    space = GraphObsSpace(4)
    base = {
        "dummy": np.zeros((1,), dtype=np.float32),
        "graph": HeteroData(),
    }
    bad_nan = {**base, "action_mask": np.array([1, 0, np.nan, 0], dtype=np.float32)}
    bad_oob = {**base, "action_mask": np.array([1, 0, 2, 0], dtype=np.float32)}
    good = {**base, "action_mask": np.array([1, 0, 1, 0], dtype=np.float32)}
    assert not space.contains(bad_nan)
    assert not space.contains(bad_oob)
    assert space.contains(good)


def test_sb3_space_graph_subspace_is_truthful():
    space = make_sb3_graph_observation_space(8)
    assert isinstance(space, spaces.Dict)
    graph_space = space.spaces["graph"]
    # Placeholder Box(1,) is not truthful; expect an opaque empty-shaped Box.
    assert isinstance(graph_space, spaces.Box)
    assert tuple(graph_space.shape) != (1,)
    assert tuple(graph_space.shape) == (0,)


def test_action_mask_is_not_shared_mutable_cache():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    m1 = env.get_action_mask()
    m1[:] = 0
    m2 = env.get_action_mask()
    assert float(m2.sum()) > 0.0
    env.close()


def test_invalid_action_raises_value_error_not_index_error():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(10_000)
    env.close()


def test_dependency_pairs_are_unique():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=7, device="cpu")
    env.reset(seed=7)
    pairs = list(env.dependency_types.keys())
    assert len(pairs) == len(set(pairs))
    env.close()


def test_successor_dependency_counts_update_on_completion():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    # Force a simple chain 0 -> 1 with known counts.
    env.job_sequences = [[0, 1]]
    env.dependency_types = {(0, 1): "sequential"}
    env.state["operation", "precede", "operation"].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, OP_SEQ_DEPS] = 0
    env.state["operation"].x[1, OP_SEQ_DEPS] = 1
    env.state["operation"].x[:, OP_PAR_DEPS] = 0
    env.state["operation"].x[:, OP_CROSS_DEPS] = 0
    env.eligibility_matrix[:, :] = True
    env._cached_action_mask = None

    # Schedule and complete op 0 alone on machine 0 with tiny remaining.
    env.schedule_operation(0, 0)
    env.state["operation"].x[0, OP_REMAINING] = env.time_step
    env.state["operation"].x[0, 0] = env.time_step
    env.state["machine"].x[0, 1] = env.time_step
    # Manually run one completion path via shared tick after forcing processing.
    completed = env._advance_time_tick()
    assert 0 in completed or bool(env.state["operation"].x[0, OP_FINISHED] > 0.5)
    assert float(env.state["operation"].x[1, OP_SEQ_DEPS].item()) == pytest.approx(0.0)
    env.close()


def test_tick_subtracts_only_actual_work_no_fraction_transfer():
    env = FJSPEnv(
        n_machines=2,
        n_jobs=1,
        avg_operations_per_job=2,
        seed=0,
        device="cpu",
        min_eligible_machines=1,
    )
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {}
    env.state["operation", "precede", "operation"].edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, :] = 0
    env.state["operation"].x[:, 0] = torch.tensor([0.4, 5.0], device=env.device)
    env.state["operation"].x[:, OP_REMAINING] = torch.tensor([0.4, 5.0], device=env.device)
    env.state["machine"].x[:] = 0
    env._cached_action_mask = None

    env.schedule_operation(0, 0)
    env.schedule_operation(0, 1)
    before_q = float(env.state["operation"].x[1, OP_REMAINING].item())
    env._advance_time_tick()
    # Front op consumes only 0.4; unused 0.6 of the tick must not hit queued op 1.
    assert float(env.state["operation"].x[1, OP_REMAINING].item()) == pytest.approx(before_q)
    assert float(env.state["operation"].x[0, OP_FINISHED].item()) == pytest.approx(1.0)
    env.close()


def test_rollout_matches_repeated_tick_advance():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {}
    env.state["operation", "precede", "operation"].edge_index = torch.empty(
        (2, 0), dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, :] = 0
    env.state["operation"].x[:, 0] = torch.tensor([1.0, 1.0], device=env.device)
    env.state["operation"].x[:, OP_REMAINING] = torch.tensor([1.0, 1.0], device=env.device)
    env.state["machine"].x[:] = 0
    env.schedule_operation(0, 0)
    env.schedule_operation(1, 1)
    snapshot = env.state.clone()
    t0 = env.current_time

    env.state = snapshot.clone()
    env.current_time = t0
    r_rollout, success_rollout = env.rollout()
    time_rollout = env.current_time

    env.state = snapshot.clone()
    env.current_time = t0
    r_ticks = 0.0
    for _ in range(100):
        if bool(torch.all(env.state["operation"].x[:, OP_FINISHED] > 0.5)):
            break
        processing, blocked = env._get_processing_operations()
        if not processing.any():
            break
        env._advance_time_tick()
        r_ticks += float(env.time_penalty)
    success_ticks = bool(torch.all(env.state["operation"].x[:, OP_FINISHED] > 0.5))
    assert success_rollout == success_ticks
    assert r_rollout == pytest.approx(r_ticks, abs=1e-5)
    assert time_rollout == pytest.approx(env.current_time, abs=1e-5)
    env.close()


def test_workload_tracks_queued_remaining_after_ticks():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=1, device="cpu")
    env.reset(seed=1)
    for _ in range(4):
        mask = env.get_action_mask()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        _, _, term, trunc, _ = env.step(int(valid[0]))
        # After each discrete tick, machine workload equals remaining work on queued ops.
        assert _sum_machine_workload(env) == pytest.approx(
            _sum_queued_remaining(env), abs=1e-4
        )
        if term or trunc:
            break
    env.close()


def test_reset_reuse_instance_restores_cached_graph():
    env = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    obs1, _ = env.reset(seed=0)
    x1 = obs1["graph"]["operation"].x.clone()
    mask = env.get_action_mask()
    env.step(int(np.flatnonzero(mask)[0]))
    obs2, _ = env.reset(seed=0, options={"reuse_instance": True})
    assert torch.allclose(obs2["graph"]["operation"].x, x1)
    env.close()


def test_unseeded_torch_generators_differ_across_envs():
    a = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=None, device="cpu")
    b = FJSPEnv(n_machines=3, n_jobs=2, avg_operations_per_job=2, seed=None, device="cpu")
    a.reset()
    b.reset()
    # Efficiency matrices come from torch_gen; unseeded envs must not share default seed.
    assert not torch.equal(a.efficiency_modifiers, b.efficiency_modifiers)
    a.close()
    b.close()


def test_gridlock_when_queued_fronts_blocked_with_idle_machine():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    # Build artificial gridlock: op1 depends on unfinished op0, queued on m0; m1 idle.
    env.dependency_types = {(0, 1): "sequential"}
    env.state["operation", "precede", "operation"].edge_index = torch.tensor(
        [[0], [1]], dtype=torch.long, device=env.device
    )
    env.state["operation"].x[:, :] = 0
    env.state["operation"].x[0, OP_FINISHED] = 0
    env.state["operation"].x[:, 0] = 2.0
    env.state["operation"].x[:, OP_REMAINING] = 2.0
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.state["machine"].x[:] = 0
    env.schedule_operation(0, 1)  # front blocked by dep on op0
    processing, blocked = env._get_processing_operations()
    assert not processing.any()
    assert env._is_gridlock(processing, blocked)
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_optional_cuda_step_and_rollout():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="cuda")
    env.reset(seed=0)
    mask = env.get_action_mask()
    valid = np.flatnonzero(mask)
    assert valid.size > 0
    env.step(int(valid[0]))
    env.close()


@pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ),
    reason="MPS not available",
)
def test_optional_mps_step_and_rollout():
    env = FJSPEnv(n_machines=2, n_jobs=2, avg_operations_per_job=2, seed=0, device="mps")
    assert env.device.type == "mps"
    env.reset(seed=0)
    mask = env.get_action_mask()
    valid = np.flatnonzero(mask)
    assert valid.size > 0
    env.step(int(valid[0]))
    env.close()

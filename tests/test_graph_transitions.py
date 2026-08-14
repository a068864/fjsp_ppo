"""Heterograph node/edge updates across FJSP state transitions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch

from envs.fjsp_env import (
    FJSPEnv,
    OP_CROSS_DEPS,
    OP_DURATION,
    OP_FINISHED,
    OP_PAR_DEPS,
    OP_PROCESSING,
    OP_PROGRESS,
    OP_REMAINING,
    OP_SCHEDULED,
    OP_SEQ_DEPS,
)
from models.graph_encoder import EDGE_TYPES

COMPAT = ("operation", "compatible", "machine")
PROC = ("machine", "processing", "operation")
NEXT = ("operation", "next", "operation")
PRECEDE = ("operation", "precede", "operation")
PROCESSED_BY = ("operation", "processed_by", "machine")


def _edge_pairs(graph, key) -> list[tuple[int, int]]:
    if key not in graph.edge_types:
        return []
    ei = graph[key].edge_index
    if ei is None or ei.numel() == 0:
        return []
    return list(zip(ei[0].tolist(), ei[1].tolist()))


def _assert_attr_aligned(graph, key: tuple, *, require: bool) -> None:
    if key not in graph.edge_types:
        if require:
            pytest.fail(f"missing required edge type {key}")
        return
    ei = graph[key].edge_index
    attr = getattr(graph[key], "edge_attr", None)
    if ei is None or ei.numel() == 0:
        if attr is not None:
            assert attr.size(0) == 0
        return
    if require:
        assert attr is not None, f"{key} missing edge_attr"
        assert attr.size(0) == ei.size(1), f"{key} attr rows {attr.size(0)} != edges {ei.size(1)}"
        assert attr.size(1) == 1


def _machine_queues(env: FJSPEnv) -> dict[int, list[int]]:
    queues: dict[int, list[int]] = defaultdict(list)
    for m, op in _edge_pairs(env.state, PROC):
        queues[int(m)].append(int(op))
    return queues


def _assert_graph_consistent(env: FJSPEnv, graph) -> None:
    op_x = graph["operation"].x
    mach_x = graph["machine"].x
    assert torch.allclose(op_x, env.state["operation"].x)
    assert torch.allclose(mach_x, env.state["machine"].x)
    assert op_x.shape == (env.n_operations, 14)
    assert mach_x.shape == (env.n_machines, 3)

    scheduled = op_x[:, OP_SCHEDULED] > 0.5
    processing = op_x[:, OP_PROCESSING] > 0.5
    finished = op_x[:, OP_FINISHED] > 0.5
    assert not bool((scheduled & finished).any().item())
    assert not bool((processing & finished).any().item())
    assert bool((~processing | scheduled).all().item())

    remaining = op_x[:, OP_REMAINING]
    duration = op_x[:, OP_DURATION]
    progress = op_x[:, OP_PROGRESS]
    assert bool((remaining >= -1e-5).all().item())
    assert bool((progress >= -1e-5).all().item() and (progress <= 1 + 1e-5).all().item())
    live = scheduled | processing
    if bool(live.any().item()):
        expected = torch.clamp(1 - remaining[live] / duration[live].clamp(min=1e-8), 0, 1)
        assert torch.allclose(progress[live], expected, atol=1e-4)
        assert bool((remaining[live] <= duration[live] + 1e-4).all().item())
    if bool(finished.any().item()):
        assert torch.allclose(remaining[finished], torch.zeros_like(remaining[finished]), atol=1e-5)
        assert torch.allclose(progress[finished], torch.ones_like(progress[finished]), atol=1e-5)

    still_alive = bool((~finished).any().item())
    _assert_attr_aligned(graph, COMPAT, require=still_alive)
    _assert_attr_aligned(graph, PROC, require=bool(_edge_pairs(graph, PROC)))
    _assert_attr_aligned(env.state, PROC, require=bool(_edge_pairs(env.state, PROC)))

    queues = _machine_queues(env)
    for m in range(env.n_machines):
        assert float(mach_x[m, 0].item()) == pytest.approx(len(queues[m]), abs=1e-5)
        queued_rem = sum(float(op_x[op, OP_REMAINING].item()) for op in queues[m])
        assert float(mach_x[m, 1].item()) == pytest.approx(queued_rem, abs=1e-4)

    expected_next = {(a, b) for q in queues.values() for a, b in zip(q, q[1:])}
    assert set(_edge_pairs(graph, NEXT)) == expected_next
    assert set(_edge_pairs(env.state, NEXT)) == expected_next
    assert set(_edge_pairs(graph, PROC)) == set(_edge_pairs(env.state, PROC))
    assert set(_edge_pairs(graph, PRECEDE)) == set(_edge_pairs(env.state, PRECEDE))

    proc_pairs = _edge_pairs(graph, PROC)
    expected_rev = {(op, m) for m, op in proc_pairs}
    assert set(_edge_pairs(env.state, PROCESSED_BY)) == expected_rev
    _assert_attr_aligned(env.state, PROCESSED_BY, require=bool(expected_rev))
    if proc_pairs:
        attr = graph[PROC].edge_attr.reshape(-1)
        rev_attr = env.state[PROCESSED_BY].edge_attr.reshape(-1)
        rev_pairs = _edge_pairs(env.state, PROCESSED_BY)
        for i, (m, op) in enumerate(proc_pairs):
            want = float(env.efficiency_modifiers[op, m].item())
            assert float(attr[i].item()) == pytest.approx(want, abs=1e-5)
        for i, (op, m) in enumerate(rev_pairs):
            want = float(env.efficiency_modifiers[op, m].item())
            assert float(rev_attr[i].item()) == pytest.approx(want, abs=1e-5)

    compat_pairs = _edge_pairs(graph, COMPAT)
    if still_alive:
        assert compat_pairs, "policy graph dropped all compatible edges too early"
        if bool(finished.any().item()):
            for op, _m in compat_pairs:
                assert not bool(finished[op].item()), f"finished op {op} still has compatible edge in obs"
        compat_attr = graph[COMPAT].edge_attr.reshape(-1)
        for i, (op, m) in enumerate(compat_pairs):
            want = float(env.efficiency_modifiers[op, m].item())
            assert float(compat_attr[i].item()) == pytest.approx(want, abs=1e-5)
            assert bool(env.eligibility_matrix[op, m].item())

    incoming = {
        OP_SEQ_DEPS: defaultdict(int),
        OP_PAR_DEPS: defaultdict(int),
        OP_CROSS_DEPS: defaultdict(int),
    }
    col = {"sequential": OP_SEQ_DEPS, "parallel": OP_PAR_DEPS, "cross_job": OP_CROSS_DEPS}
    for src, dst in _edge_pairs(env.state, PRECEDE):
        assert not bool(finished[src].item())
        assert not bool(finished[dst].item())
        kind = env.dependency_types.get((src, dst))
        if kind in col:
            incoming[col[kind]][dst] += 1
    for i in range(env.n_operations):
        if bool(finished[i].item()):
            continue
        for feature, counts in incoming.items():
            assert float(op_x[i, feature].item()) == pytest.approx(counts[i], abs=1e-5)

    for edge_type in EDGE_TYPES:
        if edge_type not in graph.edge_types:
            continue
        ei = graph[edge_type].edge_index
        if ei is None or ei.numel() == 0:
            continue
        edge_attr = getattr(graph[edge_type], "edge_attr", None)
        if edge_attr is not None:
            assert edge_attr.size(0) == ei.size(1)


def test_graph_features_and_attrs_track_demo_episode():
    env = FJSPEnv(n_machines=5, n_jobs=3, avg_operations_per_job=4, seed=0, device="cpu")
    obs, _ = env.reset(seed=0)
    _assert_graph_consistent(env, obs["graph"])
    rng = np.random.RandomState(0)
    for _ in range(env.n_operations + 2):
        valid = np.flatnonzero(obs["action_mask"])
        if valid.size == 0:
            break
        obs, _, term, trunc, _ = env.step(int(valid[rng.randint(0, valid.size)]))
        _assert_graph_consistent(env, obs["graph"])
        if term or trunc:
            break
    env.close()


def test_graph_features_and_attrs_track_full_scale_prefix():
    env = FJSPEnv(n_machines=25, n_jobs=15, avg_operations_per_job=8, seed=1, device="cpu")
    obs, _ = env.reset(seed=1)
    _assert_graph_consistent(env, obs["graph"])
    rng = np.random.RandomState(1)
    for _ in range(8):
        valid = np.flatnonzero(obs["action_mask"])
        if valid.size == 0:
            break
        obs, _, term, trunc, _ = env.step(int(valid[rng.randint(0, valid.size)]))
        _assert_graph_consistent(env, obs["graph"])
        if term or trunc:
            break
    env.close()


def test_schedule_writes_processing_edge_attr_and_machine_features():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.25
    env.schedule_operation(0, 0)
    graph = env._get_obs()["graph"]
    assert float(graph["operation"].x[0, OP_SCHEDULED].item()) == pytest.approx(1.0)
    assert (0, 0) in _edge_pairs(graph, PROC)
    assert float(graph[PROC].edge_attr.reshape(-1)[0].item()) == pytest.approx(1.25)
    assert float(graph["machine"].x[0, 0].item()) == pytest.approx(1.0)
    assert float(graph["machine"].x[0, 1].item()) == pytest.approx(
        float(graph["operation"].x[0, OP_REMAINING].item())
    )
    env.close()


def test_obs_compatible_edges_drop_finished_ops_attrs_stay_aligned():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[0, 0] = 0.7
    env.efficiency_modifiers[1, 1] = 1.3
    pairs = env.eligibility_matrix.nonzero(as_tuple=False)
    env.state[COMPAT].edge_index = pairs.t().contiguous()
    env.state[COMPAT].edge_attr = env.efficiency_modifiers[
        pairs[:, 0], pairs[:, 1]
    ].unsqueeze(-1).to(torch.float32)
    env.dependency_types = {}
    env.state[PRECEDE].edge_index = torch.empty((2, 0), dtype=torch.long, device=env.device)
    env.state["operation"].x[:, OP_DURATION] = 1.0
    env.state["operation"].x[:, OP_REMAINING] = 1.0
    env.state["operation"].x[:, OP_SEQ_DEPS : OP_CROSS_DEPS + 1] = 0
    env.schedule_operation(0, 0)
    env._advance_time_tick()
    assert float(env.state["operation"].x[0, OP_FINISHED].item()) == pytest.approx(1.0)
    graph = env._get_obs()["graph"]
    assert 0 not in [op for op, _m in _edge_pairs(graph, COMPAT)]
    assert 0 in [op for op, _m in _edge_pairs(env.state, COMPAT)]
    _assert_graph_consistent(env, graph)
    env.close()


def test_processed_by_edges_removed_when_operation_completes():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {}
    env.state[PRECEDE].edge_index = torch.empty((2, 0), dtype=torch.long, device=env.device)
    env.state["operation"].x[:, :] = 0
    env.state["operation"].x[:, OP_DURATION] = torch.tensor([1.0, 5.0], device=env.device)
    env.state["operation"].x[:, OP_REMAINING] = torch.tensor([1.0, 5.0], device=env.device)
    env.state["machine"].x[:] = 0
    env.schedule_operation(0, 0)
    env.schedule_operation(1, 1)
    assert set(_edge_pairs(env.state, PROCESSED_BY)) == {(0, 0), (1, 1)}
    env._advance_time_tick()
    assert float(env.state["operation"].x[0, OP_FINISHED].item()) == pytest.approx(1.0)
    assert set(_edge_pairs(env.state, PROCESSED_BY)) == {(1, 1)}
    env.close()


def test_processed_by_edges_removed_when_operation_completes():
    env = FJSPEnv(n_machines=2, n_jobs=1, avg_operations_per_job=2, seed=0, device="cpu")
    env.reset(seed=0)
    env.eligibility_matrix[:, :] = True
    env.efficiency_modifiers[:, :] = 1.0
    env.dependency_types = {}
    env.state[PRECEDE].edge_index = torch.empty((2, 0), dtype=torch.long, device=env.device)
    env.state["operation"].x[:, :] = 0
    env.state["operation"].x[:, OP_DURATION] = torch.tensor([1.0, 5.0], device=env.device)
    env.state["operation"].x[:, OP_REMAINING] = torch.tensor([1.0, 5.0], device=env.device)
    env.state["machine"].x[:] = 0
    env.schedule_operation(0, 0)
    env.schedule_operation(1, 1)
    assert set(_edge_pairs(env.state, PROCESSED_BY)) == {(0, 0), (1, 1)}
    env._advance_time_tick()
    assert float(env.state["operation"].x[0, OP_FINISHED].item()) == pytest.approx(1.0)
    assert set(_edge_pairs(env.state, PROCESSED_BY)) == {(1, 1)}
    env.close()


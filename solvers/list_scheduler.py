"""Precedence-aware insertion list scheduling for a fixed FJSP assignment."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from solvers.milp import FJSPInstance, MilpResult

# LRPT first so Cmax ties are credited to the stronger rule.
LIST_RULES: Tuple[str, ...] = ("LRPT", "CP")
_INSERT_TOL = 1e-9


def earliest_insertion_start(
    intervals: Sequence[Tuple[float, float]],
    ready: float,
    proc: float,
    *,
    tol: float = _INSERT_TOL,
) -> float:
    """Earliest start >= ``ready`` that does not overlap existing intervals.

    ``intervals`` are ``(start, end)`` processing windows already on the machine.
    """
    t = float(ready)
    if proc < 0.0:
        raise ValueError(f"processing time must be >= 0, got {proc}")
    if proc <= tol:
        return t
    ordered = sorted(((float(s), float(e)) for s, e in intervals), key=lambda se: (se[0], se[1]))
    for start, end in ordered:
        if t + proc <= start + tol:
            return t
        if end > t:
            t = float(end)
    return t


def _pred_succ(n_ops: int, precedences: Sequence[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]]]:
    preds: List[List[int]] = [[] for _ in range(n_ops)]
    succs: List[List[int]] = [[] for _ in range(n_ops)]
    seen = set()
    for pred, succ in precedences:
        pair = (int(pred), int(succ))
        if pair in seen:
            continue
        seen.add(pair)
        if not (0 <= pair[0] < n_ops and 0 <= pair[1] < n_ops):
            raise ValueError(f"precedence out of range: {pair}")
        preds[pair[1]].append(pair[0])
        succs[pair[0]].append(pair[1])
    return preds, succs


def _longest_paths(proc: Sequence[float], succs: Sequence[Sequence[int]]) -> List[float]:
    """Critical-path remaining work including self (DAG longest path)."""
    n = len(proc)
    cache = [None] * n  # type: List[Optional[float]]
    visiting = [False] * n

    def dfs(i: int) -> float:
        if visiting[i]:
            raise ValueError("precedence graph has a cycle")
        if cache[i] is not None:
            return float(cache[i])
        visiting[i] = True
        tail = 0.0
        for s in succs[i]:
            cand = dfs(int(s))
            if cand > tail:
                tail = cand
        visiting[i] = False
        cache[i] = float(proc[i]) + tail
        return cache[i]

    return [dfs(i) for i in range(n)]


def _descendant_work(proc: Sequence[float], succs: Sequence[Sequence[int]]) -> List[float]:
    """Self plus all descendants' processing times."""
    n = len(proc)
    cache: List[Optional[set[int]]] = [None] * n
    visiting = [False] * n

    def descendants(i: int) -> set[int]:
        if visiting[i]:
            raise ValueError("precedence graph has a cycle")
        cached = cache[i]
        if cached is not None:
            return cached
        visiting[i] = True
        out: set[int] = set()
        for s in succs[i]:
            s = int(s)
            out.add(s)
            out |= descendants(s)
        visiting[i] = False
        cache[i] = out
        return out

    return [float(proc[i]) + sum(float(proc[j]) for j in descendants(i)) for i in range(n)]


def list_schedule(
    instance: FJSPInstance,
    assignment: Sequence[Tuple[int, int]],
    rule: str = "LRPT",
) -> MilpResult:
    """Build a feasible classic FJSP schedule by insertion list scheduling.

    Only operations whose predecessors are already scheduled are candidates.
    The chosen operation is inserted at the earliest feasible gap on its
    assigned machine (not necessarily appended).
    """
    key = str(rule).upper()
    if key not in LIST_RULES:
        raise ValueError(f"unknown list rule {rule!r}; choose from {LIST_RULES}")

    n_ops = instance.n_operations
    n_machines = instance.n_machines
    proc_mat = np.asarray(instance.proc_times, dtype=np.float64)
    elig = np.asarray(instance.eligibility, dtype=bool)
    incomplete = MilpResult(status="Incomplete", makespan=float("inf"), solve_time_s=0.0)

    assigned: Dict[int, int] = {}
    for op, machine in assignment:
        op_i, m_i = int(op), int(machine)
        if op_i in assigned:
            return incomplete
        assigned[op_i] = m_i
    if len(assigned) != n_ops or set(assigned) != set(range(n_ops)):
        return incomplete

    proc = np.zeros(n_ops, dtype=np.float64)
    for i, m in assigned.items():
        if not (0 <= m < n_machines) or not bool(elig[i, m]):
            return incomplete
        proc[i] = float(proc_mat[i, m])

    preds, succs = _pred_succ(n_ops, instance.precedences)
    remaining_pred = [len(preds[i]) for i in range(n_ops)]
    priority = _longest_paths(proc, succs) if key == "CP" else _descendant_work(proc, succs)

    starts = np.full(n_ops, np.nan, dtype=np.float64)
    completions = np.zeros(n_ops, dtype=np.float64)
    scheduled = [False] * n_ops
    machine_intervals: List[List[Tuple[float, float]]] = [[] for _ in range(n_machines)]

    for _ in range(n_ops):
        ready = [i for i in range(n_ops) if not scheduled[i] and remaining_pred[i] == 0]
        if not ready:
            return incomplete

        best_op = min(ready, key=lambda i: (-float(priority[i]), i))
        pred_ready = 0.0
        for pred in preds[best_op]:
            if completions[pred] > pred_ready:
                pred_ready = float(completions[pred])
        m = assigned[best_op]
        start = earliest_insertion_start(machine_intervals[m], pred_ready, float(proc[best_op]))
        end = start + float(proc[best_op])
        starts[best_op] = start
        completions[best_op] = end
        machine_intervals[m].append((start, end))
        scheduled[best_op] = True
        for succ in succs[best_op]:
            remaining_pred[succ] -= 1

    return MilpResult(
        status="Feasible",
        makespan=float(completions.max()) if n_ops else 0.0,
        solve_time_s=0.0,
        assignment=tuple(sorted(assigned.items())),
        starts=tuple(float(s) for s in starts),
    )


def schedule_best_rule(
    instance: FJSPInstance,
    assignment: Sequence[Tuple[int, int]],
    rules: Sequence[str] = LIST_RULES,
) -> Tuple[MilpResult, str]:
    """List-schedule ``assignment`` under each rule; keep the lowest Cmax."""
    if not rules:
        raise ValueError("rules must be non-empty")
    best: Optional[MilpResult] = None
    best_rule = str(rules[0]).upper()
    for rule in rules:
        scheduled = list_schedule(instance, assignment, rule)
        if best is None:
            best = scheduled
            best_rule = str(rule).upper()
            continue
        if scheduled.makespan < best.makespan - 1e-12:
            best = scheduled
            best_rule = str(rule).upper()
    assert best is not None
    return best, best_rule


__all__ = [
    "LIST_RULES",
    "earliest_insertion_start",
    "list_schedule",
    "schedule_best_rule",
]

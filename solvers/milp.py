"""Exact MILP makespan solver for FJSP instances (PuLP + CBC)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pulp

from envs.fjsp_env import OP_DURATION, FJSPEnv


@dataclass(frozen=True)
class FJSPInstance:
    """Static FJSP data extracted from a reset ``FJSPEnv``."""

    n_operations: int
    n_machines: int
    proc_times: np.ndarray  # (n_ops, n_machines); unused cells ignored
    eligibility: np.ndarray  # bool (n_ops, n_machines)
    precedences: Tuple[Tuple[int, int], ...]  # (pred, succ)
    time_penalty: float = -0.1
    time_step: float = 1.0


@dataclass(frozen=True)
class MilpResult:
    """Outcome of a single MILP solve."""

    status: str
    makespan: float
    solve_time_s: float
    assignment: Optional[Tuple[Tuple[int, int], ...]] = None  # (op, machine)


def extract_fjsp_instance(env: FJSPEnv) -> FJSPInstance:
    """Build an ``FJSPInstance`` from a live, reset ``FJSPEnv``."""
    if env.state is None:
        raise ValueError("env.state is None; reset the environment first")

    n_ops = int(env.n_operations)
    n_machines = int(env.n_machines)
    durations = env.state["operation"].x[:, OP_DURATION].detach().cpu().numpy().astype(np.float64)
    efficiency = env.efficiency_modifiers.detach().cpu().numpy().astype(np.float64)
    eligibility = env.eligibility_matrix.detach().cpu().numpy().astype(bool)
    proc_times = durations[:, None] * efficiency

    precedences: List[Tuple[int, int]] = []
    dep = env.state["operation", "precede", "operation"].edge_index
    if dep is not None and dep.numel() > 0:
        src = dep[0].detach().cpu().numpy()
        dst = dep[1].detach().cpu().numpy()
        for a, b in zip(src.tolist(), dst.tolist()):
            precedences.append((int(a), int(b)))

    return FJSPInstance(
        n_operations=n_ops,
        n_machines=n_machines,
        proc_times=proc_times,
        eligibility=eligibility,
        precedences=tuple(precedences),
        time_penalty=float(env.time_penalty),
        time_step=float(env.time_step),
    )


def solve_makespan(
    instance: FJSPInstance,
    *,
    time_limit: Optional[float] = None,
    msg: bool = False,
) -> MilpResult:
    """Minimize makespan with a classic FJSP disjunctive MILP (CBC).

    Args:
        instance: Static instance data.
        time_limit: Optional CBC wall-clock limit in seconds.
        msg: If True, print CBC log output.

    Returns:
        ``MilpResult``. ``makespan`` is finite only when status is Optimal.
    """
    n_ops = instance.n_operations
    n_machines = instance.n_machines
    elig = np.asarray(instance.eligibility, dtype=bool)
    proc = np.asarray(instance.proc_times, dtype=np.float64)

    if n_ops <= 0 or n_machines <= 0:
        raise ValueError("instance must have positive n_operations and n_machines")
    if elig.shape != (n_ops, n_machines) or proc.shape != (n_ops, n_machines):
        raise ValueError("eligibility/proc_times shape mismatch")
    if not np.all(elig.any(axis=1)):
        raise ValueError("every operation needs at least one eligible machine")

    ops = range(n_ops)
    machines = range(n_machines)
    # Worst-case horizon: sum of each op's slowest eligible processing time.
    big_m = float(sum(float(np.max(proc[i, elig[i]])) for i in ops))
    if not np.isfinite(big_m) or big_m <= 0.0:
        big_m = 1.0

    prob = pulp.LpProblem("fjsp_makespan", pulp.LpMinimize)
    x = {
        (i, m): pulp.LpVariable(f"x_{i}_{m}", cat="Binary")
        for i in ops
        for m in machines
        if bool(elig[i, m])
    }
    start = pulp.LpVariable.dicts("S", list(ops), lowBound=0.0, cat="Continuous")
    cmax = pulp.LpVariable("Cmax", lowBound=0.0, cat="Continuous")
    prob += cmax

    for i in ops:
        eligible_m = [m for m in machines if (i, m) in x]
        prob += pulp.lpSum(x[i, m] for m in eligible_m) == 1, f"assign_{i}"
        p_i = pulp.lpSum(float(proc[i, m]) * x[i, m] for m in eligible_m)
        prob += cmax >= start[i] + p_i, f"cmax_{i}"

    for pred, succ in instance.precedences:
        if not (0 <= pred < n_ops and 0 <= succ < n_ops):
            raise ValueError(f"precedence out of range: {(pred, succ)}")
        p_pred = pulp.lpSum(
            float(proc[pred, m]) * x[pred, m]
            for m in machines
            if (pred, m) in x
        )
        prob += start[succ] >= start[pred] + p_pred, f"prec_{pred}_{succ}"

    for m in machines:
        eligible_ops = [i for i in ops if (i, m) in x]
        for a_idx in range(len(eligible_ops)):
            for b_idx in range(a_idx + 1, len(eligible_ops)):
                i = eligible_ops[a_idx]
                j = eligible_ops[b_idx]
                y = pulp.LpVariable(f"y_{i}_{j}_{m}", cat="Binary")
                # Active only when both ops are assigned to machine m.
                prob += (
                    start[j]
                    >= start[i]
                    + float(proc[i, m])
                    - big_m * (1 - y)
                    - big_m * (1 - x[i, m])
                    - big_m * (1 - x[j, m])
                ), f"disj_{i}_{j}_{m}_a"
                prob += (
                    start[i]
                    >= start[j]
                    + float(proc[j, m])
                    - big_m * y
                    - big_m * (1 - x[i, m])
                    - big_m * (1 - x[j, m])
                ), f"disj_{i}_{j}_{m}_b"

    solver_kwargs = {"msg": bool(msg)}
    if time_limit is not None:
        if float(time_limit) <= 0.0:
            raise ValueError(f"time_limit must be positive, got {time_limit}")
        solver_kwargs["timeLimit"] = float(time_limit)

    solver = pulp.PULP_CBC_CMD(**solver_kwargs)
    t0 = time.perf_counter()
    status_code = prob.solve(solver)
    solve_time_s = time.perf_counter() - t0
    status = pulp.LpStatus.get(status_code, str(status_code))

    if status != "Optimal":
        return MilpResult(
            status=status,
            makespan=float("inf"),
            solve_time_s=float(solve_time_s),
            assignment=None,
        )

    makespan = float(pulp.value(cmax))
    assignment = tuple(
        sorted(
            (i, m)
            for (i, m), var in x.items()
            if var.varValue is not None and float(var.varValue) > 0.5
        )
    )
    return MilpResult(
        status=status,
        makespan=makespan,
        solve_time_s=float(solve_time_s),
        assignment=assignment,
    )


def milp_episode_metrics(instance: FJSPInstance, result: MilpResult) -> dict:
    """Map a MILP solve into EvalResult-compatible episode fields."""
    if result.status == "Optimal" and np.isfinite(result.makespan):
        reward = float(instance.time_penalty) * (
            float(result.makespan) / float(instance.time_step)
        )
        return {
            "r": reward,
            "l": float(instance.n_operations),
            "makespan": float(result.makespan),
            "success": True,
            "truncated": False,
        }
    return {
        "r": 0.0,
        "l": float(instance.n_operations),
        "makespan": float("inf"),
        "success": False,
        "truncated": result.status != "Optimal",
    }


__all__ = [
    "FJSPInstance",
    "MilpResult",
    "extract_fjsp_instance",
    "milp_episode_metrics",
    "solve_makespan",
]

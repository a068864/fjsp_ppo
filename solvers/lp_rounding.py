"""LP relaxation + rounding baseline for FJSP (PuLP + CBC).

This is an assignment LP with a valid machine-load capacity relaxation,
followed by seeded rounding and insertion list scheduling. 
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pulp

from solvers.list_scheduler import LIST_RULES, list_schedule
from solvers.milp import FJSPInstance, MilpResult

LP_TOL = 1e-6


@dataclass(frozen=True)
class LpRelaxationResult:
    """Outcome of the continuous assignment LP (no rounding)."""

    status: str
    lp_lower_bound: float
    solve_time_s: float
    fractional_assignment: Dict[int, Dict[int, float]]
    starts: Optional[Tuple[float, ...]] = None
    max_constraint_violation: float = float("inf")
    assignment_sum_violation: float = float("inf")
    n_fractional: int = 0


@dataclass(frozen=True)
class LpRoundingResult:
    """LP lower bound plus a feasible rounded/list-scheduled schedule."""

    status: str
    lp_lower_bound: float
    makespan: float
    fractional_assignment: Dict[int, Dict[int, float]]
    assignment: Optional[Tuple[Tuple[int, int], ...]] = None
    starts: Optional[Tuple[float, ...]] = None
    best_rule: Optional[str] = None
    rounding_trials: int = 0
    n_fractional: int = 0
    max_constraint_violation: float = float("inf")
    assignment_sum_violation: float = float("inf")
    solve_time_s: float = 0.0
    rounding_time_s: float = 0.0
    lp_status: str = ""
    rule_makespans: Tuple[Tuple[str, float], ...] = ()

    def as_schedule(self) -> MilpResult:
        """Feasible classic-FJSP schedule, or Incomplete if rounding failed."""
        if (
            self.status in {"Optimal", "Feasible"}
            and self.assignment is not None
            and self.starts is not None
            and np.isfinite(self.makespan)
        ):
            return MilpResult(
                status="Feasible",
                makespan=float(self.makespan),
                solve_time_s=float(self.solve_time_s + self.rounding_time_s),
                assignment=self.assignment,
                starts=self.starts,
            )
        return MilpResult(
            status="Incomplete",
            makespan=float("inf"),
            solve_time_s=float(self.solve_time_s + self.rounding_time_s),
        )


def _failed_relaxation(status: str, solve_time_s: float) -> LpRelaxationResult:
    return LpRelaxationResult(
        status=status,
        lp_lower_bound=float("inf"),
        solve_time_s=float(solve_time_s),
        fractional_assignment={},
        starts=None,
        max_constraint_violation=float("inf"),
        assignment_sum_violation=float("inf"),
        n_fractional=0,
    )


def _failed_rounding(
    lp: LpRelaxationResult,
    *,
    rounding_trials: int,
    rounding_time_s: float = 0.0,
) -> LpRoundingResult:
    return LpRoundingResult(
        status=lp.status,
        lp_lower_bound=float(lp.lp_lower_bound),
        makespan=float("inf"),
        fractional_assignment=lp.fractional_assignment,
        assignment=None,
        starts=None,
        best_rule=None,
        rounding_trials=int(rounding_trials),
        n_fractional=int(lp.n_fractional),
        max_constraint_violation=float(lp.max_constraint_violation),
        assignment_sum_violation=float(lp.assignment_sum_violation),
        solve_time_s=float(lp.solve_time_s),
        rounding_time_s=float(rounding_time_s),
        lp_status=lp.status,
        rule_makespans=(),
    )


def solve_lp_relaxation(
    instance: FJSPInstance,
    *,
    time_limit: Optional[float] = None,
    msg: bool = False,
    tol: float = LP_TOL,
) -> LpRelaxationResult:
    """Minimize Cmax over a valid LP relaxation of FJSP.

    Integer ``x[i,m] ∈ {0,1}`` is relaxed to ``[0,1]``. Disjunctive machine
    sequencing is replaced by the valid load inequalities
    ``Σ_i p[i,m] x[i,m] ≤ Cmax``. Every feasible integral schedule remains
    feasible, so the objective is a lower bound (up to solver tolerance).
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

    prob = pulp.LpProblem("fjsp_lp_relaxation", pulp.LpMinimize)
    x = {
        (i, m): pulp.LpVariable(f"x_{i}_{m}", lowBound=0.0, upBound=1.0, cat="Continuous")
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

    seen_prec: set[Tuple[int, int]] = set()
    for pred, succ in instance.precedences:
        if not (0 <= pred < n_ops and 0 <= succ < n_ops):
            raise ValueError(f"precedence out of range: {(pred, succ)}")
        if (pred, succ) in seen_prec:
            continue
        seen_prec.add((pred, succ))
        p_pred = pulp.lpSum(
            float(proc[pred, m]) * x[pred, m] for m in machines if (pred, m) in x
        )
        prob += start[succ] >= start[pred] + p_pred, f"prec_{pred}_{succ}"

    for m in machines:
        load = pulp.lpSum(float(proc[i, m]) * x[i, m] for i in ops if (i, m) in x)
        prob += load <= cmax, f"load_{m}"

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
        return _failed_relaxation(status, solve_time_s)

    x_val = {
        (i, m): float(pulp.value(var) or 0.0) for (i, m), var in x.items()
    }
    starts = tuple(float(pulp.value(start[i]) or 0.0) for i in ops)
    lb = float(pulp.value(cmax) or 0.0)
    fractional: Dict[int, Dict[int, float]] = {
        i: {m: x_val[i, m] for m in machines if (i, m) in x_val} for i in ops
    }
    assignment_sum_violation, max_violation, n_fractional = _lp_diagnostics(
        instance, fractional, starts, lb, tol=tol
    )
    return LpRelaxationResult(
        status=status,
        lp_lower_bound=lb,
        solve_time_s=float(solve_time_s),
        fractional_assignment=fractional,
        starts=starts,
        max_constraint_violation=max_violation,
        assignment_sum_violation=assignment_sum_violation,
        n_fractional=n_fractional,
    )


def _lp_diagnostics(
    instance: FJSPInstance,
    fractional: Mapping[int, Mapping[int, float]],
    starts: Sequence[float],
    cmax: float,
    *,
    tol: float,
) -> Tuple[float, float, int]:
    n_ops = instance.n_operations
    n_machines = instance.n_machines
    proc = np.asarray(instance.proc_times, dtype=np.float64)
    elig = np.asarray(instance.eligibility, dtype=bool)

    assign_viol = 0.0
    max_viol = 0.0
    n_fractional = 0

    def _note(v: float) -> None:
        nonlocal max_viol
        if v > max_viol:
            max_viol = v

    p_frac = np.zeros(n_ops, dtype=np.float64)
    load = np.zeros(n_machines, dtype=np.float64)
    for i in range(n_ops):
        row = fractional.get(i, {})
        total = 0.0
        n_pos = 0
        for m in range(n_machines):
            val = float(row.get(m, 0.0))
            if not bool(elig[i, m]):
                _note(abs(val))
                continue
            _note(max(0.0, -val, val - 1.0))
            total += val
            p_frac[i] += float(proc[i, m]) * val
            load[m] += float(proc[i, m]) * val
            if val > tol:
                n_pos += 1
        assign_i = abs(total - 1.0)
        if assign_i > assign_viol:
            assign_viol = assign_i
        _note(assign_i)
        if n_pos > 1 or (n_pos == 1 and abs(total - 1.0) > tol):
            n_fractional += 1
        _note(max(0.0, -float(starts[i])))
        _note(max(0.0, float(starts[i]) + float(p_frac[i]) - float(cmax)))

    for pred, succ in instance.precedences:
        gap = float(starts[pred]) + float(p_frac[pred]) - float(starts[succ])
        _note(max(0.0, gap))

    for m in range(n_machines):
        _note(max(0.0, float(load[m]) - float(cmax)))
    _note(max(0.0, -float(cmax)))
    return float(assign_viol), float(max_viol), int(n_fractional)


def largest_fraction_assignment(
    fractional_assignment: Mapping[int, Mapping[int, float]],
    n_operations: int,
) -> Tuple[Tuple[int, int], ...]:
    """Assign each op to the eligible machine with largest LP fraction.

    Ties break by lowest machine index. Ineligible machines must not appear
    in ``fractional_assignment``.
    """
    assigned: List[Tuple[int, int]] = []
    for i in range(n_operations):
        row = fractional_assignment.get(i, {})
        if not row:
            raise ValueError(f"operation {i} has no fractional assignment")
        best_m = min(row.keys(), key=lambda m: (-float(row[m]), int(m)))
        assigned.append((i, int(best_m)))
    return tuple(assigned)


def sample_rounded_assignment(
    fractional_assignment: Mapping[int, Mapping[int, float]],
    n_operations: int,
    rng: np.random.Generator,
    *,
    tol: float = LP_TOL,
) -> Tuple[Tuple[int, int], ...]:
    """Sample one machine per op with probability proportional to ``x[i,m]``."""
    assigned: List[Tuple[int, int]] = []
    for i in range(n_operations):
        row = fractional_assignment.get(i, {})
        if not row:
            raise ValueError(f"operation {i} has no fractional assignment")
        machines = sorted(int(m) for m in row)
        probs = np.array([max(float(row[m]), 0.0) for m in machines], dtype=np.float64)
        total = float(probs.sum())
        if total <= tol:
            best_m = min(machines, key=lambda m: (-float(row[m]), m))
            assigned.append((i, best_m))
            continue
        probs = probs / total
        assigned.append((i, int(rng.choice(machines, p=probs))))
    return tuple(assigned)


def generate_rounded_assignments(
    fractional_assignment: Mapping[int, Mapping[int, float]],
    n_operations: int,
    *,
    n_trials: int,
    seed: int,
    tol: float = LP_TOL,
) -> List[Tuple[Tuple[int, int], ...]]:
    """Largest-fraction assignment plus up to ``n_trials-1`` randomized draws."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    first = largest_fraction_assignment(fractional_assignment, n_operations)
    candidates = [first]
    seen = {first}
    if n_trials == 1:
        return candidates
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_trials) - 1):
        draw = sample_rounded_assignment(
            fractional_assignment, n_operations, rng, tol=tol
        )
        if draw not in seen:
            seen.add(draw)
            candidates.append(draw)
    return candidates


def solve_lp_rounding(
    instance: FJSPInstance,
    *,
    seed: int = 42,
    rounding_trials: int = 20,
    time_limit: Optional[float] = None,
    msg: bool = False,
    rules: Sequence[str] = LIST_RULES,
    tol: float = LP_TOL,
) -> LpRoundingResult:
    """Solve the LP once, round several assignments, list-schedule, keep best.

    Trial 0 is deterministic largest-fraction rounding. Extra trials sample
    machines from the fractional assignment using ``seed``.
    """
    lp = solve_lp_relaxation(instance, time_limit=time_limit, msg=msg, tol=tol)
    if lp.status != "Optimal":
        return _failed_rounding(lp, rounding_trials=rounding_trials)

    t0 = time.perf_counter()
    candidates = generate_rounded_assignments(
        lp.fractional_assignment,
        instance.n_operations,
        n_trials=int(rounding_trials),
        seed=int(seed),
        tol=tol,
    )
    best_sched: Optional[MilpResult] = None
    best_rule: Optional[str] = None
    rule_best = {str(rule).upper(): float("inf") for rule in rules}
    for assignment in candidates:
        for rule in rules:
            key = str(rule).upper()
            scheduled = list_schedule(instance, assignment, rule=key)
            if scheduled.status != "Feasible" or not np.isfinite(scheduled.makespan):
                continue
            if scheduled.makespan < rule_best[key] - 1e-12:
                rule_best[key] = float(scheduled.makespan)
            if best_sched is None or scheduled.makespan < best_sched.makespan - 1e-12:
                best_sched = scheduled
                best_rule = key
    rounding_time_s = time.perf_counter() - t0
    if best_sched is None or best_sched.status != "Feasible":
        return _failed_rounding(
            lp, rounding_trials=rounding_trials, rounding_time_s=rounding_time_s
        )

    return LpRoundingResult(
        status="Feasible",
        lp_lower_bound=float(lp.lp_lower_bound),
        makespan=float(best_sched.makespan),
        fractional_assignment=lp.fractional_assignment,
        assignment=best_sched.assignment,
        starts=best_sched.starts,
        best_rule=best_rule,
        rounding_trials=int(rounding_trials),
        n_fractional=int(lp.n_fractional),
        max_constraint_violation=float(lp.max_constraint_violation),
        assignment_sum_violation=float(lp.assignment_sum_violation),
        solve_time_s=float(lp.solve_time_s),
        rounding_time_s=float(rounding_time_s),
        lp_status=lp.status,
        rule_makespans=tuple(
            (name, float(cmax))
            for name, cmax in rule_best.items()
            if np.isfinite(cmax)
        ),
    )


__all__ = [
    "LP_TOL",
    "LpRelaxationResult",
    "LpRoundingResult",
    "generate_rounded_assignments",
    "largest_fraction_assignment",
    "sample_rounded_assignment",
    "solve_lp_relaxation",
    "solve_lp_rounding",
]

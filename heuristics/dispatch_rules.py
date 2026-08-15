"""Classic FJSP dispatching rules over valid action-mask entries."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from envs.fjsp_env import OP_DURATION, OP_FINISHED, OP_SCHEDULED, FJSPEnv, MACH_QUEUE, MACH_WORKLOAD
from utils import unflatten_action

# (minimize?, scorer name) — scorer returns a float per flat action.
RuleSpec = Tuple[bool, str]

RULES: Dict[str, RuleSpec] = {
    "SPT": (True, "proc"),
    "LPT": (False, "proc"),
    "MWKR": (False, "work"),
    "LWKR": (True, "work"),
    "MOR": (False, "ops"),
    "LOR": (True, "ops"),
    "FIFO": (True, "fifo"),
    "MFE": (False, "flex"),
    "LFE": (True, "flex"),
    "SQ": (True, "queue"),
    "LWQM": (True, "workload"),
    "ECT": (True, "ect"),
}


def _effective_proc(env: FJSPEnv, machine: int, op: int) -> float:
    duration = float(env.state["operation"].x[op, OP_DURATION].item())
    eff = float(env.efficiency_modifiers[op, machine].item())
    return duration * eff


def _job_remaining(env: FJSPEnv, op: int) -> Tuple[float, int]:
    """Return (base-duration sum, op count) of unscheduled ops in ``op``'s job."""
    op_x = env.state["operation"].x
    for seq in env.job_sequences:
        if op not in seq:
            continue
        work = 0.0
        count = 0
        for other in seq:
            if float(op_x[other, OP_SCHEDULED]) > 0.5 or float(op_x[other, OP_FINISHED]) > 0.5:
                continue
            work += float(op_x[other, OP_DURATION].item())
            count += 1
        return work, count
    raise ValueError(f"operation {op} not found in job_sequences")


def _score_action(env: FJSPEnv, action: int, kind: str) -> float:
    machine, op = unflatten_action(action, env.n_operations)
    if kind == "proc":
        return _effective_proc(env, machine, op)
    if kind == "work":
        return _job_remaining(env, op)[0]
    if kind == "ops":
        return float(_job_remaining(env, op)[1])
    if kind == "fifo":
        return float(op)
    if kind == "flex":
        return float(env.eligibility_matrix[op].sum().item())
    if kind == "queue":
        return float(env.state["machine"].x[machine, MACH_QUEUE].item())
    if kind == "workload":
        return float(env.state["machine"].x[machine, MACH_WORKLOAD].item())
    if kind == "ect":
        return float(env.state["machine"].x[machine, MACH_WORKLOAD].item()) + _effective_proc(env, machine, op)
    raise ValueError(f"unknown score kind: {kind}")


def select_heuristic_action(env: FJSPEnv, rule: str) -> int:
    """Pick a valid flat action by dispatching rule.

    Ties break by lowest flat action index (deterministic).

    Args:
        env: Live ``FJSPEnv`` (unwrapped).
        rule: Rule name from ``RULES`` (case-insensitive).

    Returns:
        Flat action index ``machine_id * n_operations + operation_id``.

    Raises:
        ValueError: Unknown rule or empty action mask.
    """
    key = rule.upper()
    if key not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {sorted(RULES)}")

    minimize, kind = RULES[key]
    mask = np.asarray(env.get_action_mask(), dtype=np.float32).reshape(-1)
    valid = np.flatnonzero(mask > 0.5)
    if valid.size == 0:
        raise ValueError("empty action mask")

    scores = np.array([_score_action(env, int(action), kind) for action in valid])
    best_idx = int(np.argmin(scores) if minimize else np.argmax(scores))
    return int(valid[best_idx])


__all__ = [
    "RULES",
    "select_heuristic_action",
]

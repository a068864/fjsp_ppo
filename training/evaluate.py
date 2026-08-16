"""Shared evaluation helpers for FJSP PPO policies."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from heuristics.dispatch_rules import select_heuristic_action
from solvers.lp_rounding import LpRoundingResult, solve_lp_rounding
from solvers.milp import (
    decode_assignment_schedule,
    extract_fjsp_instance,
    milp_episode_metrics,
    solve_makespan,
)
from utils import get_logger, safe_mean

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Aggregate metrics from a deterministic (or stochastic) evaluation run."""

    mean_makespan: float
    std_makespan: float
    mean_ep_length: float
    std_ep_length: float
    success_rate: float
    mean_inference_time_s: float
    n_episodes: int
    n_success: int = 0
    n_failure: int = 0
    n_timeout: int = 0

    def format_summary(self) -> str:
        """Human-readable multi-line summary."""
        return "\n".join(
            [
                f"Episodes          : {self.n_episodes}",
                f"Successful-episode makespan : {self.mean_makespan:.4f} ± {self.std_makespan:.4f}",
                f"Episode length    : {self.mean_ep_length:.2f} ± {self.std_ep_length:.2f}",
                f"Success rate      : {self.success_rate:.2%}",
                f"Success/Failure/Timeout : {self.n_success}/{self.n_failure}/{self.n_timeout}",
                f"Inference time    : {self.mean_inference_time_s * 1000:.3f} ms/step",
            ]
        )


def _episode_from_info(
    info: Dict[str, Any],
    fallback_length: int,
) -> Dict[str, Any]:
    """Build episode stats from info, with numeric fallbacks."""

    def _filled(
        ep: Dict[str, Any],
        *,
        makespan: float,
        success: bool,
    ) -> Dict[str, Any]:
        out = dict(ep)
        out.setdefault("l", fallback_length)
        out.setdefault("makespan", makespan)
        out.setdefault("success", success)
        return out

    if "episode" in info and isinstance(info["episode"], dict):
        return _filled(info["episode"], makespan=float("inf"), success=False)

    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        makespan = float(final_info.get("makespan", float("inf")))
        success = bool(final_info.get("success", False))
        if "episode" in final_info and isinstance(final_info["episode"], dict):
            return _filled(final_info["episode"], makespan=makespan, success=success)
        return _filled({}, makespan=makespan, success=success)

    return _filled(
        {},
        makespan=float(info.get("makespan", float("inf"))),
        success=bool(info.get("success", False)),
    )


def _require_dummy_vec(env: VecEnv, purpose: str) -> None:
    if not hasattr(env, "envs"):
        raise ValueError(
            f"{purpose} requires GraphDummyVecEnv (env.envs); "
            "use make_vec_env(..., use_subprocess=False)"
        )


def _action_to_pair(action: int, n_operations: int) -> Tuple[int, int]:
    """Decode flat ``machine * n_ops + op`` into ``(operation, machine)``."""
    a = int(action)
    return a % n_operations, a // n_operations


def _rollout_eval(
    env: VecEnv,
    n_episodes: int,
    choose_actions: Callable[[Any], Any],
) -> EvalResult:
    """Collect ``n_episodes`` constructive schedules via ``choose_actions(obs)``.

    The env is only a sequential decoder (ready mask). Reported makespan is
    classic FJSP Cmax of the inferred assignment sequence.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    _require_dummy_vec(env, "FJSP instance-schedule evaluation")

    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []

    n_envs = env.num_envs
    running_lengths = np.zeros(n_envs, dtype=np.int64)
    orders: List[List[Tuple[int, int]]] = [[] for _ in range(n_envs)]

    observations = env.reset()
    instances = [extract_fjsp_instance(env.envs[i].unwrapped) for i in range(n_envs)]

    while len(episode_makespans) < n_episodes:
        start = time.perf_counter()
        actions = choose_actions(observations)
        inference_times.append(time.perf_counter() - start)

        action_arr = np.asarray(actions).reshape(-1)
        for env_idx, action in enumerate(action_arr):
            orders[env_idx].append(
                _action_to_pair(int(action), instances[env_idx].n_operations)
            )

        observations, _rewards, dones, infos = env.step(actions)
        running_lengths += 1

        for env_idx, done in enumerate(np.asarray(dones, dtype=bool)):
            if not done:
                continue

            info = infos[env_idx] if env_idx < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep = _episode_from_info(info, fallback_length=int(running_lengths[env_idx]))
            env_success = bool(ep.get("success", False))
            decoded = decode_assignment_schedule(instances[env_idx], orders[env_idx])
            success = bool(env_success and decoded.status == "Feasible")

            episode_lengths.append(float(ep.get("l", running_lengths[env_idx])))
            episode_makespans.append(
                float(decoded.makespan) if success else float("inf")
            )
            episode_successes.append(1.0 if success else 0.0)
            timed_out = bool(ep.get("truncated", info.get("TimeLimit.truncated", False)))
            episode_timeouts.append(1.0 if timed_out and not success else 0.0)

            running_lengths[env_idx] = 0
            orders[env_idx] = []
            instances[env_idx] = extract_fjsp_instance(env.envs[env_idx].unwrapped)

            if len(episode_makespans) >= n_episodes:
                break

    return _aggregate_eval(
        episode_lengths[:n_episodes],
        episode_makespans[:n_episodes],
        episode_successes[:n_episodes],
        episode_timeouts[:n_episodes],
        inference_times,
    )


def evaluate_policy_fjsp(
    model: BaseAlgorithm,
    env: VecEnv,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> EvalResult:
    """Infer classic FJSP schedules with a trained policy.

    The env supplies the sequential action mask; reported makespan is the
    earliest-start Cmax of the inferred ``(op, machine)`` sequence.

    Args:
        model: Trained SB3 model (``PPO`` with ``GraphActorCriticPolicy``).
        env: In-process ``GraphDummyVecEnv`` with ``envs``.
        n_episodes: Number of completed episodes to collect.
        deterministic: If True, use greedy actions.

    Returns:
        ``EvalResult`` with classic makespan, length, success, and inference time.
    """

    def choose_actions(observations: Any) -> Any:
        mask = observations.get("action_mask") if isinstance(observations, dict) else None
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(1, -1)
            if np.any(np.sum(mask_arr > 0.5, axis=-1) == 0):
                raise ValueError("empty action mask during evaluation")
        actions, _states = model.predict(observations, deterministic=deterministic)
        return actions

    return _rollout_eval(env, n_episodes, choose_actions)


def evaluate_random_fjsp(
    env: VecEnv,
    n_episodes: int = 20,
    seed: int = 42,
) -> EvalResult:
    """Infer classic FJSP schedules with uniform random valid actions.

    Args:
        env: In-process ``GraphDummyVecEnv`` with ``envs``.
        n_episodes: Number of completed episodes to collect.
        seed: RNG seed for action sampling.

    Returns:
        ``EvalResult`` with classic makespan, length, success, and sampling time.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    rng = np.random.default_rng(seed)
    logger.info(
        "Starting random evaluation: n_episodes=%d seed=%d n_envs=%d",
        n_episodes,
        seed,
        env.num_envs,
    )

    def choose_actions(observations: Any) -> np.ndarray:
        return sample_masked_random_actions(observations["action_mask"], rng)

    result = _rollout_eval(env, n_episodes, choose_actions)
    logger.info("Random evaluation finished:\n%s", result.format_summary())
    return result


def evaluate_heuristic_fjsp(
    env: VecEnv,
    rule: str,
    n_episodes: int = 20,
) -> EvalResult:
    """Infer classic FJSP schedules with a dispatching rule.

    Requires an in-process DummyVecEnv exposing ``env.envs`` so rules can read
    live ``FJSPEnv`` state (``n_envs=1`` recommended). Reported makespan is
    classic instance Cmax of the inferred assignment sequence.

    Args:
        env: Vectorized evaluation environment with ``envs`` attribute.
        rule: Dispatching rule name (see ``heuristics.RULES``).
        n_episodes: Number of completed episodes to collect.

    Returns:
        ``EvalResult`` with classic makespan, length, success, and select time.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    _require_dummy_vec(env, "heuristic evaluation")
    logger.info(
        "Starting heuristic evaluation: rule=%s n_episodes=%d n_envs=%d",
        rule,
        n_episodes,
        env.num_envs,
    )

    def choose_actions(_observations: Any) -> np.ndarray:
        actions = np.zeros(env.num_envs, dtype=np.int64)
        for env_idx in range(env.num_envs):
            fjsp = env.envs[env_idx].unwrapped
            actions[env_idx] = select_heuristic_action(fjsp, rule)
        return actions

    result = _rollout_eval(env, n_episodes, choose_actions)
    logger.info("Heuristic evaluation finished (%s):\n%s", rule, result.format_summary())
    return result


def evaluate_milp_fjsp(
    env: VecEnv,
    n_episodes: int = 20,
    seed: int = 42,
    *,
    time_limit: Optional[float] = None,
) -> EvalResult:
    """Solve each held-out instance with an exact makespan MILP (PuLP+CBC).

    Requires ``GraphDummyVecEnv`` with ``env.envs`` (``n_envs=1``). Episode ``i``
    is loaded with seed ``seed + i`` to match the deterministic eval suite.
    Only Optimal solves count as success; non-optimal → makespan ``inf``.

    Args:
        env: In-process vectorized env with ``envs``.
        n_episodes: Number of instances to solve.
        seed: Base eval seed (episode ``i`` uses ``seed + i``).
        time_limit: Optional CBC wall-clock limit in seconds per instance.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    _require_dummy_vec(env, "MILP evaluation")
    if env.num_envs != 1:
        raise ValueError(f"MILP evaluation requires n_envs=1, got {env.num_envs}")

    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []

    logger.info(
        "Starting MILP evaluation: n_episodes=%d seed=%d time_limit=%s",
        n_episodes,
        seed,
        time_limit,
    )

    for ep_idx in range(n_episodes):
        env.seed(int(seed) + int(ep_idx))
        env.reset()
        fjsp = env.envs[0].unwrapped
        instance = extract_fjsp_instance(fjsp)

        start = time.perf_counter()
        milp = solve_makespan(instance, time_limit=time_limit)
        inference_times.append(time.perf_counter() - start)

        ep = milp_episode_metrics(instance, milp)
        episode_lengths.append(float(ep["l"]))
        episode_makespans.append(float(ep["makespan"]))
        episode_successes.append(1.0 if ep["success"] else 0.0)
        episode_timeouts.append(1.0 if ep.get("truncated") else 0.0)

    result = _aggregate_eval(
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
    logger.info("MILP evaluation finished:\n%s", result.format_summary())
    return result


@dataclass
class LpEpisodeRecord:
    """Per-instance LP-rounding metrics for the eval suite."""

    episode: int
    seed: int
    lp_lower_bound: float
    makespan: float
    lp_ratio: float
    runtime_s: float
    rounding_trials: int
    best_rule: str
    n_fractional: int
    lp_status: str
    max_constraint_violation: float = float("inf")
    assignment_sum_violation: float = float("inf")
    opt: Optional[float] = None
    lp_to_opt_gap: Optional[float] = None
    sol_to_opt_gap: Optional[float] = None
    rule_makespans: Dict[str, float] = field(default_factory=dict)


def _lp_episode_metrics(instance, result: LpRoundingResult) -> dict:
    if result.status == "Feasible" and np.isfinite(result.makespan):
        return {
            "l": float(instance.n_operations),
            "makespan": float(result.makespan),
            "success": True,
            "truncated": False,
        }
    return {
        "l": float(instance.n_operations),
        "makespan": float("inf"),
        "success": False,
        "truncated": result.lp_status != "Optimal",
    }


def _gap(numer: float, denom: float) -> Optional[float]:
    if not (np.isfinite(numer) and np.isfinite(denom)) or abs(denom) <= 1e-12:
        return None
    return float(numer / denom)


def evaluate_lp_rounding_fjsp(
    env: VecEnv,
    n_episodes: int = 20,
    seed: int = 42,
    *,
    rounding_trials: int = 20,
    time_limit: Optional[float] = None,
    compare_milp: bool = False,
    milp_time_limit: Optional[float] = None,
) -> Tuple[EvalResult, List[LpEpisodeRecord]]:
    """Solve each held-out instance with LP relaxation + rounding/list scheduling.

    Episode ``i`` uses seed ``seed + i`` (same instance stream as the other
    baselines). The LP is solved once per instance; rounding trials then
    list-schedule candidate assignments. Optional ``compare_milp`` runs the
    existing exact solver once per instance for gap reporting.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if rounding_trials < 1:
        raise ValueError(f"rounding_trials must be >= 1, got {rounding_trials}")
    _require_dummy_vec(env, "LP-rounding evaluation")
    if env.num_envs != 1:
        raise ValueError(f"LP-rounding evaluation requires n_envs=1, got {env.num_envs}")

    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []
    records: List[LpEpisodeRecord] = []

    logger.info(
        "Starting LP-rounding evaluation: n_episodes=%d seed=%d trials=%d "
        "time_limit=%s compare_milp=%s",
        n_episodes,
        seed,
        rounding_trials,
        time_limit,
        compare_milp,
    )

    for ep_idx in range(n_episodes):
        ep_seed = int(seed) + int(ep_idx)
        env.seed(ep_seed)
        env.reset()
        fjsp = env.envs[0].unwrapped
        instance = extract_fjsp_instance(fjsp)

        start = time.perf_counter()
        lp_res = solve_lp_rounding(
            instance,
            seed=ep_seed,
            rounding_trials=int(rounding_trials),
            time_limit=time_limit,
        )
        runtime_s = time.perf_counter() - start
        inference_times.append(runtime_s)

        ep = _lp_episode_metrics(instance, lp_res)
        episode_lengths.append(float(ep["l"]))
        episode_makespans.append(float(ep["makespan"]))
        episode_successes.append(1.0 if ep["success"] else 0.0)
        episode_timeouts.append(1.0 if ep.get("truncated") else 0.0)

        opt: Optional[float] = None
        lp_to_opt: Optional[float] = None
        sol_to_opt: Optional[float] = None
        if compare_milp:
            milp = solve_makespan(
                instance,
                time_limit=milp_time_limit if milp_time_limit is not None else time_limit,
            )
            if milp.status == "Optimal" and np.isfinite(milp.makespan):
                opt = float(milp.makespan)
                lp_to_opt = _gap(opt - float(lp_res.lp_lower_bound), opt)
                sol_to_opt = _gap(float(lp_res.makespan) - opt, opt)

        records.append(
            LpEpisodeRecord(
                episode=int(ep_idx),
                seed=ep_seed,
                lp_lower_bound=float(lp_res.lp_lower_bound),
                makespan=float(lp_res.makespan),
                lp_ratio=float(lp_res.lp_ratio),
                runtime_s=float(runtime_s),
                rounding_trials=int(rounding_trials),
                best_rule=str(lp_res.best_rule or ""),
                n_fractional=int(lp_res.n_fractional),
                lp_status=str(lp_res.lp_status),
                max_constraint_violation=float(lp_res.max_constraint_violation),
                assignment_sum_violation=float(lp_res.assignment_sum_violation),
                opt=opt,
                lp_to_opt_gap=lp_to_opt,
                sol_to_opt_gap=sol_to_opt,
                rule_makespans=dict(lp_res.rule_makespans),
            )
        )

    result = _aggregate_eval(
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
    logger.info("LP-rounding evaluation finished:\n%s", result.format_summary())
    return result, records


def print_eval_result(result: EvalResult, title: str = "FJSP PPO Evaluation Results") -> None:
    """Print evaluation metrics to stdout."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(result.format_summary())
    print("=" * 60)


def sample_masked_random_actions(
    action_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one action per env uniformly among valid mask entries.

    Args:
        action_mask: Shape ``(n_envs, n_actions)`` or ``(n_actions,)``.
        rng: NumPy Generator.

    Returns:
        Int64 actions of shape ``(n_envs,)``.

    Raises:
        ValueError: If any environment has an empty action mask.
    """
    mask = np.asarray(action_mask, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    if mask.ndim != 2:
        raise ValueError(f"action_mask must be 1D or 2D, got shape {mask.shape}")

    n_envs = mask.shape[0]
    actions = np.zeros(n_envs, dtype=np.int64)
    for i in range(n_envs):
        valid = np.flatnonzero(mask[i] > 0.5)
        if valid.size == 0:
            raise ValueError(f"empty action mask for env {i}")
        actions[i] = int(rng.choice(valid))
    return actions


def _aggregate_eval(
    episode_lengths,
    episode_makespans,
    episode_successes,
    episode_timeouts,
    inference_times,
) -> EvalResult:
    n_success = int(sum(1 for s in episode_successes if s > 0.5))
    n_timeout = int(sum(1 for t in episode_timeouts if t > 0.5))
    n_failure = max(0, int(len(episode_successes) - n_success - n_timeout))
    success_makespans = [
        m
        for m, s in zip(episode_makespans, episode_successes)
        if s > 0.5 and np.isfinite(m)
    ]
    return EvalResult(
        mean_makespan=float(np.mean(success_makespans))
        if success_makespans
        else float("inf"),
        std_makespan=float(np.std(success_makespans)) if success_makespans else float("nan"),
        mean_ep_length=float(np.mean(episode_lengths)),
        std_ep_length=float(np.std(episode_lengths)),
        success_rate=float(np.mean(episode_successes)),
        mean_inference_time_s=safe_mean(np.asarray(inference_times, dtype=np.float64)),
        n_episodes=len(episode_makespans),
        n_success=n_success,
        n_failure=n_failure,
        n_timeout=n_timeout,
    )


__all__ = [
    "EvalResult",
    "LpEpisodeRecord",
    "evaluate_heuristic_fjsp",
    "evaluate_lp_rounding_fjsp",
    "evaluate_milp_fjsp",
    "evaluate_policy_fjsp",
    "evaluate_random_fjsp",
    "print_eval_result",
    "sample_masked_random_actions",
]

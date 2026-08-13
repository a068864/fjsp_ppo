"""Shared evaluation helpers for FJSP PPO policies."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from heuristics.dispatch_rules import select_heuristic_action
from solvers.milp import extract_fjsp_instance, milp_episode_metrics, solve_makespan
from utils import get_logger, safe_mean

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """Aggregate metrics from a deterministic (or stochastic) evaluation run."""

    mean_reward: float
    std_reward: float
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

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a plain dictionary."""
        return {key: float(value) for key, value in asdict(self).items()}

    def format_summary(self) -> str:
        """Human-readable multi-line summary."""
        return "\n".join(
            [
                f"Episodes          : {self.n_episodes}",
                f"Average reward    : {self.mean_reward:.4f} ± {self.std_reward:.4f}",
                f"Successful-episode makespan : {self.mean_makespan:.4f} ± {self.std_makespan:.4f}",
                f"Episode length    : {self.mean_ep_length:.2f} ± {self.std_ep_length:.2f}",
                f"Success rate      : {self.success_rate:.2%}",
                f"Success/Failure/Timeout : {self.n_success}/{self.n_failure}/{self.n_timeout}",
                f"Inference time    : {self.mean_inference_time_s * 1000:.3f} ms/step",
            ]
        )


def _episode_from_info(
    info: Dict[str, Any],
    fallback_reward: float,
    fallback_length: int,
) -> Dict[str, Any]:
    """Build episode stats from info, with numeric fallbacks."""
    if "episode" in info and isinstance(info["episode"], dict):
        ep = dict(info["episode"])
        ep.setdefault("r", fallback_reward)
        ep.setdefault("l", fallback_length)
        ep.setdefault("makespan", float("inf"))
        ep.setdefault("success", False)
        return ep

    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        if "episode" in final_info and isinstance(final_info["episode"], dict):
            ep = dict(final_info["episode"])
            ep.setdefault("r", fallback_reward)
            ep.setdefault("l", fallback_length)
            ep.setdefault("makespan", float(final_info.get("makespan", float("inf"))))
            ep.setdefault("success", bool(final_info.get("success", False)))
            return ep
        return {
            "r": fallback_reward,
            "l": fallback_length,
            "makespan": float(final_info.get("makespan", float("inf"))),
            "success": bool(final_info.get("success", False)),
        }

    return {
        "r": fallback_reward,
        "l": fallback_length,
        "makespan": float(info.get("makespan", float("inf"))),
        "success": bool(info.get("success", False)),
    }


def evaluate_policy_fjsp(
    model: BaseAlgorithm,
    env: VecEnv,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> EvalResult:
    """Run evaluation rollouts and compute FJSP metrics.

    Args:
        model: Trained SB3 model (``PPO`` with ``GraphActorCriticPolicy``).
        env: Vectorized evaluation environment.
        n_episodes: Number of completed episodes to collect.
        deterministic: If True, use greedy actions.

    Returns:
        ``EvalResult`` with reward, makespan, length, success, and inference time.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    episode_rewards: List[float] = []
    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []

    n_envs = env.num_envs
    running_rewards = np.zeros(n_envs, dtype=np.float64)
    running_lengths = np.zeros(n_envs, dtype=np.int64)

    observations = env.reset()

    while len(episode_rewards) < n_episodes:
        mask = observations.get("action_mask") if isinstance(observations, dict) else None
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(1, -1)
            if np.any(np.sum(mask_arr > 0.5, axis=-1) == 0):
                raise ValueError("empty action mask during evaluation")

        start = time.perf_counter()
        actions, _states = model.predict(observations, deterministic=deterministic)
        inference_times.append(time.perf_counter() - start)

        observations, rewards, dones, infos = env.step(actions)
        running_rewards += np.asarray(rewards, dtype=np.float64)
        running_lengths += 1

        for env_idx, done in enumerate(np.asarray(dones, dtype=bool)):
            if not done:
                continue

            info = infos[env_idx] if env_idx < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep = _episode_from_info(
                info,
                fallback_reward=float(running_rewards[env_idx]),
                fallback_length=int(running_lengths[env_idx]),
            )

            episode_rewards.append(float(ep.get("r", running_rewards[env_idx])))
            episode_lengths.append(float(ep.get("l", running_lengths[env_idx])))
            episode_makespans.append(float(ep.get("makespan", float("inf"))))
            success = bool(ep.get("success", False))
            episode_successes.append(1.0 if success else 0.0)
            timed_out = bool(ep.get("truncated", info.get("TimeLimit.truncated", False)))
            episode_timeouts.append(1.0 if timed_out and not success else 0.0)

            running_rewards[env_idx] = 0.0
            running_lengths[env_idx] = 0

            if len(episode_rewards) >= n_episodes:
                break

    episode_rewards = episode_rewards[:n_episodes]
    episode_lengths = episode_lengths[:n_episodes]
    episode_makespans = episode_makespans[:n_episodes]
    episode_successes = episode_successes[:n_episodes]
    episode_timeouts = episode_timeouts[:n_episodes]

    result = _aggregate_eval(
        episode_rewards,
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
    return result


def evaluate_random_fjsp(
    env: VecEnv,
    n_episodes: int = 20,
    seed: int = 42,
) -> EvalResult:
    """Run evaluation rollouts with uniform random valid actions.

    Args:
        env: Vectorized evaluation environment.
        n_episodes: Number of completed episodes to collect.
        seed: RNG seed for action sampling.

    Returns:
        ``EvalResult`` with reward, makespan, length, success, and sampling time.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")

    rng = np.random.default_rng(seed)

    episode_rewards: List[float] = []
    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []

    n_envs = env.num_envs
    running_rewards = np.zeros(n_envs, dtype=np.float64)
    running_lengths = np.zeros(n_envs, dtype=np.int64)

    observations = env.reset()
    logger.info(
        "Starting random evaluation: n_episodes=%d seed=%d n_envs=%d",
        n_episodes,
        seed,
        n_envs,
    )

    while len(episode_rewards) < n_episodes:
        start = time.perf_counter()
        actions = sample_masked_random_actions(observations["action_mask"], rng)
        inference_times.append(time.perf_counter() - start)

        observations, rewards, dones, infos = env.step(actions)
        running_rewards += np.asarray(rewards, dtype=np.float64)
        running_lengths += 1

        for env_idx, done in enumerate(np.asarray(dones, dtype=bool)):
            if not done:
                continue

            info = infos[env_idx] if env_idx < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep = _episode_from_info(
                info,
                fallback_reward=float(running_rewards[env_idx]),
                fallback_length=int(running_lengths[env_idx]),
            )

            episode_rewards.append(float(ep.get("r", running_rewards[env_idx])))
            episode_lengths.append(float(ep.get("l", running_lengths[env_idx])))
            episode_makespans.append(float(ep.get("makespan", float("inf"))))
            success = bool(ep.get("success", False))
            episode_successes.append(1.0 if success else 0.0)
            timed_out = bool(ep.get("truncated", info.get("TimeLimit.truncated", False)))
            episode_timeouts.append(1.0 if timed_out and not success else 0.0)

            running_rewards[env_idx] = 0.0
            running_lengths[env_idx] = 0

            if len(episode_rewards) >= n_episodes:
                break

    episode_rewards = episode_rewards[:n_episodes]
    episode_lengths = episode_lengths[:n_episodes]
    episode_makespans = episode_makespans[:n_episodes]
    episode_successes = episode_successes[:n_episodes]
    episode_timeouts = episode_timeouts[:n_episodes]

    result = _aggregate_eval(
        episode_rewards,
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
    logger.info("Random evaluation finished:\n%s", result.format_summary())
    return result


def evaluate_heuristic_fjsp(
    env: VecEnv,
    rule: str,
    n_episodes: int = 20,
) -> EvalResult:
    """Run evaluation rollouts with a classic dispatching rule.

    Requires an in-process DummyVecEnv exposing ``env.envs`` so rules can read
    live ``FJSPEnv`` state (``n_envs=1`` recommended).

    Args:
        env: Vectorized evaluation environment with ``envs`` attribute.
        rule: Dispatching rule name (see ``heuristics.RULES``).
        n_episodes: Number of completed episodes to collect.

    Returns:
        ``EvalResult`` with reward, makespan, length, success, and select time.
    """
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be positive, got {n_episodes}")
    if not hasattr(env, "envs"):
        raise ValueError(
            "heuristic evaluation requires GraphDummyVecEnv (env.envs); "
            "use make_vec_env(..., use_subprocess=False)"
        )

    episode_rewards: List[float] = []
    episode_lengths: List[float] = []
    episode_makespans: List[float] = []
    episode_successes: List[float] = []
    episode_timeouts: List[float] = []
    inference_times: List[float] = []

    n_envs = env.num_envs
    running_rewards = np.zeros(n_envs, dtype=np.float64)
    running_lengths = np.zeros(n_envs, dtype=np.int64)

    observations = env.reset()
    logger.info(
        "Starting heuristic evaluation: rule=%s n_episodes=%d n_envs=%d",
        rule,
        n_episodes,
        n_envs,
    )

    while len(episode_rewards) < n_episodes:
        start = time.perf_counter()
        actions = np.zeros(n_envs, dtype=np.int64)
        for env_idx in range(n_envs):
            fjsp = env.envs[env_idx].unwrapped
            actions[env_idx] = select_heuristic_action(fjsp, rule)
        inference_times.append(time.perf_counter() - start)

        observations, rewards, dones, infos = env.step(actions)
        running_rewards += np.asarray(rewards, dtype=np.float64)
        running_lengths += 1

        for env_idx, done in enumerate(np.asarray(dones, dtype=bool)):
            if not done:
                continue

            info = infos[env_idx] if env_idx < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep = _episode_from_info(
                info,
                fallback_reward=float(running_rewards[env_idx]),
                fallback_length=int(running_lengths[env_idx]),
            )

            episode_rewards.append(float(ep.get("r", running_rewards[env_idx])))
            episode_lengths.append(float(ep.get("l", running_lengths[env_idx])))
            episode_makespans.append(float(ep.get("makespan", float("inf"))))
            success = bool(ep.get("success", False))
            episode_successes.append(1.0 if success else 0.0)
            timed_out = bool(ep.get("truncated", info.get("TimeLimit.truncated", False)))
            episode_timeouts.append(1.0 if timed_out and not success else 0.0)

            running_rewards[env_idx] = 0.0
            running_lengths[env_idx] = 0

            if len(episode_rewards) >= n_episodes:
                break

    episode_rewards = episode_rewards[:n_episodes]
    episode_lengths = episode_lengths[:n_episodes]
    episode_makespans = episode_makespans[:n_episodes]
    episode_successes = episode_successes[:n_episodes]
    episode_timeouts = episode_timeouts[:n_episodes]

    result = _aggregate_eval(
        episode_rewards,
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
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
    if not hasattr(env, "envs"):
        raise ValueError(
            "MILP evaluation requires GraphDummyVecEnv (env.envs); "
            "use make_vec_env(..., use_subprocess=False)"
        )
    if env.num_envs != 1:
        raise ValueError(f"MILP evaluation requires n_envs=1, got {env.num_envs}")

    episode_rewards: List[float] = []
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
        episode_rewards.append(float(ep["r"]))
        episode_lengths.append(float(ep["l"]))
        episode_makespans.append(float(ep["makespan"]))
        episode_successes.append(1.0 if ep["success"] else 0.0)
        episode_timeouts.append(1.0 if ep.get("truncated") else 0.0)

    result = _aggregate_eval(
        episode_rewards,
        episode_lengths,
        episode_makespans,
        episode_successes,
        episode_timeouts,
        inference_times,
    )
    logger.info("MILP evaluation finished:\n%s", result.format_summary())
    return result


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
    episode_rewards,
    episode_lengths,
    episode_makespans,
    episode_successes,
    episode_timeouts,
    inference_times,
) -> EvalResult:
    n_success = int(sum(1 for s in episode_successes if s > 0.5))
    n_timeout = int(sum(1 for t in episode_timeouts if t > 0.5))
    n_failure = max(0, int(len(episode_successes) - n_success - n_timeout))
    # Label makespan as successful-episode makespan.
    success_makespans = [
        m
        for m, s in zip(episode_makespans, episode_successes)
        if s > 0.5 and np.isfinite(m)
    ]
    return EvalResult(
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        mean_makespan=float(np.mean(success_makespans))
        if success_makespans
        else float("inf"),
        std_makespan=float(np.std(success_makespans)) if success_makespans else float("nan"),
        mean_ep_length=float(np.mean(episode_lengths)),
        std_ep_length=float(np.std(episode_lengths)),
        success_rate=float(np.mean(episode_successes)),
        mean_inference_time_s=safe_mean(np.asarray(inference_times, dtype=np.float64)),
        n_episodes=len(episode_rewards),
        n_success=n_success,
        n_failure=n_failure,
        n_timeout=n_timeout,
    )


__all__ = [
    "EvalResult",
    "evaluate_heuristic_fjsp",
    "evaluate_milp_fjsp",
    "evaluate_policy_fjsp",
    "evaluate_random_fjsp",
    "print_eval_result",
    "sample_masked_random_actions",
]

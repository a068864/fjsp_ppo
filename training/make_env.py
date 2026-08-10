"""Environment factories for FJSP PPO (single env and vectorized)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from torch_geometric.data import HeteroData

from config import EnvConfig, TrainConfig
from envs.fjsp_env import FJSPEnv, make_sb3_graph_observation_space
from monitor import FJSPMonitor
from utils import get_logger, worker_seed

logger = get_logger(__name__)


def stack_fjsp_obs(obs_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack per-env observations while preserving ``HeteroData`` graphs.

    ``dummy`` and ``action_mask`` are stacked as numeric arrays. ``graph`` is
    stored as an object array so PyG graphs are never flattened into vectors.
    """
    if len(obs_list) == 0:
        raise ValueError("Cannot stack empty observation list")

    dummy = np.stack([np.asarray(o["dummy"], dtype=np.float32) for o in obs_list], axis=0)
    action_mask = np.stack(
        [np.asarray(o["action_mask"], dtype=np.float32) for o in obs_list],
        axis=0,
    )
    graphs = np.empty((len(obs_list),), dtype=object)
    for i, obs in enumerate(obs_list):
        graph = obs["graph"]
        if not isinstance(graph, HeteroData):
            raise TypeError(f"Expected HeteroData graph, got {type(graph)}")
        graphs[i] = graph

    return {
        "dummy": dummy,
        "action_mask": action_mask,
        "graph": graphs,
    }


def make_sb3_observation_space(n_actions: int) -> spaces.Dict:
    """Backward-compatible alias for the canonical GraphObs SB3 Dict space."""
    return make_sb3_graph_observation_space(n_actions)


def make_env_fn(
    cfg: TrainConfig,
    rank: int = 0,
    *,
    monitor_dir: Optional[str] = None,
    for_eval: bool = False,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds one wrapped FJSP environment."""

    def _init() -> gym.Env:
        env_cfg: EnvConfig = cfg.env
        seed = worker_seed(cfg.seed, rank)
        env = FJSPEnv(
            n_machines=env_cfg.n_machines,
            n_jobs=env_cfg.n_jobs,
            avg_operations_per_job=env_cfg.avg_operations_per_job,
            time_penalty=env_cfg.time_penalty,
            max_operation_duration=env_cfg.max_operation_duration,
            completion_reward=env_cfg.completion_reward,
            connection_drop_prob=env_cfg.connection_drop_prob,
            compatible_efficiency_std=env_cfg.compatible_efficiency_std,
            time_step=env_cfg.time_step,
            min_eligible_machines=env_cfg.min_eligible_machines,
            cross_job_dep_prob=env_cfg.cross_job_dep_prob,
            shared_dep_prob=env_cfg.shared_dep_prob,
            seed=seed,
            device="cpu",
        )
        monitor_path = None
        if monitor_dir is not None:
            monitor_path = f"{monitor_dir}/worker_{rank}"
        env = FJSPMonitor(env, filename=monitor_path, allow_early_resets=True)
        # Do not eager-reset here; SB3 VecEnv applies seed/options on first reset.
        return env

    return _init


class GraphDummyVecEnv(VecEnv):
    """In-process vectorized env that preserves HeteroData observations."""

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]) -> None:
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        n_actions = int(env.action_space.n)
        observation_space = make_sb3_observation_space(n_actions)
        super().__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=env.action_space,
        )
        self.actions: Optional[np.ndarray] = None
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float64)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def reset(self) -> Dict[str, Any]:
        obs_list: List[Dict[str, Any]] = []
        for env_idx, env in enumerate(self.envs):
            maybe_options = (
                {"options": self._options[env_idx]} if self._options[env_idx] else {}
            )
            obs, self.reset_infos[env_idx] = env.reset(
                seed=self._seeds[env_idx], **maybe_options
            )
            obs_list.append(obs)
        self._reset_seeds()
        self._reset_options()
        return stack_fjsp_obs(obs_list)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        if self.actions is None:
            raise RuntimeError("step_async() must be called before step_wait()")
        obs_list: List[Dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            action = self.actions[env_idx]
            obs, reward, terminated, truncated, info = self.envs[env_idx].step(action)
            done = bool(terminated or truncated)
            self.buf_rews[env_idx] = float(reward)
            self.buf_dones[env_idx] = done
            # Match SB3 DummyVecEnv: bootstrap value only on pure truncations.
            info = dict(info)
            info["TimeLimit.truncated"] = bool(truncated and not terminated)
            self.buf_infos[env_idx] = info
            if done:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, reset_info = self.envs[env_idx].reset()
                self.reset_infos[env_idx] = reset_info
            obs_list.append(obs)
        return (
            stack_fjsp_obs(obs_list),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name: str, indices=None):
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        indices = self._get_indices(indices)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in indices
        ]

    def env_is_wrapped(self, wrapper_class, indices=None):
        indices = self._get_indices(indices)
        from stable_baselines3.common.vec_env.util import is_wrapped

        return [is_wrapped(self.envs[i], wrapper_class) for i in indices]

    def seed(self, seed: Optional[int] = None):
        # SB3 deferred seed contract: store seeds for the next reset(); do not reset now.
        if seed is None:
            seed = 0
        self._seeds = [seed + i for i in range(self.num_envs)]
        return list(self._seeds)


class GraphSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv that stacks opaque FJSP graph observations correctly."""

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]], start_method: Optional[str] = None):
        super().__init__(env_fns, start_method=start_method)
        # Replace observation space with SB3-friendly Dict (graphs remain object arrays).
        n_actions = int(self.action_space.n)
        self.observation_space = make_sb3_observation_space(n_actions)

    def reset(self) -> Dict[str, Any]:
        # Match SB3 SubprocVecEnv: worker expects (seed, options), not None.
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        self._reset_seeds()
        self._reset_options()
        return stack_fjsp_obs(list(obs))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # SB3 >= 2.1 workers return (obs, rew, done, info, reset_info).
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return (
            stack_fjsp_obs(list(obs)),
            np.stack(rews),
            np.stack(dones),
            list(infos),
        )


def make_vec_env(
    cfg: TrainConfig,
    *,
    n_envs: Optional[int] = None,
    use_subprocess: Optional[bool] = None,
    monitor_dir: Optional[str] = None,
    for_eval: bool = False,
) -> VecEnv:
    """Build a vectorized FJSP environment.

    Args:
        cfg: Training configuration.
        n_envs: Override for ``cfg.n_envs``. Documented presets: 8, 16, 32.
        use_subprocess: If True, use ``GraphSubprocVecEnv``; if False, use
            ``GraphDummyVecEnv``. Defaults to subprocess when ``n_envs > 1``.
        monitor_dir: Optional monitor output directory.
        for_eval: Unused; kept for call-site compatibility.

    Returns:
        Vectorized environment preserving ``HeteroData`` graphs in observations.
    """
    n = int(n_envs if n_envs is not None else cfg.n_envs)
    if n <= 0:
        raise ValueError(f"n_envs must be positive, got {n}")

    if use_subprocess is None:
        use_subprocess = n > 1

    env_fns = [
        make_env_fn(cfg, rank=i, monitor_dir=monitor_dir, for_eval=for_eval)
        for i in range(n)
    ]

    if use_subprocess:
        logger.info("Creating GraphSubprocVecEnv with n_envs=%d", n)
        return GraphSubprocVecEnv(env_fns)

    logger.info("Creating GraphDummyVecEnv with n_envs=%d", n)
    return GraphDummyVecEnv(env_fns)


__all__ = [
    "GraphDummyVecEnv",
    "GraphSubprocVecEnv",
    "make_env_fn",
    "make_sb3_observation_space",
    "make_vec_env",
    "stack_fjsp_obs",
]

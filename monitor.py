"""Episode monitoring for FJSP environments.

Tracks reward, length, makespan, and success in an SB3-compatible format so
callbacks and TensorBoard can consume episode statistics from ``info``.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
from gymnasium.core import Env

from utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


class FJSPMonitor(gym.Wrapper):
    """Monitor episode statistics for Flexible Job Shop Scheduling.

    At episode end, attaches::

        info["episode"] = {
            "r": episode_reward,
            "l": episode_length,
            "t": wall_time_seconds,
            "makespan": makespan,
            "success": success,
        }

    Optionally writes a CSV log compatible with classic Gym Monitor exports.
    """

    def __init__(
        self,
        env: Env,
        filename: Optional[PathLike] = None,
        allow_early_resets: bool = True,
    ) -> None:
        """
        Args:
            env: Environment to wrap.
            filename: Optional CSV path (without or with ``.csv`` / ``.monitor.csv``).
            allow_early_resets: If False, raise when ``reset`` is called mid-episode.
        """
        super().__init__(env)
        self.allow_early_resets = bool(allow_early_resets)

        self.rewards: List[float] = []
        self.needs_reset = True
        self.total_episodes = 0
        self.t_start = time.time()
        self._episode_start_time = self.t_start

        self.results_writer: Optional[_MonitorCSVWriter] = None
        if filename is not None:
            self.results_writer = _MonitorCSVWriter(filename)
            logger.info("FJSPMonitor writing episode CSV to %s", self.results_writer.path)

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an FJSPMonitor env before done. "
                "Set allow_early_resets=True to allow this."
            )
        self.rewards = []
        self.needs_reset = False
        self._episode_start_time = time.time()
        return self.env.reset(**kwargs)

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.needs_reset:
            raise RuntimeError("Tried to step FJSPMonitor env before reset().")

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        # Prefer explicit raw_reward from RewardNormalizeWrapper when present.
        if "raw_reward" in info:
            raw = float(info["raw_reward"])
        else:
            raw = float(reward)
        self.rewards.append(raw)

        done = bool(terminated or truncated)
        if done:
            ep_rew = float(sum(self.rewards))
            ep_len = int(len(self.rewards))
            ep_time = float(time.time() - self._episode_start_time)
            makespan = float(info.get("makespan", float("inf")))
            success = bool(info.get("success", False))

            # Prefer final_info fields when autoreset wrappers nest terminal info.
            final_info = info.get("final_info")
            if isinstance(final_info, dict):
                makespan = float(final_info.get("makespan", makespan))
                success = bool(final_info.get("success", success))

            episode_info = {
                "r": ep_rew,
                "l": ep_len,
                "t": ep_time,
                "makespan": makespan,
                "success": success,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            info["episode"] = episode_info
            self.total_episodes += 1
            self.needs_reset = True

            if self.results_writer is not None:
                self.results_writer.write_row(
                    {
                        "r": ep_rew,
                        "l": ep_len,
                        "t": time.time() - self.t_start,
                        "makespan": makespan,
                        "success": int(success),
                    }
                )

            logger.debug(
                "Episode %d finished: reward=%.3f length=%d makespan=%.3f success=%s",
                self.total_episodes,
                ep_rew,
                ep_len,
                makespan,
                success,
            )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.results_writer is not None:
            self.results_writer.close()
        return self.env.close()


class _MonitorCSVWriter:
    """Minimal CSV writer for episode rows (append-safe on resume)."""

    def __init__(self, filename: PathLike) -> None:
        path = Path(filename)
        if path.suffix == "":
            path = path.with_suffix(".monitor.csv")
        elif path.suffix == ".csv" and not str(path).endswith(".monitor.csv"):
            path = path.with_name(path.stem + ".monitor.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        new_file = not path.is_file() or path.stat().st_size == 0
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["r", "l", "t", "makespan", "success"],
        )
        if new_file:
            # Gym Monitor-compatible metadata comment + header.
            self._file.write("# gymnasium_monitor_version=1.0\n")
            self._file.write(f"# env_id=FJSPEnv t_start={time.time()}\n")
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


__all__ = [
    "FJSPMonitor",
]

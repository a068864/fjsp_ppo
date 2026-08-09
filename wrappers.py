"""Environment wrappers for FJSP PPO training.

Provides TimeLimit application and the standard ``wrap_fjsp_env`` stack.
Reward normalization was removed so resume cannot silently change reward scale.
"""

from __future__ import annotations

from typing import Any

from gymnasium.core import Env
from gymnasium.wrappers import TimeLimit

from utils import get_logger

logger = get_logger(__name__)


def apply_timelimit(env: Env, max_episode_steps: int) -> Env:
    """Wrap ``env`` with Gymnasium ``TimeLimit`` if a positive limit is given."""
    if max_episode_steps is None or max_episode_steps <= 0:
        return env
    return TimeLimit(env, max_episode_steps=int(max_episode_steps))


def wrap_fjsp_env(
    env: Env,
    *,
    max_episode_steps: int = 500,
    normalize_reward: bool = False,
) -> Env:
    """Apply the standard FJSP wrapper stack (TimeLimit only)."""
    del normalize_reward  # retained for call-site compatibility; intentionally unused
    env = apply_timelimit(env, max_episode_steps)
    logger.debug("Wrapped FJSP env: timelimit=%s", max_episode_steps)
    return env


__all__ = [
    "TimeLimit",
    "apply_timelimit",
    "wrap_fjsp_env",
]

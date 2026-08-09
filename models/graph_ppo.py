"""PPO subclass that keeps HeteroData graphs out of ``torch.as_tensor``."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from training.graph_buffer import GraphDictRolloutBuffer, graph_obs_as_tensor


class GraphPPO(PPO):
    """PPO with graph-safe rollout collection and rollout buffer.

    SB3's default ``collect_rollouts`` calls ``obs_as_tensor`` on the full
    observation dict, which cannot convert ``HeteroData``. This subclass:

    - defaults ``rollout_buffer_class`` to ``GraphDictRolloutBuffer``
    - converts only numeric fields to tensors during rollouts
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = GraphDictRolloutBuffer
        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: GraphDictRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Copy of SB3 collect_rollouts with graph-safe observation conversion."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = graph_obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(
                graph_obs_as_tensor(new_obs, self.device)
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


__all__ = ["GraphPPO"]

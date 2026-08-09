"""Graph-aware rollout buffer and observation tensor helpers for SB3 PPO."""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from torch_geometric.data import HeteroData

GRAPH_KEY = "graph"


def slim_graph_for_policy(graph: HeteroData) -> HeteroData:
    """Retain only tensors consumed by the policy encoder / efficiency scoring."""
    from models.graph_encoder import EDGE_TYPES

    out = HeteroData()
    out["operation"].x = graph["operation"].x.clone()
    out["machine"].x = graph["machine"].x.clone()
    for edge_type in EDGE_TYPES:
        if edge_type not in graph.edge_types:
            continue
        edge_index = graph[edge_type].edge_index
        if edge_index is None or edge_index.numel() == 0:
            continue
        out[edge_type].edge_index = edge_index.clone()
        edge_attr = getattr(graph[edge_type], "edge_attr", None)
        if edge_attr is not None:
            out[edge_type].edge_attr = edge_attr.clone()
    return out


def graph_obs_as_tensor(
    obs: Dict[str, Any],
    device: Union[th.device, str],
) -> Dict[str, Any]:
    """Convert numeric obs fields to tensors while leaving ``HeteroData`` intact.

    Unlike SB3 ``obs_as_tensor``, this does not call ``torch.as_tensor`` on the
    ``graph`` field (object array / list of ``HeteroData``).
    """
    if not isinstance(obs, dict):
        raise TypeError(f"Expected dict observation, got {type(obs)}")

    converted: Dict[str, Any] = {}
    for key, value in obs.items():
        if key == GRAPH_KEY:
            converted[key] = value
            continue
        tensor = th.as_tensor(value)
        if not th.is_floating_point(tensor):
            tensor = tensor.float()
        converted[key] = tensor.to(device)
    return converted


class GraphDictRolloutBuffer(DictRolloutBuffer):
    """Dict rollout buffer that stores opaque ``HeteroData`` graphs as objects.

    Numeric keys (``dummy``, ``action_mask``, ...) remain float32. The ``graph``
    key is stored as ``dtype=object`` and is never cast through ``to_torch``.
    """

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "GraphDictRolloutBuffer requires Dict obs"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            if key == GRAPH_KEY:
                self.observations[key] = np.empty(
                    (self.buffer_size, self.n_envs),
                    dtype=object,
                )
            else:
                self.observations[key] = np.zeros(
                    (self.buffer_size, self.n_envs, *obs_input_shape),
                    dtype=np.float32,
                )

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        # Skip RolloutBuffer.reset() — it unpacks dict obs_shape keys as ints.
        # Match SB3 DictRolloutBuffer: only reset BaseBuffer pos/full flags.
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            if key == GRAPH_KEY:
                graphs = obs[key]
                arr = np.empty((self.n_envs,), dtype=object)
                if isinstance(graphs, np.ndarray) and graphs.dtype == object:
                    iterable = [graphs[i] for i in range(graphs.shape[0])]
                else:
                    iterable = list(graphs)
                for i, graph in enumerate(iterable):
                    arr[i] = slim_graph_for_policy(graph) if graph is not None else graph
                self.observations[key][self.pos] = arr
                continue

            obs_ = np.array(obs[key])
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        action = action.reshape((self.n_envs, self.action_dim))
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                if key == GRAPH_KEY:
                    # (buffer_size, n_envs) -> (buffer_size * n_envs,)
                    self.observations[key] = obs.swapaxes(0, 1).reshape(-1)
                else:
                    self.observations[key] = self.swap_and_flatten(obs)

            for tensor in ("actions", "values", "log_probs", "advantages", "returns"):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        observations: Dict[str, Any] = {}
        for key, obs in self.observations.items():
            if key == GRAPH_KEY:
                observations[key] = obs[batch_inds]
            else:
                observations[key] = self.to_torch(obs[batch_inds])

        return DictRolloutBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )


__all__ = ["GRAPH_KEY", "GraphDictRolloutBuffer", "graph_obs_as_tensor", "slim_graph_for_policy"]

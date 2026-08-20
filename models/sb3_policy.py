"""Stable-Baselines3 policy that consumes FJSP HeteroData observations directly."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch_geometric.data import HeteroData

from config import ModelConfig
from models.actor_critic import GraphActorCritic
from training.graph_buffer import ACTION_MASK_KEY, GRAPH_KEY, graph_obs_as_tensor
from utils import get_logger

logger = get_logger(__name__)


class GraphPassthroughExtractor(BaseFeaturesExtractor):
    """Dummy feature extractor for opaque graph observations.

    SB3 requires a features extractor. Real computation happens in
    ``GraphActorCriticPolicy`` using ``obs["graph"]``; this extractor only
    returns the ``dummy`` vector so the parent class can construct itself.
    """

    def __init__(self, observation_space: spaces.Space, features_dim: int = 1) -> None:
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations: Any) -> torch.Tensor:
        if isinstance(observations, dict):
            dummy = observations["dummy"]
            tensor = torch.as_tensor(dummy)
        else:
            tensor = torch.as_tensor(observations)
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.view(tensor.shape[0], -1)[..., : self.features_dim]


class GraphActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with HeteroData encoder, EdgePredictor actor, and graph critic.

    Invalid actions are masked with ``-1e9`` before the categorical distribution
    is constructed, so they are never sampled.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        model_config: Optional[ModelConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model_config = model_config or ModelConfig()

        kwargs["features_extractor_class"] = GraphPassthroughExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 1}
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.graph_ac = GraphActorCritic(model_config=self.model_config)
        # Parent built CategoricalDistribution(action_space.n) for the unused
        # Discrete(2) head. Live categoricals use logits width (see _categorical).

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
        logger.info(
            "Initialized GraphActorCriticPolicy (hidden_dim=%d, predictor=%s)",
            self.model_config.hidden_dim,
            self.model_config.predictor_type,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(model_config=self.model_config)
        return data

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, Any]]
    ) -> Tuple[PyTorchObs, bool]:
        """Convert obs to policy input without tensorizing HeteroData graphs."""
        if not isinstance(observation, dict):
            raise TypeError(
                "GraphActorCriticPolicy expects dict observations, "
                f"got {type(observation)}"
            )

        vectorized = not isinstance(observation.get(GRAPH_KEY), HeteroData)

        packed: Dict[str, Any] = {}
        for key, value in observation.items():
            if key == GRAPH_KEY:
                if isinstance(value, HeteroData):
                    arr = np.empty((1,), dtype=object)
                    arr[0] = value
                    packed[key] = arr
                elif isinstance(value, np.ndarray) and value.dtype == object:
                    packed[key] = value if vectorized else np.array([value.reshape(-1)[0]], dtype=object)
                elif isinstance(value, (list, tuple)):
                    arr = np.empty((len(value),), dtype=object)
                    for i, graph in enumerate(value):
                        arr[i] = graph
                    packed[key] = arr
                else:
                    arr = np.empty((1,), dtype=object)
                    arr[0] = value
                    packed[key] = arr
            elif key == ACTION_MASK_KEY:
                if torch.is_tensor(value):
                    packed[key] = value
                else:
                    packed[key] = self._pack_action_mask(value, vectorized=vectorized)
            else:
                arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                if not vectorized and arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                packed[key] = arr

        return graph_obs_as_tensor(packed, self.device), vectorized

    @staticmethod
    def _pack_action_mask(value: Any, *, vectorized: bool) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            arr = np.empty((len(value),), dtype=object)
            for i, mask in enumerate(value):
                arr[i] = np.asarray(mask, dtype=np.float32).reshape(-1)
            return arr
        arr = np.asarray(value)
        if arr.dtype == object:
            return arr
        arr = np.asarray(arr, dtype=np.float32)
        if not vectorized and arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _to_list(obj: Any) -> List[Any]:
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                return [obj[i] for i in range(obj.shape[0])]
            if obj.ndim == 0:
                return [obj.item()]
            return list(obj)
        return [obj]

    def _unpack_obs(
        self, obs: Any
    ) -> Tuple[List[HeteroData], Any]:
        """Extract graph list and optional action masks from an observation."""
        if not isinstance(obs, dict):
            raise TypeError(
                "GraphActorCriticPolicy expects dict observations with keys "
                f"'graph' and 'action_mask'; got {type(obs)}"
            )

        graphs = self._to_list(obs[GRAPH_KEY])
        cleaned: List[HeteroData] = []
        for graph in graphs:
            if isinstance(graph, np.ndarray) and graph.dtype == object:
                graph = graph.item()
            if not isinstance(graph, HeteroData):
                raise TypeError(
                    f"Each graph observation must be HeteroData, got {type(graph)}"
                )
            cleaned.append(graph)

        masks_in: Any = None
        if ACTION_MASK_KEY in obs and obs[ACTION_MASK_KEY] is not None:
            masks = obs[ACTION_MASK_KEY]
            if isinstance(masks, np.ndarray) and masks.dtype == object:
                masks_in = [
                    torch.as_tensor(masks[i], device=self.device, dtype=torch.float32).reshape(-1)
                    for i in range(masks.shape[0])
                ]
            elif isinstance(masks, (list, tuple)):
                masks_in = [
                    torch.as_tensor(item, device=self.device, dtype=torch.float32).reshape(-1)
                    for item in masks
                ]
            else:
                if not torch.is_tensor(masks):
                    masks = torch.as_tensor(masks)
                masks = masks.to(device=self.device)
                if not torch.is_floating_point(masks):
                    masks = masks.float()
                if masks.dim() == 1:
                    masks = masks.unsqueeze(0)
                masks_in = masks

        return cleaned, masks_in

    @staticmethod
    def _mask_has_empty_row(masks: Any) -> bool:
        if torch.is_tensor(masks):
            return bool(((masks >= 0.5).sum(dim=-1) == 0).any().item())
        return any(int((mask >= 0.5).sum().item()) == 0 for mask in masks)

    @staticmethod
    def _categorical(logits: torch.Tensor) -> CategoricalDistribution:
        dist = CategoricalDistribution(int(logits.size(-1)))
        return dist.proba_distribution(action_logits=logits)

    def _masked_logits_and_values(
        self, obs: Any, *, require_valid_actions: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graphs, masks = self._unpack_obs(obs)
        if require_valid_actions and masks is not None:
            if self._mask_has_empty_row(masks):
                raise ValueError(
                    "Empty action mask: cannot construct an action distribution"
                )
        logits, values = self.graph_ac(graphs, action_mask=masks)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        return logits, values

    def forward(
        self,
        obs: Any,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self._masked_logits_and_values(obs, require_valid_actions=True)
        distribution = self._categorical(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Any, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits, values = self._masked_logits_and_values(obs, require_valid_actions=True)
        distribution = self._categorical(logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: Any) -> CategoricalDistribution:
        logits, _ = self._masked_logits_and_values(obs, require_valid_actions=True)
        return self._categorical(logits)

    def predict_values(self, obs: Any) -> torch.Tensor:
        # Value-only terminal bootstrapping must remain valid with empty masks.
        graphs, _masks = self._unpack_obs(obs)
        values = self.graph_ac.get_value(graphs)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        return values

    def _predict(
        self, observation: Any, deterministic: bool = False
    ) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def extract_features(
        self, obs: Any, features_extractor: Optional[Any] = None
    ) -> torch.Tensor:
        extractor = (
            self.features_extractor if features_extractor is None else features_extractor
        )
        return extractor(obs)


def make_policy_kwargs(model_config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    """Build ``policy_kwargs`` for graph PPO policies."""
    return {
        "model_config": model_config or ModelConfig(),
    }


__all__ = [
    "GraphActorCriticPolicy",
    "make_policy_kwargs",
]

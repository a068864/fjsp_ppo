"""PPO subclass that keeps HeteroData graphs out of ``torch.as_tensor``."""

from __future__ import annotations

from typing import Any

from stable_baselines3 import PPO

from training.graph_buffer import GraphDictRolloutBuffer, graph_obs_as_tensor


class GraphPPO(PPO):
    """PPO with graph-safe observation conversion during rollouts.

    SB3's ``collect_rollouts`` calls ``obs_as_tensor`` on the full observation
    dict, which cannot convert ``HeteroData``. This subclass:

    - defaults ``rollout_buffer_class`` to ``GraphDictRolloutBuffer``
    - patches ``obs_as_tensor`` for the duration of the SB3 rollout loop
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("rollout_buffer_class") is None:
            kwargs["rollout_buffer_class"] = GraphDictRolloutBuffer
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, *args: Any, **kwargs: Any) -> bool:
        """Delegate to SB3 after swapping in graph-safe obs conversion."""
        import stable_baselines3.common.on_policy_algorithm as on_policy_algorithm

        original = on_policy_algorithm.obs_as_tensor
        on_policy_algorithm.obs_as_tensor = graph_obs_as_tensor
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            on_policy_algorithm.obs_as_tensor = original


def load_graph_ppo(path: Any, env: Any, device: Any) -> GraphPPO:
    """Load a GraphPPO zip with graph policy and rollout buffer classes restored."""
    from models.sb3_policy import GraphActorCriticPolicy

    return GraphPPO.load(
        str(path),
        env=env,
        device=device,
        custom_objects={
            "policy_class": GraphActorCriticPolicy,
            "rollout_buffer_class": GraphDictRolloutBuffer,
        },
    )


__all__ = ["GraphPPO", "load_graph_ppo"]

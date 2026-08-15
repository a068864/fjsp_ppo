"""Graph-based actor-critic network for FJSP PPO."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from config import ModelConfig
from models.edge_predictor import EdgePredictor, PredictorType
from models.graph_encoder import GraphEncoder

MASK_LOGIT = -1e9

GraphInput = Union[HeteroData, Sequence[HeteroData]]
MaskInput = Optional[Union[torch.Tensor, np.ndarray, Sequence[Union[torch.Tensor, np.ndarray]]]]


class GraphActorCritic(nn.Module):
    """Actor-critic over HeteroData states.

    Actor:
        ``GraphEncoder`` -> ``EdgePredictor`` -> logits for every
        ``(machine, operation)`` pair (flat layout matching the env).

    Critic:
        Graph embedding -> MLP -> scalar ``V(s)``.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        *,
        operation_in_dim: Optional[int] = None,
        machine_in_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        predictor_type: Optional[PredictorType] = None,
        critic_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        cfg = model_config or ModelConfig()

        self.operation_in_dim = int(
            operation_in_dim if operation_in_dim is not None else cfg.operation_in_dim
        )
        self.machine_in_dim = int(
            machine_in_dim if machine_in_dim is not None else cfg.machine_in_dim
        )
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else cfg.hidden_dim)
        self.num_layers = int(num_layers if num_layers is not None else cfg.num_layers)
        self.num_heads = int(num_heads if num_heads is not None else cfg.num_heads)
        self.dropout = float(dropout if dropout is not None else cfg.dropout)
        if self.dropout != 0.0:
            raise ValueError(
                "dropout must be 0.0 for deterministic PPO likelihoods, "
                f"got {self.dropout}"
            )
        self.predictor_type: PredictorType = (
            predictor_type if predictor_type is not None else cfg.predictor_type
        )
        self.critic_hidden_dim = int(
            critic_hidden_dim
            if critic_hidden_dim is not None
            else cfg.critic_hidden_dim
        )

        self.encoder = GraphEncoder(
            operation_in_dim=self.operation_in_dim,
            machine_in_dim=self.machine_in_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.actor = EdgePredictor(
            hidden_dim=self.hidden_dim,
            predictor_type=self.predictor_type,
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.critic_hidden_dim),
            nn.LayerNorm(self.critic_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.critic_hidden_dim, self.critic_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.critic_hidden_dim // 2, 1),
        )

    @staticmethod
    def _as_graph_list(data: GraphInput) -> List[HeteroData]:
        if isinstance(data, HeteroData):
            return [data]
        return list(data)

    @staticmethod
    def efficiency_matrix(data: HeteroData, device: torch.device) -> torch.Tensor:
        """Build an (n_ops, n_machines) efficiency matrix from compatible edges."""
        n_ops = int(data["operation"].x.size(0))
        n_mach = int(data["machine"].x.size(0))
        mat = torch.zeros((n_ops, n_mach), dtype=torch.float32, device=device)
        key = ("operation", "compatible", "machine")
        if key not in data.edge_types:
            return mat
        edge_index = data[key].edge_index
        edge_attr = getattr(data[key], "edge_attr", None)
        if edge_index is None or edge_index.numel() == 0 or edge_attr is None:
            return mat
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
        attr = edge_attr.reshape(-1).to(device=device, dtype=torch.float32)
        mat[src, dst] = attr
        return mat

    @staticmethod
    def dispatch_score_matrix(
        data: HeteroData,
        efficiency: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """ECT-style pair scores (ops×machines); higher is better.

        Env efficiency is a processing-time multiplier (higher = slower), so
        expected completion is ``machine_workload + duration * efficiency``.
        """
        duration = data["operation"].x[:, 0].to(device=device, dtype=torch.float32)
        workload = data["machine"].x[:, 1].to(device=device, dtype=torch.float32)
        eff = efficiency.to(device=device, dtype=torch.float32)
        proc = duration.unsqueeze(1) * eff
        ect = workload.unsqueeze(0) + proc
        scale = duration.mean().clamp(min=1.0)
        return -ect / scale

    @staticmethod
    def _as_mask_list(
        action_mask: MaskInput,
        batch_size: int,
        n_actions: int,
        device: torch.device,
    ) -> Optional[List[torch.Tensor]]:
        if action_mask is None:
            return None

        if isinstance(action_mask, np.ndarray):
            action_mask = torch.as_tensor(action_mask)

        if isinstance(action_mask, torch.Tensor):
            if action_mask.dim() == 1:
                masks = [action_mask]
            elif action_mask.dim() == 2:
                masks = [action_mask[i] for i in range(action_mask.size(0))]
            else:
                raise ValueError(
                    f"action_mask tensor must be 1D or 2D, got shape {tuple(action_mask.shape)}"
                )
        else:
            masks = []
            for item in action_mask:
                if isinstance(item, np.ndarray):
                    item = torch.as_tensor(item)
                masks.append(torch.as_tensor(item))

        if len(masks) != batch_size:
            raise ValueError(
                f"Expected {batch_size} action masks, got {len(masks)}"
            )

        out: List[torch.Tensor] = []
        for mask in masks:
            flat = mask.reshape(-1).to(device=device, dtype=torch.float32)
            if flat.numel() != n_actions:
                raise ValueError(
                    f"Action mask length {flat.numel()} != n_actions {n_actions}"
                )
            out.append(flat)
        return out

    @staticmethod
    def apply_action_mask(
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        mask_value: float = MASK_LOGIT,
    ) -> torch.Tensor:
        """Mask invalid actions by setting their logits to ``mask_value``."""
        if action_mask is None:
            return logits
        mask = action_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if mask.numel() != logits.numel():
            raise ValueError(
                f"Mask size {mask.numel()} does not match logits size {logits.numel()}"
            )
        invalid = mask < 0.5
        if invalid.any():
            return logits.masked_fill(invalid, mask_value)
        return logits

    def _assignment_logits(
        self,
        data: HeteroData,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        efficiency = self.efficiency_matrix(data, device=device)
        dispatch = self.dispatch_score_matrix(data, efficiency, device=device)
        return self.actor(machine_emb, operation_emb, dispatch)

    def forward_single(
        self,
        data: HeteroData,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward one graph.

        Returns:
            ``(logits, value)`` with shapes ``(n_actions,)`` and ``()``.
        """
        device = next(self.parameters()).device
        machine_emb, operation_emb, graph_emb = self.encoder(data)
        logits = self._assignment_logits(data, machine_emb, operation_emb, device)
        logits = self.apply_action_mask(logits, action_mask)
        value = self.critic(graph_emb).squeeze(-1)
        return logits, value

    def forward(
        self,
        data: GraphInput,
        action_mask: MaskInput = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward one graph or a batch of graphs.

        Args:
            data: Single ``HeteroData`` or a sequence of graphs.
            action_mask: Optional mask(s) aligned with ``data``.

        Returns:
            ``(logits, values)``:
                - single graph: logits ``(A,)``, value ``()``
                - batch: logits ``(B, A)``, values ``(B,)``
        """
        graphs = self._as_graph_list(data)
        batch_size = len(graphs)
        if batch_size == 0:
            raise ValueError("Empty graph batch")

        device = next(self.parameters()).device
        # Same-size graphs collate without changing embeddings (encode_batch
        # splits on ptr). Mixed sizes fall back to per-graph forwards.
        machine_list, operation_list, graph_emb_batch = self.encoder.encode_batch(graphs)
        logits_list: List[torch.Tensor] = []
        n_actions: Optional[int] = None
        for i, graph in enumerate(graphs):
            logits_i = self._assignment_logits(
                graph, machine_list[i], operation_list[i], device
            )
            if n_actions is None:
                n_actions = int(logits_i.numel())
            elif int(logits_i.numel()) != n_actions:
                raise ValueError(
                    "All graphs in a batch must share the same action dimension; "
                    f"got {logits_i.numel()} vs {n_actions}"
                )
            logits_list.append(logits_i)

        assert n_actions is not None
        values = self.critic(graph_emb_batch).squeeze(-1)
        masks = self._as_mask_list(action_mask, batch_size, n_actions, device)
        masked = [
            self.apply_action_mask(logits, None if masks is None else masks[i])
            for i, logits in enumerate(logits_list)
        ]

        if batch_size == 1:
            return masked[0], values.squeeze(0)
        return torch.stack(masked, dim=0), values

    def get_value(self, data: GraphInput) -> torch.Tensor:
        """Return critic value(s) only (skips the actor / edge predictor)."""
        graphs = self._as_graph_list(data)
        if not graphs:
            raise ValueError("Empty graph batch")
        _m, _o, graph_emb_batch = self.encoder.encode_batch(graphs)
        values = self.critic(graph_emb_batch).squeeze(-1)
        if len(graphs) == 1:
            return values.squeeze(0)
        return values


__all__ = ["GraphActorCritic"]

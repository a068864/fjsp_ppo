"""Edge prediction heads for machine–operation assignment logits."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm


PredictorType = Literal["dot_product", "bilinear"]


class EdgePredictor(nn.Module):
    """Score every machine–operation pair from node embeddings.

    The flat logit layout matches the environment action encoding::

        action_idx = machine_id * n_operations + operation_id

    which is exactly ``(machine_emb @ operation_emb.T).view(-1)`` order
    (row-major over machines, columns over operations).

    Optional efficiency matrices (ops×machines or batched B×ops×machines) are
    added as a learned bias. Callers pass ECT-style scores (higher = better;
    typically ``-expected_completion / mean_duration``) so slower or more
    loaded machines are down-ranked.
    """

    def __init__(self, hidden_dim: int, predictor_type: PredictorType = "dot_product") -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.predictor_type = predictor_type
        self.efficiency_scale = nn.Parameter(torch.tensor(0.1))

        self.norm_m = GraphNorm(hidden_dim)
        self.norm_o = GraphNorm(hidden_dim)

        if predictor_type not in ("dot_product", "bilinear"):
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        # Slightly cooler than 1/sqrt(H): GraphNorm does not guarantee
        # unit-variance embeddings, and hot logits collapse PPO entropy.
        self.scale = math.sqrt(2.0 * hidden_dim)
        if predictor_type == "bilinear":
            self.interaction_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self._gn_index_cache: dict[tuple, torch.Tensor] = {}
        self.reset_parameters()

    def _graph_norm_index(
        self, batch_size: int, n_nodes: int, device: torch.device
    ) -> torch.Tensor:
        key = (batch_size, n_nodes, device.type, device.index)
        cached = self._gn_index_cache.get(key)
        if cached is None or cached.device != device:
            cached = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
            self._gn_index_cache[key] = cached
        return cached

    def _pair_scores(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
    ) -> torch.Tensor:
        m = self.norm_m(machine_emb)
        o = self.norm_o(operation_emb)
        if self.predictor_type == "bilinear":
            scores = torch.matmul(torch.matmul(m, self.interaction_weight), o.t())
        else:
            scores = torch.matmul(m, o.t())
        return scores.div(self.scale).view(-1)

    def _score_pair(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
        efficiency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self._pair_scores(machine_emb, operation_emb)
        if efficiency is None:
            return logits
        # efficiency: (n_ops, n_machines) -> machine-major flat
        eff = efficiency.to(device=logits.device, dtype=logits.dtype)
        if eff.dim() != 2:
            raise ValueError(f"efficiency must be 2D (n_ops, n_machines), got {tuple(eff.shape)}")
        return logits + self.efficiency_scale * eff.transpose(0, 1).reshape(-1)

    def forward_batched(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
        efficiency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score a fixed-size batch.

        Args:
            machine_emb: ``(B, n_machines, H)``
            operation_emb: ``(B, n_operations, H)``
            efficiency: optional ``(B, n_ops, n_machines)``

        Returns:
            ``(B, n_machines * n_operations)`` logits.
        """
        if machine_emb.dim() != 3 or operation_emb.dim() != 3:
            raise ValueError("forward_batched expects 3D embeddings")
        batch_size, n_machines, hidden = machine_emb.shape
        n_operations = operation_emb.size(1)
        device = machine_emb.device
        m_batch = self._graph_norm_index(batch_size, n_machines, device)
        o_batch = self._graph_norm_index(batch_size, n_operations, device)
        m = self.norm_m(
            machine_emb.reshape(batch_size * n_machines, hidden),
            batch=m_batch,
            batch_size=batch_size,
        ).view(batch_size, n_machines, hidden)
        o = self.norm_o(
            operation_emb.reshape(batch_size * n_operations, hidden),
            batch=o_batch,
            batch_size=batch_size,
        ).view(batch_size, n_operations, hidden)
        if self.predictor_type == "bilinear":
            scores = torch.matmul(torch.matmul(m, self.interaction_weight), o.transpose(-1, -2))
            logits = scores.div(self.scale).reshape(batch_size, n_machines * n_operations)
        else:
            scores = torch.matmul(m, o.transpose(-1, -2))
            logits = scores.div(self.scale).reshape(batch_size, n_machines * n_operations)
        if efficiency is None:
            return logits
        eff = efficiency.to(device=logits.device, dtype=logits.dtype)
        if eff.dim() == 2:
            eff = eff.unsqueeze(0).expand(batch_size, -1, -1)
        if eff.dim() != 3:
            raise ValueError(
                f"batched efficiency must be (B, n_ops, n_machines), got {tuple(eff.shape)}"
            )
        return logits + self.efficiency_scale * eff.transpose(1, 2).reshape(
            batch_size, -1
        )

    def forward(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
        efficiency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return flat logits over all machine–operation pairs.

        Args:
            machine_emb: Tensor of shape ``(n_machines, hidden_dim)`` or batched.
            operation_emb: Tensor of shape ``(n_operations, hidden_dim)`` or batched.
            efficiency: Optional ops×machines matrix (or batched).

        Returns:
            Tensor of shape ``(n_machines * n_operations,)`` or ``(B, A)``.
        """
        if machine_emb.dim() == 3:
            return self.forward_batched(machine_emb, operation_emb, efficiency)
        return self._score_pair(machine_emb, operation_emb, efficiency)

    def reset_parameters(self) -> None:
        """Reset efficiency bias and pair metric to the PPO-safe prior."""
        # Weak ECT residual (1.0 cloned greedy dispatch; GNN logits are ~0.7 std).
        nn.init.constant_(self.efficiency_scale, 0.1)
        if self.predictor_type == "bilinear":
            # Start as identity so init logits match scaled dot-product;
            # kaiming on HxH is too heavy and collapses policy entropy.
            nn.init.eye_(self.interaction_weight)

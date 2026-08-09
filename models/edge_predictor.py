"""Edge prediction heads for machine–operation assignment logits."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm


PredictorType = Literal["dot_product", "bilinear", "attention"]


class EdgePredictor(nn.Module):
    """Score every machine–operation pair from node embeddings.

    The flat logit layout matches the environment action encoding::

        action_idx = machine_id * n_operations + operation_id

    which is exactly ``(machine_emb @ operation_emb.T).view(-1)`` order
    (row-major over machines, columns over operations).

    Optional efficiency matrices (ops×machines or batched B×ops×machines) are
    added as a learned bias so compatibility attributes affect pair scores.
    """

    def __init__(self, hidden_dim: int, predictor_type: PredictorType = "dot_product") -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_dim = hidden_dim
        self.predictor_type = predictor_type
        self.efficiency_scale = nn.Parameter(torch.tensor(1.0))

        self.norm_m = GraphNorm(hidden_dim)
        self.norm_o = GraphNorm(hidden_dim)

        if predictor_type == "dot_product":
            # Slightly cooler than 1/sqrt(H): GraphNorm does not guarantee
            # unit-variance embeddings, and hot logits collapse PPO entropy.
            self.scale = math.sqrt(2.0 * hidden_dim)
            self.predictor = self._dot_product_predictor
        elif predictor_type == "bilinear":
            # Same cooler 1/sqrt(2H) scale as dot-product: unscaled bilinear
            # logits have O(H) magnitude at init and collapse the categorical
            # to one-hot, which blows PPO KL on the first update.
            self.scale = math.sqrt(2.0 * hidden_dim)
            self.interaction_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.predictor = self._bilinear_predictor
        elif predictor_type == "attention":
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)

            self.gate_network = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                GraphNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

            self.score_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

            self.predictor = self._attention_predictor
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

        self.reset_parameters()

    def _dot_product_predictor(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
    ) -> torch.Tensor:
        m = self.norm_m(machine_emb)
        o = self.norm_o(operation_emb)
        return torch.matmul(m, o.t()).div(self.scale).view(-1)

    def _bilinear_predictor(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
    ) -> torch.Tensor:
        m = self.norm_m(machine_emb)
        o = self.norm_o(operation_emb)
        return torch.matmul(torch.matmul(m, self.interaction_weight), o.t()).div(self.scale).view(-1)

    def _attention_predictor(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
    ) -> torch.Tensor:
        m = self.norm_m(machine_emb)
        o = self.norm_o(operation_emb)

        q = self.query_proj(m)
        k = self.key_proj(o)
        v = self.value_proj(o)

        attn_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)),
            dim=-1,
        )
        attn_output = torch.matmul(attn_weights, v)

        gates = self.gate_network(torch.cat([m, attn_output], dim=-1))
        gated_output = gates * attn_output + (1.0 - gates) * m
        machine_scores = self.score_proj(gated_output)
        return (machine_scores * attn_weights).view(-1)

    def _score_pair(
        self,
        machine_emb: torch.Tensor,
        operation_emb: torch.Tensor,
        efficiency: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.predictor(machine_emb, operation_emb)
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
        batch_size = machine_emb.size(0)
        outs = []
        for i in range(batch_size):
            eff_i = None if efficiency is None else efficiency[i]
            outs.append(self._score_pair(machine_emb[i], operation_emb[i], eff_i))
        return torch.stack(outs, dim=0)

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
        """Reset parameters using Kaiming initialization for GELU activations."""
        nn.init.constant_(self.efficiency_scale, 1.0)
        if self.predictor_type == "bilinear":
            # Start as identity so init logits match scaled dot-product;
            # kaiming on HxH is too heavy and collapses policy entropy.
            nn.init.eye_(self.interaction_weight)
        elif self.predictor_type == "attention":
            for layer in (self.query_proj, self.key_proj, self.value_proj):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)

            for layer in self.gate_network:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
                    nn.init.zeros_(layer.bias)

            for layer in self.score_proj:
                if isinstance(layer, nn.Linear):
                    if layer is self.score_proj[-1]:
                        nn.init.kaiming_normal_(
                            layer.weight, nonlinearity="leaky_relu", a=0.1
                        )
                    else:
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
                    nn.init.zeros_(layer.bias)

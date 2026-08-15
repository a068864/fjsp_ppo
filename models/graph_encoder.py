"""Heterogeneous graph encoder for FJSP states."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import AttentionalAggregation, HeteroConv, TransformerConv

from envs.fjsp_env import (
    MACH_IDLE_DURATION,
    MACH_WORKLOAD,
    OP_CP_REMAINING,
    OP_DURATION,
    OP_JOB_REMAINING_WORK,
    OP_REMAINING,
)

_OP_TIME_COLS = (
    OP_DURATION,
    OP_REMAINING,
    OP_CP_REMAINING,
    OP_JOB_REMAINING_WORK,
)

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("operation", "precede", "operation"),
    ("operation", "next", "operation"),
    ("machine", "processing", "operation"),
    ("operation", "compatible", "machine"),
]

REVERSE_EDGE_TYPES: Dict[
    Tuple[str, str, str], Tuple[str, str, str]
] = {
    ("operation", "precede", "operation"): ("operation", "succeed", "operation"),
    ("operation", "next", "operation"): ("operation", "previous", "operation"),
    ("machine", "processing", "operation"): ("operation", "processed_by", "machine"),
    ("operation", "compatible", "machine"): ("machine", "compatible_with", "operation"),
}

MESSAGE_EDGE_TYPES = EDGE_TYPES + list(REVERSE_EDGE_TYPES.values())


class HeteroResidualBlock(nn.Module):
    """Edge-aware heterogeneous TransformerConv layer with residual normalization."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        convs: Dict[Tuple[str, str, str], nn.Module] = {}
        for edge_type in MESSAGE_EDGE_TYPES:
            # root_weight=False: residual is applied outside the conv (same as
            # GATv2 residual=False). Edge attrs enter attention and values.
            convs[edge_type] = TransformerConv(
                (hidden_dim, hidden_dim),
                hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=self.dropout,
                edge_dim=1,
                root_weight=False,
            )

        self.conv = HeteroConv(convs, aggr=aggr)
        self.norm_operation = nn.LayerNorm(hidden_dim)
        self.norm_machine = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        residual = {key: value for key, value in x_dict.items()}
        # Keep empty (2, 0) indices. Dropping them makes sequential encoding skip
        # a TransformerConv, while a collated batch still runs it (other graphs
        # have the type) and add_self_loops fires on every node — PPO collect vs
        # train KL.
        filtered_edges = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in self.conv.convs
            and edge_index is not None
            and edge_index.dim() == 2
            and edge_index.size(0) == 2
        }

        if filtered_edges:
            filtered_attrs = {
                edge_type: edge_attr_dict[edge_type]
                for edge_type in filtered_edges
            }
            out = self.conv(
                x_dict,
                filtered_edges,
                edge_attr_dict=filtered_attrs,
            )
        else:
            out = {key: value for key, value in x_dict.items()}

        updated: Dict[str, torch.Tensor] = {}
        for node_type in ("operation", "machine"):
            if node_type not in residual:
                continue
            message = out.get(node_type) if filtered_edges else None
            if (
                message is not None
                and message.shape == residual[node_type].shape
            ):
                fused = residual[node_type] + message
            else:
                # No real message for this type — keep residual (avoid 2x features).
                fused = residual[node_type]
            if node_type == "operation":
                fused = self.norm_operation(fused)
            else:
                fused = self.norm_machine(fused)
            fused = F.gelu(fused)
            fused = self.dropout_layer(fused)
            updated[node_type] = fused
        return updated


class GraphEncoder(nn.Module):
    """Encode FJSP ``HeteroData`` into machine, operation, and graph embeddings.

    Architecture:
        Linear input projections -> edge-aware bidirectional TransformerConv
        blocks -> attentional pooling over operations and machines -> graph MLP.

    Args:
        operation_in_dim: Operation node feature dimension (default 12).
        machine_in_dim: Machine node feature dimension (default 3).
        hidden_dim: Latent width for node embeddings.
        num_layers: Number of heterogeneous residual blocks.
        num_heads: Number of TransformerConv attention heads.
        dropout: Dropout probability after each block.
    """

    def __init__(
        self,
        operation_in_dim: int = 12,
        machine_in_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.operation_in_dim = int(operation_in_dim)
        self.machine_in_dim = int(machine_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.operation_encoder = nn.Sequential(
            nn.Linear(self.operation_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(self.machine_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.layers = nn.ModuleList(
            [
                HeteroResidualBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.operation_pool = AttentionalAggregation(
            gate_nn=nn.Linear(hidden_dim, 1),
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
        )
        self.machine_pool = AttentionalAggregation(
            gate_nn=nn.Linear(hidden_dim, 1),
            nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
        )
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _project_nodes(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        if "operation" not in data.node_types or data["operation"].x is None:
            raise ValueError("HeteroData is missing operation node features")
        if "machine" not in data.node_types or data["machine"].x is None:
            raise ValueError("HeteroData is missing machine node features")

        op_x = data["operation"].x.float()
        mach_x = data["machine"].x.float()

        if op_x.size(-1) != self.operation_in_dim:
            raise ValueError(
                f"Expected operation features of dim {self.operation_in_dim}, "
                f"got {op_x.size(-1)}"
            )
        if mach_x.size(-1) != self.machine_in_dim:
            raise ValueError(
                f"Expected machine features of dim {self.machine_in_dim}, "
                f"got {mach_x.size(-1)}"
            )

        op_x, mach_x = self._normalize_time_features(
            op_x,
            mach_x,
            op_batch=getattr(data["operation"], "batch", None),
            mach_batch=getattr(data["machine"], "batch", None),
        )
        return {
            "operation": self.operation_encoder(op_x),
            "machine": self.machine_encoder(mach_x),
        }

    @staticmethod
    def _normalize_time_features(
        op_x: torch.Tensor,
        mach_x: torch.Tensor,
        op_batch: torch.Tensor | None = None,
        mach_batch: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-graph mean duration scale so collation cannot mix instance units."""
        op_x = op_x.clone()
        mach_x = mach_x.clone()
        if op_batch is None:
            scale = op_x[:, OP_DURATION].mean().clamp(min=1.0)
            op_x[:, list(_OP_TIME_COLS)] = op_x[:, list(_OP_TIME_COLS)] / scale
            mach_x[:, MACH_WORKLOAD] = mach_x[:, MACH_WORKLOAD] / scale
            mach_x[:, MACH_IDLE_DURATION] = mach_x[:, MACH_IDLE_DURATION] / scale
            return op_x, mach_x

        n_graph = int(op_batch.max().item()) + 1
        dur = op_x[:, OP_DURATION]
        sums = torch.zeros(n_graph, device=op_x.device, dtype=op_x.dtype)
        counts = torch.zeros(n_graph, device=op_x.device, dtype=op_x.dtype)
        sums.scatter_add_(0, op_batch, dur)
        counts.scatter_add_(0, op_batch, torch.ones_like(dur))
        scale = (sums / counts.clamp(min=1.0)).clamp(min=1.0)
        op_scale = scale[op_batch].unsqueeze(1)
        op_x[:, list(_OP_TIME_COLS)] = op_x[:, list(_OP_TIME_COLS)] / op_scale
        if mach_batch is None:
            mach_scale = scale.mean()
        else:
            mach_scale = scale[mach_batch]
        mach_x[:, MACH_WORKLOAD] = mach_x[:, MACH_WORKLOAD] / mach_scale
        mach_x[:, MACH_IDLE_DURATION] = mach_x[:, MACH_IDLE_DURATION] / mach_scale
        return op_x, mach_x

    def _message_edges(
        self, data: HeteroData
    ) -> Tuple[
        Dict[Tuple[str, str, str], torch.Tensor],
        Dict[Tuple[str, str, str], torch.Tensor],
    ]:
        device = data["operation"].x.device
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for edge_type in EDGE_TYPES:
            edge_index = None
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.zeros((0, 1), dtype=torch.float32, device=device)
            else:
                edge_attr = getattr(data[edge_type], "edge_attr", None)
                if edge_attr is None:
                    edge_attr = torch.zeros(
                        (edge_index.size(1), 1),
                        dtype=torch.float32,
                        device=edge_index.device,
                    )
                else:
                    edge_attr = edge_attr.float().reshape(edge_index.size(1), -1)
                    if edge_attr.size(1) != 1:
                        raise ValueError(
                            f"{edge_type} edge_attr must have one feature, "
                            f"got {edge_attr.size(1)}"
                        )
            edge_index_dict[edge_type] = edge_index
            edge_attr_dict[edge_type] = edge_attr
            reverse_type = REVERSE_EDGE_TYPES[edge_type]
            edge_index_dict[reverse_type] = edge_index.flip(0)
            edge_attr_dict[reverse_type] = edge_attr
        return edge_index_dict, edge_attr_dict

    def _encode_x_dict(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict

    def _pool_graph(
        self,
        operation_emb: torch.Tensor,
        machine_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.graph_mlp(
            torch.cat(
                [
                    self.operation_pool(operation_emb).reshape(-1),
                    self.machine_pool(machine_emb).reshape(-1),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def _batchable_view(data: HeteroData) -> HeteroData:
        """Minimal HeteroData copy for Batch collate (local tensors only)."""
        view = HeteroData()
        view["operation"].x = data["operation"].x
        view["machine"].x = data["machine"].x
        device = data["operation"].x.device
        for edge_type in EDGE_TYPES:
            edge_index = None
            edge_attr = None
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                edge_attr = getattr(data[edge_type], "edge_attr", None)
            if edge_index is None or edge_index.numel() == 0:
                view[edge_type].edge_index = torch.empty(
                    (2, 0), dtype=torch.long, device=device
                )
                view[edge_type].edge_attr = torch.zeros(
                    (0, 1), dtype=torch.float32, device=device
                )
            else:
                view[edge_type].edge_index = edge_index
                if edge_attr is None:
                    view[edge_type].edge_attr = torch.zeros(
                        (edge_index.size(1), 1),
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    view[edge_type].edge_attr = edge_attr
        return view

    def _encode_serial(
        self, graphs: List[HeteroData]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        machine_list: List[torch.Tensor] = []
        operation_list: List[torch.Tensor] = []
        graph_list: List[torch.Tensor] = []
        for graph in graphs:
            machine_emb, operation_emb, graph_emb = self.forward(graph)
            machine_list.append(machine_emb)
            operation_list.append(operation_emb)
            graph_list.append(graph_emb)
        return machine_list, operation_list, torch.stack(graph_list, dim=0)

    def forward(
        self, data: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single FJSP graph without mutating the caller-owned graph.

        Args:
            data: Environment ``HeteroData`` state.

        Returns:
            ``(machine_embeddings, operation_embeddings, graph_embedding)`` where
            machine/operation embeddings have shape ``(N, hidden_dim)`` and
            ``graph_embedding`` has shape ``(hidden_dim,)``.
        """
        device = next(self.parameters()).device
        # Move local views only — never call data.to(device) in-place.
        local = HeteroData()
        local["operation"].x = data["operation"].x.float().to(device)
        local["machine"].x = data["machine"].x.float().to(device)
        for edge_type in EDGE_TYPES:
            if edge_type not in data.edge_types:
                continue
            edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                continue
            local[edge_type].edge_index = edge_index.to(device)
            edge_attr = getattr(data[edge_type], "edge_attr", None)
            if edge_attr is not None:
                local[edge_type].edge_attr = edge_attr.to(device)

        x_dict = self._project_nodes(local)
        edge_index_dict, edge_attr_dict = self._message_edges(local)
        x_dict = self._encode_x_dict(x_dict, edge_index_dict, edge_attr_dict)

        operation_emb = x_dict["operation"]
        machine_emb = x_dict["machine"]
        graph_emb = self._pool_graph(operation_emb, machine_emb)
        return machine_emb, operation_emb, graph_emb

    def encode_batch(
        self, graphs: List[HeteroData]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Encode a list of graphs (for SB3 minibatches with fixed action dims).

        Returns:
            ``(machine_emb_list, operation_emb_list, graph_emb_batch)`` where
            ``graph_emb_batch`` has shape ``(batch_size, hidden_dim)``.
        """
        if not graphs:
            raise ValueError("encode_batch received an empty graph list")

        n_ops = [int(g["operation"].x.size(0)) for g in graphs]
        n_mach = [int(g["machine"].x.size(0)) for g in graphs]
        same_size = len(graphs) > 1 and len(set(n_ops)) == 1 and len(set(n_mach)) == 1
        if not same_size:
            return self._encode_serial(graphs)

        device = next(self.parameters()).device
        batch_graphs = [self._batchable_view(g) for g in graphs]
        try:
            batch = Batch.from_data_list(batch_graphs)
            # Move the collated batch, not the caller-owned graphs.
            batch = batch.to(device)
        except (KeyError, RuntimeError, ValueError, TypeError) as exc:
            # Only fall back for recognized collation / schema mismatches.
            msg = str(exc).lower()
            if not any(
                token in msg
                for token in ("edge_attr", "keyerror", "size", "match", "batch", "collat")
            ) and not isinstance(exc, (KeyError, TypeError)):
                raise
            return self._encode_serial(graphs)

        x_dict = self._project_nodes(batch)
        edge_index_dict, edge_attr_dict = self._message_edges(batch)
        x_dict = self._encode_x_dict(x_dict, edge_index_dict, edge_attr_dict)

        operation_emb = x_dict["operation"]
        machine_emb = x_dict["machine"]
        op_ptr = batch["operation"].ptr
        mach_ptr = batch["machine"].ptr
        n_graph = len(graphs)
        graph_emb_batch = self.graph_mlp(
            torch.cat(
                [
                    self.operation_pool(
                        operation_emb,
                        index=batch["operation"].batch,
                        dim_size=n_graph,
                    ),
                    self.machine_pool(
                        machine_emb,
                        index=batch["machine"].batch,
                        dim_size=n_graph,
                    ),
                ],
                dim=-1,
            )
        )
        machine_list = [
            machine_emb[mach_ptr[i] : mach_ptr[i + 1]] for i in range(n_graph)
        ]
        operation_list = [
            operation_emb[op_ptr[i] : op_ptr[i + 1]] for i in range(n_graph)
        ]
        return machine_list, operation_list, graph_emb_batch


__all__ = [
    "EDGE_TYPES",
    "MESSAGE_EDGE_TYPES",
    "REVERSE_EDGE_TYPES",
    "GraphEncoder",
    "HeteroResidualBlock",
]

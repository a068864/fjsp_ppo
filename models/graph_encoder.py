"""Heterogeneous graph encoder for FJSP states."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from utils import get_logger

logger = get_logger(__name__)

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("operation", "precede", "operation"),
    ("operation", "next", "operation"),
    ("machine", "processing", "operation"),
    ("operation", "compatible", "machine"),
]


class HeteroResidualBlock(nn.Module):
    """One HeteroConv layer with residual connection, LayerNorm, and Dropout."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        convs: Dict[Tuple[str, str, str], nn.Module] = {}
        for edge_type in EDGE_TYPES:
            src, _, dst = edge_type
            if src == dst:
                convs[edge_type] = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            else:
                convs[edge_type] = SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    aggr="mean",
                )

        self.conv = HeteroConv(convs, aggr=aggr)
        self.norm_operation = nn.LayerNorm(hidden_dim)
        self.norm_machine = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        residual = {key: value for key, value in x_dict.items()}
        filtered_edges = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if edge_type in self.conv.convs
            and edge_index is not None
            and edge_index.numel() > 0
            and edge_index.dim() == 2
            and edge_index.size(0) == 2
            and edge_index.size(1) > 0
        }

        if filtered_edges:
            out = self.conv(x_dict, filtered_edges)
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
        Linear input projections -> stacked HeteroConv residual blocks ->
        global mean pooling over operations and machines -> graph MLP.

    Args:
        operation_in_dim: Operation node feature dimension (default 10).
        machine_in_dim: Machine node feature dimension (default 3).
        hidden_dim: Latent width for node embeddings.
        num_layers: Number of heterogeneous residual blocks.
        num_heads: Kept for config compatibility (SAGEConv path does not use heads).
        dropout: Dropout probability after each block.
    """

    def __init__(
        self,
        operation_in_dim: int = 10,
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
                HeteroResidualBlock(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(self.num_layers)
            ]
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

        return {
            "operation": self.operation_encoder(op_x),
            "machine": self.machine_encoder(mach_x),
        }

    def _edge_index_dict(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for edge_type in EDGE_TYPES:
            if edge_type in data.edge_types:
                edge_index = data[edge_type].edge_index
                if edge_index is not None and edge_index.numel() > 0:
                    edge_index_dict[edge_type] = edge_index
        return edge_index_dict

    def _encode_x_dict(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict

    @staticmethod
    def _batchable_view(data: HeteroData) -> HeteroData:
        """Minimal HeteroData copy for Batch collate (local tensors only)."""
        view = HeteroData()
        view["operation"].x = data["operation"].x
        view["machine"].x = data["machine"].x
        for edge_type in EDGE_TYPES:
            if edge_type not in data.edge_types:
                continue
            edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                continue
            view[edge_type].edge_index = edge_index
            edge_attr = getattr(data[edge_type], "edge_attr", None)
            if edge_attr is not None:
                view[edge_type].edge_attr = edge_attr
        return view

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
        local["operation"].x = data["operation"].x.float().to(device, copy=True)
        local["machine"].x = data["machine"].x.float().to(device, copy=True)
        for edge_type in EDGE_TYPES:
            if edge_type not in data.edge_types:
                continue
            edge_index = data[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                continue
            local[edge_type].edge_index = edge_index.to(device, copy=True)
            edge_attr = getattr(data[edge_type], "edge_attr", None)
            if edge_attr is not None:
                local[edge_type].edge_attr = edge_attr.to(device, copy=True)

        x_dict = self._project_nodes(local)
        edge_index_dict = self._edge_index_dict(local)
        x_dict = self._encode_x_dict(x_dict, edge_index_dict)

        operation_emb = x_dict["operation"]
        machine_emb = x_dict["machine"]
        graph_emb = self.graph_mlp(
            torch.cat([operation_emb.mean(dim=0), machine_emb.mean(dim=0)], dim=-1)
        )
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

        if len(graphs) == 1:
            machine_emb, operation_emb, graph_emb = self.forward(graphs[0])
            return [machine_emb], [operation_emb], graph_emb.unsqueeze(0)

        n_ops = [int(g["operation"].x.size(0)) for g in graphs]
        n_mach = [int(g["machine"].x.size(0)) for g in graphs]
        same_size = len(set(n_ops)) == 1 and len(set(n_mach)) == 1

        if not same_size:
            machine_list: List[torch.Tensor] = []
            operation_list: List[torch.Tensor] = []
            graph_list: List[torch.Tensor] = []
            for graph in graphs:
                machine_emb, operation_emb, graph_emb = self.forward(graph)
                machine_list.append(machine_emb)
                operation_list.append(operation_emb)
                graph_list.append(graph_emb)
            return machine_list, operation_list, torch.stack(graph_list, dim=0)

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
            machine_list = []
            operation_list = []
            graph_list = []
            for graph in graphs:
                machine_emb, operation_emb, graph_emb = self.forward(graph)
                machine_list.append(machine_emb)
                operation_list.append(operation_emb)
                graph_list.append(graph_emb)
            return machine_list, operation_list, torch.stack(graph_list, dim=0)

        x_dict = self._project_nodes(batch)
        edge_index_dict = self._edge_index_dict(batch)
        x_dict = self._encode_x_dict(x_dict, edge_index_dict)

        operation_emb = x_dict["operation"]
        machine_emb = x_dict["machine"]
        # Prefer PyG pointers over repeated whole-batch boolean scans.
        op_ptr = batch["operation"].ptr
        mach_ptr = batch["machine"].ptr

        machine_list = []
        operation_list = []
        graph_list = []
        for i in range(len(graphs)):
            op_i = operation_emb[op_ptr[i] : op_ptr[i + 1]]
            mach_i = machine_emb[mach_ptr[i] : mach_ptr[i + 1]]
            operation_list.append(op_i)
            machine_list.append(mach_i)
            graph_list.append(
                self.graph_mlp(torch.cat([op_i.mean(dim=0), mach_i.mean(dim=0)], dim=-1))
            )
        return machine_list, operation_list, torch.stack(graph_list, dim=0)


__all__ = ["EDGE_TYPES", "GraphEncoder", "HeteroResidualBlock"]

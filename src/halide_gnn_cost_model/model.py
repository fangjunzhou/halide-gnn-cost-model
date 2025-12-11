"""Model definitions for Halide pipeline runtime prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GPSConv, SAGEConv

try:  # PyG optional dependency when running type checkers.
    from torch_geometric.data import HeteroData
except ImportError:  # pragma: no cover - allows documentation builds without PyG
    HeteroData = object  # type: ignore


__all__ = ["PipeGCN", "PipeGAT", "PipeGPS", "PipelineModel"]


class PipeGCN(nn.Module):
    """A lightweight GraphSAGE-style encoder for heterogenous graphs."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: str = "add",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("PipeGCN requires at least two layers.")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(-1, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(-1, hidden_channels, aggr=aggr))
        self.convs.append(SAGEConv(-1, out_channels, aggr=aggr))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class PipeGAT(nn.Module):
    """A multi-layer GAT encoder suitable for heterogenous graphs."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.5,
        add_self_loops: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("PipeGAT requires at least two layers.")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                -1,
                hidden_channels,
                heads=heads,
                concat=False,
                add_self_loops=add_self_loops,
            )
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    -1,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    add_self_loops=add_self_loops,
                )
            )
        self.convs.append(
            GATConv(
                -1,
                out_channels,
                heads=heads,
                concat=False,
                add_self_loops=add_self_loops,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class PipeGPS(nn.Module):
    """Graph transformer-style encoder built with GPSConv blocks."""

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        num_attn_heads: int,
        vocab_size: int,
        num_runtime_targets: int,
        pe_dim: int = 24,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        schedule_node_type: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("PipeGPS requires at least one GPSConv layer.")
        if hidden_channels <= pe_dim:
            raise ValueError("hidden_channels must exceed pe_dim for feature fusion.")

        self.pe_dim = pe_dim
        self.dropout = dropout
        self.schedule_node_type = schedule_node_type

        node_embedding_dim = hidden_channels - pe_dim
        self.node_emb = nn.Embedding(vocab_size, node_embedding_dim)
        self.pe_lin = nn.Linear(pe_dim, pe_dim)
        self.pe_norm = nn.BatchNorm1d(pe_dim)

        attn_kwargs = attn_kwargs or {}
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(
                    hidden_channels,
                    None,
                    heads=num_attn_heads,
                    attn_type=attn_type,
                    attn_kwargs=attn_kwargs,
                )
            )

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, num_runtime_targets)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.pe_lin.weight)
        if self.pe_lin.bias is not None:
            nn.init.zeros_(self.pe_lin.bias)
        self.pe_norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def forward(
        self,
        node_tokens: Tensor,
        positional_encoding: Tensor,
        node_type: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        node_tokens = node_tokens.view(-1).long()
        positional_encoding = positional_encoding.view(-1, self.pe_dim)
        node_type = node_type.view(-1)

        x_pe = self.pe_norm(positional_encoding)
        x = torch.cat((self.node_emb(node_tokens), self.pe_lin(x_pe)), dim=1)

        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)

        schedule_mask = node_type.eq(self.schedule_node_type)

        if schedule_mask.any():
            x = x[schedule_mask]
            if batch is not None:
                batch = batch.view(-1)[schedule_mask]
                num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
                pooled = x.new_zeros((num_graphs, x.size(-1)))
                pooled.index_add_(0, batch, x)
                x = pooled
            else:
                x = x.sum(dim=0, keepdim=True)
        else:
            num_graphs = (
                int(batch.max().item()) + 1
                if batch is not None and batch.numel() > 0
                else 1
            )
            x = x.new_zeros((num_graphs, x.size(-1)))

        x = F.relu(self.fc1(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        return x


class PipelineModel(nn.Module):
    """Predicts runtime from Halide pipeline graphs using a node encoder."""

    def __init__(
        self,
        gnn: nn.Module,
        feat_channels: int,
        num_runtime_targets: int,
        ast_vocab_size: int,
        sched_vocab_size: int,
        hidden_channels: int = 32,
        ast_embedding_dim: int = 32,
        sched_embedding_dim: int = 32,
        function_embedding_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_runtime_targets <= 0:
            raise ValueError("num_runtime_targets must be positive.")

        self.function_gnn = gnn
        self.dropout = dropout

        self.ast_embedding = nn.Embedding(ast_vocab_size, ast_embedding_dim)
        self.sched_embedding = nn.Embedding(sched_vocab_size, sched_embedding_dim)
        self.function_embedding = nn.Parameter(torch.empty(function_embedding_dim))
        self.fc1 = nn.Linear(feat_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, num_runtime_targets)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ast_embedding.weight)
        nn.init.xavier_uniform_(self.sched_embedding.weight)
        nn.init.normal_(self.function_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)
        if hasattr(self.function_gnn, "reset_parameters"):
            self.function_gnn.reset_parameters()

    def forward(self, data: Any, ptr: Optional[Tensor] = None) -> Tensor:
        num_functions = data["function"].x.size(0)
        if num_functions > 0:
            data["function"].x = (
                self.function_embedding.unsqueeze(0).expand(num_functions, -1).clone()
            )

        ast_tokens = data["ast_node"].x.view(-1).long()
        data["ast_node"].x = self.ast_embedding(ast_tokens)

        sched_tokens = data["loop_level"].x.view(-1).long()
        data["loop_level"].x = self.sched_embedding(sched_tokens)

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        out_dict = self.function_gnn(x_dict, edge_index_dict)

        loop_level_x = out_dict["loop_level"]
        loop_level_ptr = (
            ptr if ptr is not None else getattr(data["loop_level"], "ptr", None)
        )
        loop_level_batch = getattr(data["loop_level"], "batch", None)
        pipeline_feat = self._pool_loop_features(
            loop_level_x, loop_level_ptr, loop_level_batch
        )

        if self.dropout > 0:
            pipeline_feat = F.dropout(
                pipeline_feat, p=self.dropout, training=self.training
            )

        x = F.relu(self.fc1(pipeline_feat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        log_runtime = self.fc3(x)
        return log_runtime

    @staticmethod
    def _pool_loop_features(
        features: Tensor,
        ptr: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> Tensor:
        """Aggregate loop-level features into a single embedding per pipeline."""
        if features.numel() == 0:
            return features

        if ptr is not None and ptr.numel() > 1:
            counts = ptr[1:] - ptr[:-1]
            num_graphs = int(counts.numel())
            if num_graphs == 0:
                return features
            index = torch.repeat_interleave(
                torch.arange(num_graphs, device=features.device), counts
            )
            pooled = torch.zeros(
                (num_graphs, features.size(-1)),
                device=features.device,
                dtype=features.dtype,
            )
            pooled.index_add_(0, index, features)
            return pooled

        if batch is not None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            pooled = torch.zeros(
                (num_graphs, features.size(-1)),
                device=features.device,
                dtype=features.dtype,
            )
            pooled.index_add_(0, batch, features)
            return pooled

        return features.sum(dim=0, keepdim=True)

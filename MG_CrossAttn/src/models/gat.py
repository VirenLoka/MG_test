#!/usr/bin/env python3
"""
Graph attention transformer over the molecule graph.

Each block is a transformer block whose attention operator is a graph
convolution, so attention is restricted to bonded neighbours instead of being
all-to-all:

    h = h + Dropout(GraphAttn(Norm(h), edge_index, edge_attr))
    h = h + Dropout(FFN(Norm(h)))

Pre-norm is the default because it trains more stably once stacked a few deep.
The attention operator is selected by config: GATv2 (default; fixes the
original GAT's static-attention limitation), GAT, or TransformerConv.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv

CONV_TYPES = {"gat": GATConv, "gatv2": GATv2Conv, "transformer_conv": TransformerConv}
ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "elu": nn.ELU}


class GraphAttentionBlock(nn.Module):
    """One pre-norm (or post-norm) graph-attention transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_type: str = "gatv2",
        edge_dim: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_mult: int = 2,
        norm: str = "pre",
        residual: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        if conv_type not in CONV_TYPES:
            raise ValueError(
                f"unknown conv_type {conv_type!r}; choose from {sorted(CONV_TYPES)}"
            )
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.norm_style = norm
        self.residual = residual

        # concat=True with per-head width d_model//n_heads returns d_model.
        self.conv = CONV_TYPES[conv_type](
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            add_self_loops=False,  # isolated atoms would otherwise be required
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        act = ACTIVATIONS[activation]
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            act(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, h, edge_index, edge_attr=None):
        if self.norm_style == "pre":
            a = self.conv(self.norm1(h), edge_index, edge_attr)
            h = h + self.drop(a) if self.residual else self.drop(a)
            f = self.ffn(self.norm2(h))
            h = h + self.drop(f) if self.residual else self.drop(f)
        else:
            a = self.conv(h, edge_index, edge_attr)
            h = self.norm1(h + self.drop(a)) if self.residual else self.norm1(self.drop(a))
            f = self.ffn(h)
            h = self.norm2(h + self.drop(f)) if self.residual else self.norm2(self.drop(f))
        return h


class GraphAttentionTransformer(nn.Module):
    """
    Stack of graph-attention transformer blocks producing per-atom embeddings.

    Returns [total_atoms, d_model]; pooling to one vector per molecule happens
    downstream, after cross-attention, so the cross-attention can operate at
    atom resolution.
    """

    def __init__(self, atom_dim: int, bond_dim: int, cfg: dict, d_model: int):
        super().__init__()
        use_edges = bool(cfg.get("use_edge_features", True)) and bond_dim > 0
        edge_dim = bond_dim if use_edges else None
        self.use_edges = use_edges

        self.input_proj = nn.Linear(atom_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(cfg.get("dropout", 0.1))

        self.blocks = nn.ModuleList([
            GraphAttentionBlock(
                d_model=d_model,
                n_heads=int(cfg["n_heads"]),
                conv_type=cfg.get("conv_type", "gatv2"),
                edge_dim=edge_dim,
                dropout=cfg.get("dropout", 0.1),
                attention_dropout=cfg.get("attention_dropout", 0.1),
                ffn_mult=int(cfg.get("ffn_mult", 2)),
                norm=cfg.get("norm", "pre"),
                residual=bool(cfg.get("residual", True)),
            )
            for _ in range(int(cfg["n_layers"]))
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.input_drop(self.input_norm(self.input_proj(x)))
        edges = edge_attr if self.use_edges else None
        for block in self.blocks:
            h = block(h, edge_index, edges)
        return self.out_norm(h)


def to_dense_batch(h: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    """
    Scatter flat per-atom features into a padded [B, N_max, D] tensor.

    Returns (dense, pad_mask) with pad_mask True at padding positions, matching
    the key_padding_mask convention of nn.MultiheadAttention.
    """
    device = h.device
    counts = torch.bincount(batch, minlength=n_graphs)
    n_max = int(counts.max().item()) if counts.numel() else 0

    dense = h.new_zeros((n_graphs, n_max, h.shape[-1]))
    pad_mask = torch.ones((n_graphs, n_max), dtype=torch.bool, device=device)

    # Position of each atom within its own molecule.
    order = torch.argsort(batch, stable=True)
    sorted_batch = batch[order]
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(batch.numel(), device=device) - starts[sorted_batch]

    dense[sorted_batch, within] = h[order]
    pad_mask[sorted_batch, within] = False
    return dense, pad_mask

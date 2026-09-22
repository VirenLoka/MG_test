#!/usr/bin/env python3
"""
Bidirectional co-attention between atom embeddings and ESM residue embeddings.

One block updates both streams:

    atoms    = atoms    + FFN(atoms    + CrossAttn(Q=atoms,    KV=residues))
    residues = residues + FFN(residues + CrossAttn(Q=residues, KV=atoms))

`direction` selects which halves run. Both streams carry key-padding masks -
molecules differ in atom count, proteins in residue count - and attention is
never allowed to read a padded position.

A padding row would otherwise attend over an all-masked key set and produce
NaN; padded query rows are therefore zeroed after each attention call rather
than left to propagate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "elu": nn.ELU}
DIRECTIONS = ("bidirectional", "mol_to_prot", "prot_to_mol")


def _ffn(d_model: int, mult: int, dropout: float, activation: str) -> nn.Sequential:
    act = ACTIVATIONS[activation]
    return nn.Sequential(
        nn.Linear(d_model, d_model * mult),
        act(),
        nn.Dropout(dropout),
        nn.Linear(d_model * mult, d_model),
    )


class CrossAttentionSublayer(nn.Module):
    """One direction of cross-attention plus its feed-forward, pre- or post-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        attention_dropout: float,
        ffn_mult: int,
        norm: str,
        residual: bool,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm_style = norm
        self.residual = residual

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = _ffn(d_model, ffn_mult, dropout, activation)

    def forward(self, query, key_value, q_pad_mask, kv_pad_mask):
        """
        query        [B, Nq, D]
        key_value    [B, Nk, D]
        q_pad_mask   [B, Nq]  True at padding
        kv_pad_mask  [B, Nk]  True at padding
        """
        if self.norm_style == "pre":
            q, kv = self.norm_q(query), self.norm_kv(key_value)
        else:
            q, kv = query, key_value

        attended, _ = self.attn(
            query=q, key=kv, value=kv,
            key_padding_mask=kv_pad_mask,
            need_weights=False,
        )
        # Padded queries have no meaningful output; zero them so the residual
        # stream stays clean and no NaN can enter from a fully-masked row.
        attended = attended.masked_fill(q_pad_mask.unsqueeze(-1), 0.0)

        if self.norm_style == "pre":
            h = query + self.drop(attended) if self.residual else self.drop(attended)
            f = self.ffn(self.norm_ffn(h))
            h = h + self.drop(f) if self.residual else self.drop(f)
        else:
            h = self.norm_q(query + self.drop(attended)) if self.residual \
                else self.norm_q(self.drop(attended))
            f = self.ffn(h)
            h = self.norm_ffn(h + self.drop(f)) if self.residual \
                else self.norm_ffn(self.drop(f))

        return h.masked_fill(q_pad_mask.unsqueeze(-1), 0.0)


class CoAttentionBlock(nn.Module):
    """Both cross-attention directions, updated from the same block inputs."""

    def __init__(self, d_model: int, cfg: dict):
        super().__init__()
        direction = cfg.get("direction", "bidirectional")
        if direction not in DIRECTIONS:
            raise ValueError(
                f"unknown direction {direction!r}; choose from {DIRECTIONS}"
            )
        self.direction = direction

        kwargs = dict(
            d_model=d_model,
            n_heads=int(cfg["n_heads"]),
            dropout=cfg.get("dropout", 0.1),
            attention_dropout=cfg.get("attention_dropout", 0.1),
            ffn_mult=int(cfg.get("ffn_mult", 2)),
            norm=cfg.get("norm", "pre"),
            residual=bool(cfg.get("residual", True)),
        )
        self.mol_to_prot = (
            CrossAttentionSublayer(**kwargs)
            if direction in ("bidirectional", "mol_to_prot") else None
        )
        self.prot_to_mol = (
            CrossAttentionSublayer(**kwargs)
            if direction in ("bidirectional", "prot_to_mol") else None
        )

    def forward(self, mol, prot, mol_mask, prot_mask):
        # Both directions read the block's *input* streams, so the update is
        # symmetric and neither direction sees the other's output early.
        new_mol, new_prot = mol, prot
        if self.mol_to_prot is not None:
            new_mol = self.mol_to_prot(mol, prot, mol_mask, prot_mask)
        if self.prot_to_mol is not None:
            new_prot = self.prot_to_mol(prot, mol, prot_mask, mol_mask)
        return new_mol, new_prot


class CoAttentionStack(nn.Module):
    """Stacked co-attention blocks."""

    def __init__(self, d_model: int, cfg: dict):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CoAttentionBlock(d_model, cfg) for _ in range(int(cfg["n_blocks"]))]
        )
        self.mol_norm = nn.LayerNorm(d_model)
        self.prot_norm = nn.LayerNorm(d_model)

    def forward(self, mol, prot, mol_mask, prot_mask):
        for block in self.blocks:
            mol, prot = block(mol, prot, mol_mask, prot_mask)
        mol = self.mol_norm(mol).masked_fill(mol_mask.unsqueeze(-1), 0.0)
        prot = self.prot_norm(prot).masked_fill(prot_mask.unsqueeze(-1), 0.0)
        return mol, prot


class MaskedPooling(nn.Module):
    """
    Pool a padded sequence to one vector, ignoring padding.

    `attention` learns a single query vector and scores each position against
    it, which lets the readout concentrate on the atoms or residues that
    matter instead of averaging them away.
    """

    def __init__(self, d_model: int, mode: str = "attention"):
        super().__init__()
        if mode not in ("attention", "mean", "max", "mean_max"):
            raise ValueError(f"unknown pooling mode {mode!r}")
        self.mode = mode
        self.out_dim = d_model * (2 if mode == "mean_max" else 1)
        if mode == "attention":
            self.score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
            )

    def forward(self, h, pad_mask):
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)   # [B, N, 1]
        n_valid = valid.sum(dim=1).clamp(min=1.0)

        if self.mode == "attention":
            logits = self.score(h).masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
            # A row with no valid positions would softmax to NaN; such rows
            # cannot occur for molecules, but proteins are padded per batch.
            all_pad = pad_mask.all(dim=1, keepdim=True).unsqueeze(-1)
            logits = torch.where(all_pad, torch.zeros_like(logits), logits)
            weights = torch.softmax(logits, dim=1)
            return (h * weights).sum(dim=1)

        if self.mode == "mean":
            return (h * valid).sum(dim=1) / n_valid

        if self.mode == "max":
            return h.masked_fill(pad_mask.unsqueeze(-1), float("-inf")).max(dim=1).values

        mean = (h * valid).sum(dim=1) / n_valid
        mx = h.masked_fill(pad_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        return torch.cat([mean, mx], dim=-1)

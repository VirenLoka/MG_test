#!/usr/bin/env python3
"""
The assembled model: GAT molecule tower + frozen-ESM protein tower + co-attention.

    SMILES -> graph -> GraphAttentionTransformer -> atom embeddings  [B, Na, D]
    gene   -> cached ESM residues -> projection  -> residue emb.     [B, Nr, D]
                              |
                    CoAttentionStack (bidirectional)
                              |
                 masked pooling of both streams
                              |
              [mol ; prot ; mol*prot] -> MLP -> 1 logit

ESM is never instantiated here. Embeddings arrive pre-computed from
`src.esm_embed`, so this module holds no protein language model weights and
the protein tower is a projection only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cross_attention import CoAttentionStack, MaskedPooling
from .gat import GraphAttentionTransformer, to_dense_batch

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "elu": nn.ELU}


class ProteinProjection(nn.Module):
    """Projects cached ESM embeddings to the shared model width."""

    def __init__(self, esm_dim: int, d_model: int, cfg: dict):
        super().__init__()
        kind = cfg.get("projection", "linear")
        if kind == "linear":
            self.proj: nn.Module = nn.Linear(esm_dim, d_model)
        elif kind == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(esm_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            raise ValueError(f"unknown protein projection {kind!r}")

        self.norm = nn.LayerNorm(d_model) if cfg.get("layer_norm", True) else nn.Identity()
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))

        # Strided subsampling of the residue axis; 1651 residues is the worst
        # case and attention over it is the dominant cost in a batch.
        stride = cfg.get("pool_stride")
        self.pool_stride = int(stride) if stride else None

    def forward(self, prot, prot_mask):
        if self.pool_stride and self.pool_stride > 1:
            prot = prot[:, :: self.pool_stride]
            prot_mask = prot_mask[:, :: self.pool_stride]
        h = self.drop(self.norm(self.proj(prot)))
        return h.masked_fill(prot_mask.unsqueeze(-1), 0.0), prot_mask


class Classifier(nn.Module):
    """MLP head over the concatenated pooled representations."""

    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        act = ACTIVATIONS[cfg.get("activation", "gelu")]
        dims = [in_dim] + list(cfg.get("hidden_dims", [256, 128]))

        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(a, b))
            if cfg.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(b))
            layers.append(act())
            layers.append(nn.Dropout(cfg.get("dropout", 0.2)))
        layers.append(nn.Linear(dims[-1], int(cfg.get("n_outputs", 1))))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class GlueTargetCrossAttention(nn.Module):
    """Full model. Call `from_config` rather than constructing directly."""

    def __init__(self, atom_dim: int, bond_dim: int, esm_dim: int, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        d_model = int(model_cfg["d_model"])
        self.d_model = d_model

        self.molecule_tower = GraphAttentionTransformer(
            atom_dim=atom_dim, bond_dim=bond_dim,
            cfg=model_cfg["gat"], d_model=d_model,
        )
        self.protein_tower = ProteinProjection(esm_dim, d_model, model_cfg["protein"])
        self.co_attention = CoAttentionStack(d_model, model_cfg["cross_attention"])

        readout = model_cfg["readout"]
        self.mol_pool = MaskedPooling(d_model, readout.get("mol_pool", "attention"))
        self.prot_pool = MaskedPooling(d_model, readout.get("prot_pool", "attention"))

        self.interactions = list(readout.get("interactions", []) or [])
        unknown = set(self.interactions) - {"product", "difference"}
        if unknown:
            raise ValueError(f"unknown readout interactions: {sorted(unknown)}")

        head_dim = self.mol_pool.out_dim + self.prot_pool.out_dim
        # Interaction terms require the two pooled vectors to be the same width.
        if self.interactions:
            if self.mol_pool.out_dim != self.prot_pool.out_dim:
                raise ValueError(
                    "readout.interactions needs mol_pool and prot_pool to produce "
                    f"equal widths, got {self.mol_pool.out_dim} and "
                    f"{self.prot_pool.out_dim}"
                )
            head_dim += self.mol_pool.out_dim * len(self.interactions)

        self.classifier = Classifier(head_dim, model_cfg["classifier"])
        self.head_dim = head_dim

    @classmethod
    def from_config(cls, cfg: dict, atom_dim: int, bond_dim: int):
        return cls(atom_dim, bond_dim, int(cfg["esm"]["embed_dim"]), cfg)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns logits of shape [B] for binary classification."""
        atoms = self.molecule_tower(
            batch["x"], batch["edge_index"], batch.get("edge_attr")
        )
        mol, mol_mask = to_dense_batch(atoms, batch["batch"], batch["n_graphs"])

        # Proteins are stored once per distinct target in the batch; expand to
        # one row per sample so cross-attention lines up with the molecules.
        prot = batch["prot"].index_select(0, batch["prot_index"])
        prot_mask = batch["prot_mask"].index_select(0, batch["prot_index"])
        prot, prot_mask = self.protein_tower(prot, prot_mask)

        mol, prot = self.co_attention(mol, prot, mol_mask, prot_mask)

        mol_vec = self.mol_pool(mol, mol_mask)
        prot_vec = self.prot_pool(prot, prot_mask)

        parts = [mol_vec, prot_vec]
        if "product" in self.interactions:
            parts.append(mol_vec * prot_vec)
        if "difference" in self.interactions:
            parts.append(mol_vec - prot_vec)

        return self.classifier(torch.cat(parts, dim=-1)).squeeze(-1)

    def n_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def parameter_summary(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "molecule_tower": count(self.molecule_tower),
            "protein_tower": count(self.protein_tower),
            "co_attention": count(self.co_attention),
            "pooling": count(self.mol_pool) + count(self.prot_pool),
            "classifier": count(self.classifier),
            "total": self.n_parameters(trainable_only=False),
        }

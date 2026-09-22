#!/usr/bin/env python3
"""
Dataset and collate for (molecule graph, protein embedding, label) triples.

Molecule graphs are batched in the flat PyG style: node features concatenated
along dim 0, edge indices offset per molecule, and a `batch` vector mapping
each node to its molecule. Protein embeddings are padded to the longest
sequence in the batch with a boolean key-padding mask, so cross-attention
never attends to padding.

One row's protein is a whole ESM tensor, and proteins repeat heavily (27
distinct proteins over thousands of rows). Collate therefore embeds each
distinct protein in a batch once and indexes into that, rather than copying a
1651x1280 tensor per row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .featurize import MoleculeFeaturizer, MoleculeGraph


class GlueTargetDataset(Dataset):
    """
    Rows of (SMILES, target gene symbol, binary label).

    Rows whose SMILES RDKit cannot featurise, or whose target has no cached
    embedding, are dropped at construction and counted in `self.report`.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        featurizer: MoleculeFeaturizer,
        embedding_store,
        smiles_col: str,
        target_col: str,
        label_col: str,
        cache_graphs: bool = True,
    ):
        self.featurizer = featurizer
        self.store = embedding_store
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.label_col = label_col
        self.cache_graphs = cache_graphs

        smiles = frame[smiles_col].astype(str).tolist()
        targets = frame[target_col].astype(str).tolist()
        labels = frame[label_col].astype(int).tolist()

        keep, graphs = [], []
        n_bad_smiles = n_missing_protein = 0
        for i, (smi, tgt) in enumerate(zip(smiles, targets)):
            if tgt not in self.store:
                n_missing_protein += 1
                continue
            graph = featurizer(smi)
            if graph is None:
                n_bad_smiles += 1
                continue
            keep.append(i)
            graphs.append(graph if cache_graphs else None)

        self.smiles = [smiles[i] for i in keep]
        self.targets = [targets[i] for i in keep]
        self.labels = np.asarray([labels[i] for i in keep], dtype=np.float32)
        self._graphs = graphs
        self.frame = frame.iloc[keep].reset_index(drop=True)

        self.report = {
            "rows_in": int(len(frame)),
            "rows_kept": int(len(keep)),
            "dropped_bad_smiles": n_bad_smiles,
            "dropped_missing_protein": n_missing_protein,
            "positive_rate": round(float(self.labels.mean()), 4) if len(keep) else None,
        }

    def __len__(self) -> int:
        return len(self.smiles)

    def graph(self, idx: int) -> MoleculeGraph:
        if self.cache_graphs:
            return self._graphs[idx]
        graph = self.featurizer(self.smiles[idx])
        if graph is None:  # pragma: no cover - filtered at construction
            raise RuntimeError(f"SMILES became unparseable: {self.smiles[idx]}")
        return graph

    def __getitem__(self, idx: int) -> dict:
        return {
            "graph": self.graph(idx),
            "target": self.targets[idx],
            "label": float(self.labels[idx]),
            "index": idx,
        }

    @property
    def pos_weight(self) -> float:
        """n_negative / n_positive, for BCEWithLogitsLoss class balancing."""
        pos = float(self.labels.sum())
        neg = float(len(self.labels) - pos)
        return (neg / pos) if pos > 0 else 1.0


class Collator:
    """
    Builds a batch dict from a list of dataset items.

    Returns:
        x            [total_atoms, atom_dim]
        edge_index   [2, total_edges]      already offset per molecule
        edge_attr    [total_edges, bond_dim]
        batch        [total_atoms]         molecule index per atom
        n_graphs     int
        prot         [n_unique_prot, L_max, esm_dim]
        prot_mask    [n_unique_prot, L_max]  True where padding
        prot_index   [n_graphs]            row -> index into prot
        label        [n_graphs]
    """

    def __init__(self, embedding_store, esm_dim: int):
        self.store = embedding_store
        self.esm_dim = int(esm_dim)

    def __call__(self, items: list[dict]) -> dict:
        graphs = [it["graph"] for it in items]

        xs, edge_indices, edge_attrs, batch_vec = [], [], [], []
        offset = 0
        for i, g in enumerate(graphs):
            xs.append(g.x)
            if g.edge_index.numel():
                edge_indices.append(g.edge_index + offset)
                edge_attrs.append(g.edge_attr)
            batch_vec.append(torch.full((g.n_atoms,), i, dtype=torch.long))
            offset += g.n_atoms

        x = torch.cat(xs, dim=0)
        batch = torch.cat(batch_vec, dim=0)
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_attr = torch.cat(edge_attrs, dim=0)
        else:
            bond_dim = graphs[0].edge_attr.shape[1]
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, bond_dim), dtype=torch.float32)

        # Embed each distinct protein once; rows index into it.
        unique_targets = sorted({it["target"] for it in items})
        slot = {t: i for i, t in enumerate(unique_targets)}
        tensors = [self.store.get(t) for t in unique_targets]
        lengths = [t.shape[0] for t in tensors]
        max_len = max(lengths)

        prot = torch.zeros(len(tensors), max_len, self.esm_dim, dtype=torch.float32)
        prot_mask = torch.ones(len(tensors), max_len, dtype=torch.bool)
        for i, (tensor, length) in enumerate(zip(tensors, lengths)):
            prot[i, :length] = tensor.to(torch.float32)
            prot_mask[i, :length] = False  # False = real residue, True = padding

        prot_index = torch.tensor([slot[it["target"]] for it in items], dtype=torch.long)
        label = torch.tensor([it["label"] for it in items], dtype=torch.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "batch": batch,
            "n_graphs": len(graphs),
            "prot": prot,
            "prot_mask": prot_mask,
            "prot_index": prot_index,
            "label": label,
            "targets": [it["target"] for it in items],
            "row_index": [it["index"] for it in items],
        }


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in a batch dict; leave python objects alone."""
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

#!/usr/bin/env python3
"""
Molecule -> attributed graph, for the graph attention transformer.

Atom and bond feature blocks are selected by name from the config, and the
resulting input dimensions are derived from that selection rather than being
written down anywhere. Dropping a name from `featurizer.atom_features` is a
complete ablation: the tensor narrows and the model's input layer follows.

Bonds are emitted in both directions so message passing is symmetric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Fixed vocabularies for the one-hot blocks. Each gets a trailing "other" slot,
# so an unexpected value never silently collides with a real category.
DEGREE_VOCAB = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_VOCAB = [-2, -1, 0, 1, 2]
NUM_HS_VOCAB = [0, 1, 2, 3, 4]
HYBRIDIZATION_VOCAB = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITY_VOCAB = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO_VOCAB = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def one_hot(value, vocab: list) -> list[float]:
    """One-hot with a trailing slot for anything outside `vocab`."""
    out = [0.0] * (len(vocab) + 1)
    try:
        out[vocab.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


@dataclass
class MoleculeGraph:
    """One molecule as flat tensors, ready to be batched."""

    x: torch.Tensor           # [n_atoms, atom_dim]
    edge_index: torch.Tensor  # [2, n_edges]
    edge_attr: torch.Tensor   # [n_edges, bond_dim]

    @property
    def n_atoms(self) -> int:
        return int(self.x.shape[0])


class MoleculeFeaturizer:
    """Builds MoleculeGraph objects according to a featurizer config block."""

    def __init__(self, cfg: dict):
        self.atom_features: list[str] = list(cfg["atom_features"])
        self.bond_features: list[str] = list(cfg["bond_features"])
        self.atomic_num_vocab: list[int] = list(cfg["atomic_num_vocab"])
        self.explicit_hydrogens: bool = bool(cfg.get("explicit_hydrogens", False))
        self.add_self_loops: bool = bool(cfg.get("add_self_loops", False))

        unknown_atom = set(self.atom_features) - set(self._ATOM_BLOCKS)
        unknown_bond = set(self.bond_features) - set(self._BOND_BLOCKS)
        if unknown_atom:
            raise ValueError(f"unknown atom features: {sorted(unknown_atom)}")
        if unknown_bond:
            raise ValueError(f"unknown bond features: {sorted(unknown_bond)}")

    # -- per-block builders; each returns a list of floats ------------------
    @property
    def _ATOM_BLOCKS(self) -> dict:
        return {
            "atomic_num": lambda a: one_hot(a.GetAtomicNum(), self.atomic_num_vocab),
            "degree": lambda a: one_hot(a.GetTotalDegree(), DEGREE_VOCAB),
            "formal_charge": lambda a: one_hot(a.GetFormalCharge(), FORMAL_CHARGE_VOCAB),
            "num_hs": lambda a: one_hot(a.GetTotalNumHs(), NUM_HS_VOCAB),
            "hybridization": lambda a: one_hot(a.GetHybridization(), HYBRIDIZATION_VOCAB),
            "chirality": lambda a: one_hot(a.GetChiralTag(), CHIRALITY_VOCAB),
            "aromatic": lambda a: [float(a.GetIsAromatic())],
            "in_ring": lambda a: [float(a.IsInRing())],
            "mass": lambda a: [a.GetMass() * 0.01],
        }

    @property
    def _BOND_BLOCKS(self) -> dict:
        return {
            "bond_type": lambda b: one_hot(b.GetBondType(), BOND_TYPE_VOCAB),
            "conjugated": lambda b: [float(b.GetIsConjugated())],
            "in_ring": lambda b: [float(b.IsInRing())],
            "stereo": lambda b: one_hot(b.GetStereo(), BOND_STEREO_VOCAB),
        }

    @property
    def atom_dim(self) -> int:
        """Derived from the selected blocks, using a probe molecule."""
        probe = Chem.MolFromSmiles("CC")
        return len(self._atom_vector(probe.GetAtomWithIdx(0)))

    @property
    def bond_dim(self) -> int:
        probe = Chem.MolFromSmiles("CC")
        return len(self._bond_vector(probe.GetBondWithIdx(0)))

    def _atom_vector(self, atom) -> list[float]:
        blocks = self._ATOM_BLOCKS
        out: list[float] = []
        for name in self.atom_features:
            out.extend(blocks[name](atom))
        return out

    def _bond_vector(self, bond) -> list[float]:
        blocks = self._BOND_BLOCKS
        out: list[float] = []
        for name in self.bond_features:
            out.extend(blocks[name](bond))
        return out

    def __call__(self, smiles: str) -> MoleculeGraph | None:
        """Featurise one SMILES, or return None if RDKit cannot parse it."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if self.explicit_hydrogens:
            mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() == 0:
            return None

        x = np.array(
            [self._atom_vector(a) for a in mol.GetAtoms()], dtype=np.float32
        )

        src, dst, attrs = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            vec = self._bond_vector(bond)
            # Both directions, sharing the bond's feature vector.
            src += [i, j]
            dst += [j, i]
            attrs += [vec, vec]

        if self.add_self_loops:
            zero = [0.0] * self.bond_dim
            for i in range(mol.GetNumAtoms()):
                src.append(i)
                dst.append(i)
                attrs.append(zero)

        if not src:
            # A single-atom molecule has no bonds; emit an empty edge set of
            # the right shape so downstream code needs no special case.
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.bond_dim), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(np.array(attrs, dtype=np.float32))

        return MoleculeGraph(
            x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr
        )

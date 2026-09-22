#!/usr/bin/env python3
"""
Featurisation: MACCS structural keys for the glue + a target descriptor block.

The feature vector for one (molecule, target) row is the concatenation of
  - 166 MACCS key bits (bit 0 is dropped; it is a padding bit that is never set)
  - a target block, chosen by `target_encoding`:
      "onehot" - indicator over the target gene symbols seen in training
      "aac"    - 20 amino-acid composition fractions of the target sequence

The two encodings differ in what they can generalise to. A one-hot block is
fitted on training targets, so a target seen only in val/test becomes an
all-zero block; it can never represent an unseen target. AAC is computed
directly from the sequence, so an unseen target still receives a meaningful
vector — which is what makes a target-disjoint evaluation possible at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys

# RDKit is chatty about sanitisation on patent-derived SMILES; we report
# failures ourselves from the returned mask.
RDLogger.DisableLog("rdApp.*")

MACCS_N_BITS = 167  # RDKit's MACCS output width, including the unused bit 0

# The 20 standard proteinogenic amino acids, in the conventional order.
# Non-standard residue codes (X, U, B, Z, O) are excluded from both the counts
# and the denominator, so the 20 fractions always sum to 1.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def maccs_matrix(smiles: pd.Series, drop_bit0: bool = True):
    """
    MACCS keys for a series of SMILES.

    Returns (matrix, valid_mask, bit_names). Rows whose SMILES RDKit cannot
    parse are all-zero in the matrix and False in the mask; the caller decides
    whether to drop them.
    """
    n_bits = MACCS_N_BITS
    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    valid = np.ones(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid[i] = False
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        on = list(fp.GetOnBits())
        if on:
            matrix[i, on] = 1

    start = 1 if drop_bit0 else 0
    bit_names = [f"MACCS_{b}" for b in range(start, n_bits)]
    return matrix[:, start:], valid, bit_names


def aac_vector(sequence: str) -> np.ndarray:
    """Fraction of each standard amino acid in one sequence (sums to 1)."""
    seq = "".join(c for c in str(sequence).upper() if c in AMINO_ACIDS)
    if not seq:
        raise ValueError("sequence contains no standard amino acids")
    n = len(seq)
    return np.array([seq.count(aa) / n for aa in AMINO_ACIDS], dtype=np.float32)


def aac_matrix(sequences: pd.Series):
    """
    AAC block for a series of sequences.

    Each distinct sequence is computed once and broadcast; there are only a
    few dozen targets but thousands of rows. Returns (matrix, feature_names).
    """
    cache = {seq: aac_vector(seq) for seq in pd.unique(sequences)}
    matrix = np.vstack([cache[seq] for seq in sequences])
    names = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    return matrix, names


def encode_targets(train_targets: pd.Series, *split_targets: pd.Series):
    """
    One-hot the target gene symbol, with the vocabulary taken from training.

    Returns (matrices, feature_names, unseen) where `matrices` lines up with
    the positional arguments (train first, then each extra split) and `unseen`
    maps split index -> the targets in that split absent from training.
    """
    vocab = sorted(train_targets.unique().tolist())
    index = {t: i for i, t in enumerate(vocab)}
    names = [f"target_{t}" for t in vocab]

    matrices, unseen = [], {}
    for pos, series in enumerate((train_targets, *split_targets)):
        mat = np.zeros((len(series), len(vocab)), dtype=np.uint8)
        missing = []
        for row, tgt in enumerate(series):
            col = index.get(tgt)
            if col is None:
                missing.append(tgt)
            else:
                mat[row, col] = 1
        matrices.append(mat)
        if missing:
            unseen[pos] = sorted(set(missing))

    return matrices, names, unseen


def build_split_features(
    frames: dict[str, pd.DataFrame],
    smiles_col: str,
    target_col: str,
    drop_bit0: bool = True,
    target_encoding: str = "onehot",
    sequence_col: str = "target_seq",
):
    """
    Featurise an already-split dataset.

    `frames` maps split name -> dataframe, and must contain "train" (the
    one-hot vocabulary is fitted there). `target_encoding` selects the target
    block: "onehot" or "aac". Returns (features, feature_names, kept_frames,
    report); `kept_frames` has unparseable SMILES removed so the row order
    still matches the feature matrices.
    """
    if target_encoding not in ("onehot", "aac"):
        raise ValueError(f"unknown target_encoding: {target_encoding!r}")
    order = ["train"] + [k for k in frames if k != "train"]

    kept_frames, maccs, dropped = {}, {}, {}
    bit_names = None
    for name in order:
        df = frames[name]
        mat, valid, bit_names = maccs_matrix(df[smiles_col], drop_bit0=drop_bit0)
        n_bad = int((~valid).sum())
        if n_bad:
            dropped[name] = n_bad
            bad_smiles = df.loc[~valid, smiles_col].tolist()[:5]
            print(
                f"  WARNING: {name}: dropped {n_bad} rows with unparseable SMILES"
                f" (e.g. {bad_smiles})"
            )
        kept_frames[name] = df.loc[valid].reset_index(drop=True)
        maccs[name] = mat[valid]

    unseen_by_split = {}
    if target_encoding == "onehot":
        target_mats, target_names, unseen = encode_targets(
            *[kept_frames[name][target_col] for name in order]
        )
        unseen_by_split = {order[pos]: tgts for pos, tgts in unseen.items()}
        for name, tgts in unseen_by_split.items():
            print(
                f"  WARNING: {name}: {len(tgts)} target(s) absent from train,"
                f" encoded as all-zero: {tgts}"
            )
    else:
        target_mats, target_names = [], None
        for name in order:
            mat, target_names = aac_matrix(kept_frames[name][sequence_col])
            target_mats.append(mat)

    features = {
        name: np.hstack([maccs[name], target_mats[i]]).astype(np.float32)
        for i, name in enumerate(order)
    }
    feature_names = list(bit_names) + list(target_names)

    report = {
        "target_encoding": target_encoding,
        "n_maccs_bits": len(bit_names),
        "n_target_features": len(target_names),
        "n_features": len(feature_names),
        "rows_dropped_bad_smiles": dropped,
        "targets_unseen_in_train": unseen_by_split,
    }
    print(
        f"  features: {len(bit_names)} MACCS bits"
        f" + {len(target_names)} target {target_encoding} = {len(feature_names)}"
    )

    return features, feature_names, kept_frames, report

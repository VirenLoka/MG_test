#!/usr/bin/env python3
"""
Grouped train/val/test splitting, plus an audit that the grouping held.

Splits are grouped on `canonical_smiles`: every row sharing a SMILES goes to
the same split. This is stricter than requiring unique (glue, target) pairs
across splits — those pairs are already unique inside each source CSV, so a
plain row split would guarantee nothing. Grouping on the molecule stops a glue
from being learned against one target in train and then scored against a
different target in test.

`audit_split` re-derives the overlap numbers from the finished splits and
raises if any SMILES crosses a boundary, so a silent regression in the split
logic cannot pass unnoticed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """
    Three-way split on whole groups.

    Done in two passes: hold out (val + test) worth of groups, then halve that
    holdout into val and test in proportion to their configured fractions.
    Fractions apply to groups, so realised row counts drift a little.
    """
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    groups = df[group_col].to_numpy()

    holdout_frac = val_frac + test_frac
    first = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_idx, holdout_idx = next(first.split(df, groups=groups))

    train = df.iloc[train_idx].reset_index(drop=True)
    holdout = df.iloc[holdout_idx].reset_index(drop=True)

    # Within the holdout, test_frac / (val_frac + test_frac) becomes test.
    test_share = test_frac / holdout_frac
    second = GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=seed)
    val_idx, test_idx = next(
        second.split(holdout, groups=holdout[group_col].to_numpy())
    )

    return {
        "train": train,
        "val": holdout.iloc[val_idx].reset_index(drop=True),
        "test": holdout.iloc[test_idx].reset_index(drop=True),
    }


def audit_split(
    splits: dict[str, pd.DataFrame],
    smiles_col: str,
    target_col: str,
    label_col: str,
) -> dict:
    """
    Verify the split really is molecule-disjoint, and describe what it produced.

    Raises AssertionError on any SMILES or (SMILES, target) pair shared between
    two splits. Also reports target coverage, which grouping on the molecule
    does not control.
    """
    names = list(splits)
    total_rows = sum(len(d) for d in splits.values())

    smiles_sets = {n: set(d[smiles_col]) for n, d in splits.items()}
    pair_sets = {n: set(zip(d[smiles_col], d[target_col])) for n, d in splits.items()}

    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_smiles = smiles_sets[a] & smiles_sets[b]
            shared_pairs = pair_sets[a] & pair_sets[b]
            overlaps[f"{a}|{b}"] = {
                "shared_smiles": len(shared_smiles),
                "shared_pairs": len(shared_pairs),
                "examples": sorted(shared_smiles)[:3],
            }

    per_split = {
        n: {
            "rows": len(d),
            "row_frac": round(len(d) / total_rows, 4),
            "unique_smiles": int(d[smiles_col].nunique()),
            "unique_targets": int(d[target_col].nunique()),
            "label_counts": d[label_col].value_counts().sort_index().to_dict(),
            "positive_rate": round(float(d[label_col].mean()), 4),
        }
        for n, d in splits.items()
    }

    train_targets = set(splits["train"][target_col])
    coverage = {
        n: {
            "targets_not_in_train": sorted(set(d[target_col]) - train_targets),
        }
        for n, d in splits.items()
        if n != "train"
    }

    print("  split sizes (grouped on %s):" % smiles_col)
    for n, info in per_split.items():
        print(
            f"    {n:5s} {info['rows']:5d} rows ({info['row_frac']:.1%})"
            f"  {info['unique_smiles']:5d} SMILES"
            f"  {info['unique_targets']:3d} targets"
            f"  pos rate {info['positive_rate']:.3f}"
        )

    bad = {k: v for k, v in overlaps.items() if v["shared_smiles"] or v["shared_pairs"]}
    if bad:
        for k, v in bad.items():
            print(
                f"    LEAK {k}: {v['shared_smiles']} shared SMILES,"
                f" {v['shared_pairs']} shared pairs -> {v['examples']}"
            )
        raise AssertionError(f"grouped split leaked molecules across splits: {bad}")
    print("    leakage audit: 0 shared SMILES and 0 shared (SMILES, target) pairs")

    for n, info in coverage.items():
        if info["targets_not_in_train"]:
            print(
                f"    note: {n} contains {len(info['targets_not_in_train'])}"
                f" target(s) never seen in train: {info['targets_not_in_train']}"
            )

    return {
        "group_col": smiles_col,
        "per_split": per_split,
        "pairwise_overlap": overlaps,
        "target_coverage": coverage,
    }

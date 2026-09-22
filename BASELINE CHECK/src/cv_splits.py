#!/usr/bin/env python3
"""
Doubly-disjoint cross-validation: no target AND no scaffold shared across splits.

Targets are partitioned into K folds, greedily balanced by row count. For fold
k, test = fold k, val = fold k+1, train = the rest, so the three splits are
target-disjoint by construction.

Scaffold-disjointness is then imposed on top. A Bemis-Murcko scaffold whose
rows straddle two splits is assigned to whichever split holds most of its rows,
and its rows in the other splits are dropped. That costs data - roughly 15% of
dc50 and 20% of dmax rows - which is the price of the stronger claim: a test
molecule shares neither its target nor its core structure with anything seen in
training.

`audit_folds` re-derives the overlaps from the finished splits and raises if
any target, scaffold or molecule crosses a boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SPLIT_NAMES = ("train", "val", "test")


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None if RDKit cannot process it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def add_scaffolds(
    df: pd.DataFrame, smiles_col: str, scaffold_col: str = "scaffold"
) -> tuple[pd.DataFrame, dict]:
    """
    Attach a scaffold column, dropping rows whose scaffold cannot be derived.

    Fully acyclic molecules yield an empty scaffold string. They are grouped
    under a single sentinel so they are treated as one scaffold rather than as
    a free-for-all that could leak across splits.
    """
    scaffolds = [murcko_scaffold(s) for s in df[smiles_col]]
    out = df.copy()
    out[scaffold_col] = scaffolds

    n_failed = int(out[scaffold_col].isna().sum())
    out = out[out[scaffold_col].notna()].copy()

    n_acyclic = int((out[scaffold_col] == "").sum())
    out.loc[out[scaffold_col] == "", scaffold_col] = "__ACYCLIC__"

    report = {
        "scaffold_failed": n_failed,
        "acyclic_rows": n_acyclic,
        "n_scaffolds": int(out[scaffold_col].nunique()),
    }
    print(
        f"  scaffolds: {report['n_scaffolds']} unique"
        f" ({n_acyclic} acyclic rows pooled as one scaffold,"
        f" {n_failed} rows unparseable)"
    )
    return out.reset_index(drop=True), report


def target_folds(
    df: pd.DataFrame, target_col: str, n_folds: int, seed: int
) -> dict[str, int]:
    """
    Assign whole targets to K folds, balancing row counts.

    Greedy largest-first bin packing: the biggest targets are placed first, each
    into the currently lightest fold. With row counts as skewed as these
    (one target can be 20-29% of the data) this keeps folds far more even than
    a random partition would.
    """
    counts = df[target_col].value_counts()
    # Shuffle within equal counts so the seed has some effect on ties.
    rng = np.random.default_rng(seed)
    order = sorted(
        counts.index, key=lambda t: (-counts[t], rng.random())
    )

    load = np.zeros(n_folds, dtype=int)
    assignment: dict[str, int] = {}
    for target in order:
        k = int(np.argmin(load))
        assignment[target] = k
        load[k] += counts[target]

    print(f"  target folds (K={n_folds}), rows per fold: {load.tolist()}")
    return assignment


def make_fold(
    df: pd.DataFrame,
    fold: int,
    assignment: dict[str, int],
    n_folds: int,
    target_col: str,
    scaffold_col: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build one doubly-disjoint train/val/test split.

    test = fold, val = next fold, train = everything else. Scaffold conflicts
    are resolved by majority row count, ties going to train.
    """
    test_fold = fold
    val_fold = (fold + 1) % n_folds

    def split_of(target: str) -> str:
        k = assignment[target]
        if k == test_fold:
            return "test"
        if k == val_fold:
            return "val"
        return "train"

    work = df.copy()
    work["_split"] = work[target_col].map(split_of)

    # Resolve each scaffold to the split holding most of its rows. Ties are
    # broken by an explicit priority rather than by sort order, so the outcome
    # is deterministic and favours train (keeping test and val clean).
    priority = {"train": 0, "val": 1, "test": 2}
    counts = (
        work.groupby([scaffold_col, "_split"]).size().rename("n").reset_index()
    )
    counts["_pri"] = counts["_split"].map(priority)
    counts = counts.sort_values(["n", "_pri"], ascending=[False, True])
    scaffold_home = counts.drop_duplicates(subset=[scaffold_col], keep="first")
    home = dict(zip(scaffold_home[scaffold_col], scaffold_home["_split"]))

    work["_scaffold_home"] = work[scaffold_col].map(home)
    keep = work["_split"] == work["_scaffold_home"]

    dropped_by_split = (
        work.loc[~keep, "_split"].value_counts().to_dict()
    )
    kept = work[keep]

    splits = {
        name: kept[kept["_split"] == name]
        .drop(columns=["_split", "_scaffold_home"])
        .reset_index(drop=True)
        for name in SPLIT_NAMES
    }

    report = {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "rows_before": int(len(work)),
        "rows_after": int(len(kept)),
        "retention": round(float(len(kept) / len(work)), 4),
        "rows_dropped_for_scaffold_disjointness": {
            k: int(v) for k, v in dropped_by_split.items()
        },
    }
    return splits, report


def audit_folds(
    splits: dict[str, pd.DataFrame],
    target_col: str,
    scaffold_col: str,
    smiles_col: str,
    label_col: str,
) -> dict:
    """
    Verify target, scaffold and molecule disjointness; describe the result.

    Raises AssertionError on any crossing, so a regression in the split logic
    cannot quietly inflate the reported scores.
    """
    names = [n for n in SPLIT_NAMES if len(splits[n])]
    sets = {
        n: {
            "targets": set(splits[n][target_col]),
            "scaffolds": set(splits[n][scaffold_col]),
            "smiles": set(splits[n][smiles_col]),
        }
        for n in names
    }

    overlaps, bad = {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            entry = {
                kind: len(sets[a][kind] & sets[b][kind])
                for kind in ("targets", "scaffolds", "smiles")
            }
            overlaps[f"{a}|{b}"] = entry
            if any(entry.values()):
                bad[f"{a}|{b}"] = entry

    total = sum(len(splits[n]) for n in names)
    per_split = {
        n: {
            "rows": len(splits[n]),
            "row_frac": round(len(splits[n]) / total, 4),
            "n_targets": int(splits[n][target_col].nunique()),
            "targets": sorted(splits[n][target_col].unique().tolist()),
            "n_scaffolds": int(splits[n][scaffold_col].nunique()),
            "label_counts": splits[n][label_col].value_counts().sort_index().to_dict(),
            "positive_rate": round(float(splits[n][label_col].mean()), 4),
        }
        for n in names
    }

    for n, info in per_split.items():
        print(
            f"    {n:5s} {info['rows']:5d} rows ({info['row_frac']:.1%})"
            f"  {info['n_targets']:2d} targets"
            f"  {info['n_scaffolds']:4d} scaffolds"
            f"  pos={info['positive_rate']:.3f}"
        )
    print(f"    test targets: {per_split['test']['targets']}")

    if bad:
        for k, v in bad.items():
            print(f"    LEAK {k}: {v}")
        raise AssertionError(f"doubly-disjoint split leaked: {bad}")
    print("    audit: 0 shared targets, 0 shared scaffolds, 0 shared SMILES")

    return {"per_split": per_split, "pairwise_overlap": overlaps}

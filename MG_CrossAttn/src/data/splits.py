#!/usr/bin/env python3
"""
Target-disjoint cross-validation folds with optional scaffold-disjointness.

This is a corrected version of the split used for the MACCS/AAC baselines. The
baseline packed individual targets into folds, which placed paralogs in
different folds; because paralogs share chemical series, the scaffold-conflict
pass then destroyed a biased subsample of the smaller partner's rows (SMARCA2
retained only 25.3% of its dc50 rows, and what survived was precisely its
non-shared chemistry).

Here, `paralog_families` are packed as single units, so a family's shared
scaffolds never straddle a fold boundary. The held-out question becomes "can
the model generalise to a new target *family*" rather than "to a paralog of a
target it has already seen", which is both a fairer and a harder question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SPLIT_NAMES = ("train", "val", "test")
ACYCLIC_SENTINEL = "__ACYCLIC__"


def murcko_scaffold(smiles: str) -> str | None:
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
    Attach Bemis-Murcko scaffolds, dropping rows RDKit cannot process.

    Fully acyclic molecules have an empty scaffold; they are pooled under one
    sentinel so they behave as a single group rather than leaking freely.
    """
    out = df.copy()
    out[scaffold_col] = [murcko_scaffold(s) for s in out[smiles_col]]

    n_failed = int(out[scaffold_col].isna().sum())
    out = out[out[scaffold_col].notna()].copy()
    n_acyclic = int((out[scaffold_col] == "").sum())
    out.loc[out[scaffold_col] == "", scaffold_col] = ACYCLIC_SENTINEL

    report = {
        "scaffold_failed": n_failed,
        "acyclic_rows": n_acyclic,
        "n_scaffolds": int(out[scaffold_col].nunique()),
    }
    return out.reset_index(drop=True), report


def build_target_groups(
    targets: list[str], families: list[list[str]] | None, enabled: bool
) -> dict[str, str]:
    """
    Map each target to the packing unit it belongs to.

    A target in no family is its own unit. Family names are only applied for
    targets actually present in the dataset, so a config listing extra genes
    costs nothing.
    """
    group_of = {t: t for t in targets}
    if not enabled or not families:
        return group_of

    present = set(targets)
    for family in families:
        members = [t for t in family if t in present]
        if len(members) < 2:
            continue
        name = "FAM:" + "+".join(sorted(members))
        for t in members:
            group_of[t] = name
    return group_of


def assign_groups_to_folds(
    df: pd.DataFrame,
    target_col: str,
    group_of: dict[str, str],
    n_folds: int,
    seed: int,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Greedy largest-first packing of target groups into folds by row count.

    Row counts are extremely skewed here - one group can be a third of the
    data - so largest-first placement into the currently lightest fold keeps
    folds far more even than a random partition, though it cannot fully even
    out a group that is itself larger than a fold's fair share.
    """
    group_series = df[target_col].map(group_of)
    counts = group_series.value_counts()

    rng = np.random.default_rng(seed)
    order = sorted(counts.index, key=lambda g: (-counts[g], rng.random()))

    load = np.zeros(n_folds, dtype=int)
    fold_of_group: dict[str, int] = {}
    for group in order:
        k = int(np.argmin(load))
        fold_of_group[group] = k
        load[k] += int(counts[group])

    return fold_of_group, load


def make_fold(
    df: pd.DataFrame,
    fold: int,
    fold_of_group: dict[str, int],
    group_of: dict[str, str],
    n_folds: int,
    target_col: str,
    scaffold_col: str,
    scaffold_disjoint: bool = True,
    tie_priority: tuple[str, ...] = SPLIT_NAMES,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build one split. test = fold, val = fold+1, train = the rest.

    With `scaffold_disjoint`, a scaffold straddling two splits is assigned to
    whichever split holds most of its rows (ties by `tie_priority`) and its
    rows in other splits are dropped.
    """
    test_fold, val_fold = fold, (fold + 1) % n_folds

    def split_of_target(target: str) -> str:
        k = fold_of_group[group_of[target]]
        if k == test_fold:
            return "test"
        if k == val_fold:
            return "val"
        return "train"

    work = df.copy()
    work["_split"] = work[target_col].map(split_of_target)

    report: dict = {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "rows_before": int(len(work)),
        "scaffold_disjoint": scaffold_disjoint,
    }

    if scaffold_disjoint:
        priority = {name: i for i, name in enumerate(tie_priority)}
        counts = work.groupby([scaffold_col, "_split"]).size().rename("n").reset_index()
        counts["_pri"] = counts["_split"].map(priority)
        counts = counts.sort_values(["n", "_pri"], ascending=[False, True])
        winners = counts.drop_duplicates(subset=[scaffold_col], keep="first")
        home = dict(zip(winners[scaffold_col], winners["_split"]))
        work["_home"] = work[scaffold_col].map(home)
        keep = work["_split"] == work["_home"]
        report["rows_dropped_for_scaffold_disjointness"] = {
            k: int(v) for k, v in work.loc[~keep, "_split"].value_counts().items()
        }
        work = work[keep]
        work = work.drop(columns=["_home"])

    report["rows_after"] = int(len(work))
    report["retention"] = round(report["rows_after"] / report["rows_before"], 4)

    splits = {
        name: work[work["_split"] == name].drop(columns=["_split"]).reset_index(drop=True)
        for name in SPLIT_NAMES
    }
    return splits, report


def audit_fold(
    splits: dict[str, pd.DataFrame],
    target_col: str,
    scaffold_col: str,
    smiles_col: str,
    label_col: str,
    require_scaffold_disjoint: bool = True,
) -> dict:
    """
    Verify disjointness and summarise the split. Raises on any leak.

    Molecules are checked too: a shared SMILES would mean a shared scaffold,
    so this is a redundant guard that makes a logic regression loud.
    """
    names = [n for n in SPLIT_NAMES if len(splits[n])]
    kinds = ["targets", "scaffolds", "smiles"]
    cols = {"targets": target_col, "scaffolds": scaffold_col, "smiles": smiles_col}
    sets = {n: {k: set(splits[n][cols[k]]) for k in kinds} for n in names}

    overlaps, leaks = {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            entry = {k: len(sets[a][k] & sets[b][k]) for k in kinds}
            overlaps[f"{a}|{b}"] = entry
            checked = kinds if require_scaffold_disjoint else ["targets"]
            if any(entry[k] for k in checked):
                leaks[f"{a}|{b}"] = entry

    total = sum(len(splits[n]) for n in names)
    per_split = {
        n: {
            "rows": len(splits[n]),
            "row_frac": round(len(splits[n]) / total, 4),
            "n_targets": int(splits[n][target_col].nunique()),
            "targets": sorted(splits[n][target_col].unique().tolist()),
            "n_scaffolds": int(splits[n][scaffold_col].nunique()),
            "positive_rate": round(float(splits[n][label_col].mean()), 4),
        }
        for n in names
    }

    if leaks:
        raise AssertionError(f"split leaked across folds: {leaks}")

    return {"per_split": per_split, "pairwise_overlap": overlaps}


def build_cv_folds(
    df: pd.DataFrame,
    cfg: dict,
    smiles_col: str,
    target_col: str,
    label_col: str,
    scaffold_col: str = "scaffold",
):
    """
    Full pipeline: scaffolds -> target groups -> folds, each audited.

    Returns (folds, meta) where `folds` is a list of (splits, report, audit).
    """
    frame, scaffold_report = add_scaffolds(df, smiles_col, scaffold_col)

    targets = sorted(frame[target_col].unique().tolist())
    group_of = build_target_groups(
        targets, cfg.get("paralog_families"), cfg.get("group_paralogs", True)
    )
    n_folds = int(cfg["n_folds"])
    fold_of_group, load = assign_groups_to_folds(
        frame, target_col, group_of, n_folds, int(cfg["seed"])
    )

    tie_priority = tuple(cfg.get("scaffold_tie_priority", SPLIT_NAMES))
    scaffold_disjoint = bool(cfg.get("scaffold_disjoint", True))

    folds = []
    for fold in range(n_folds):
        splits, report = make_fold(
            frame, fold, fold_of_group, group_of, n_folds,
            target_col, scaffold_col,
            scaffold_disjoint=scaffold_disjoint, tie_priority=tie_priority,
        )
        if any(len(splits[s]) == 0 for s in SPLIT_NAMES):
            report["skipped"] = "a split came out empty"
            folds.append((splits, report, None))
            continue
        audit = audit_fold(
            splits, target_col, scaffold_col, smiles_col, label_col,
            require_scaffold_disjoint=scaffold_disjoint,
        )
        folds.append((splits, report, audit))

    meta = {
        "scaffold_report": scaffold_report,
        "n_folds": n_folds,
        "group_of_target": group_of,
        "fold_of_group": fold_of_group,
        "rows_per_fold": load.tolist(),
        "paralog_grouping_enabled": bool(cfg.get("group_paralogs", True)),
        "scaffold_disjoint": scaffold_disjoint,
    }
    return folds, meta

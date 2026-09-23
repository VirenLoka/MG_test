#!/usr/bin/env python3
"""
Target-disjoint cross-validation folds, balanced on class ratio and size.

Every target lands in exactly one fold, so a fold always answers "can the model
generalise to a target it has never seen". A molecule may recur across folds
provided it is paired with a different target each time; that is the whole
point of a pairwise dataset and is not leakage, so scaffold-disjointness is
off by default (`scaffold_disjoint`). It remains available, but note what it
costs: it deletes rows, and it deletes them non-uniformly - under the baseline
split SMARCA2 retained only 25.3% of its dc50 rows, and what survived was
precisely its non-shared chemistry.

The harder problem here is class ratio. In this data the label is close to a
function of the target - dc50 CSNK1A1 is 92.5% positive, SMARCA4 15.9%; dmax
VAV1 is 90.3% and CDK2 1.3% - so packing folds by row count alone, as the
baseline did, produced folds running from 24% to 93% positive (dc50) and 4% to
90% (dmax). `assign_groups_to_folds` therefore optimises the class ratio and
the row count of every fold together, by choosing which targets share a fold.
It never drops a row to achieve this.

That optimisation is bounded by the data, not by the search: with 801 dmax rows
at 4% positive and 937 at 90%, no target-disjoint partition can put every fold
near the 49.6% base rate. The fold report records what was actually achieved.

`paralog_families` are packed as single units, so a family's shared chemical
series never straddles a fold boundary and the held-out question becomes "a new
target *family*" rather than "a paralog of a target already seen".
"""
from __future__ import annotations

import math

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


DEFAULT_BALANCE = {
    "enabled": True,
    # Fold size is a constraint, not an objective: any fold holding between
    # these multiples of its fair share (total/n_folds) is acceptable, and
    # within that band the search is free to chase the class ratio. Trading
    # the two as weighted objectives instead does not work here - fold size
    # has by far the larger dynamic range, so it silently dominates and the
    # partition barely moves off the row-count-only packing.
    "min_fold_row_frac": 0.40,
    "max_fold_row_frac": 2.00,
    # A fold serves as test once and as val once, so a fold holding fewer than
    # this many of either class makes ROC-AUC/MCC undefined or meaningless
    # there. A penalty rather than a hard constraint: an infeasible request
    # should still return the least bad partition.
    "min_class_rows": 10,
    "restarts": 24,
    "sweeps": 60,
    "kicks": 120,
    "anneal_rungs": 3,
}

# Steep enough that no ratio gain is worth leaving the band or starving a
# class, but finite, so an infeasible request degrades instead of failing.
_BAND_PENALTY = 100.0
_FLOOR_PENALTY = 10.0


def _fold_totals(
    rows_of_group: np.ndarray, pos_of_group: np.ndarray, assign: np.ndarray, n_folds: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.zeros(n_folds, dtype=np.int64)
    pos = np.zeros(n_folds, dtype=np.int64)
    np.add.at(rows, assign, rows_of_group)
    np.add.at(pos, assign, pos_of_group)
    return rows, pos


def partition_cost(
    rows: np.ndarray,
    pos: np.ndarray,
    overall: float,
    cfg: dict,
    fair: float,
    strict: bool = True,
) -> float:
    """
    Distance of each fold's class ratio from the base rate, plus hinge
    penalties for leaving the size band or starving a class.

    The ratio term is the mean squared deviation *plus* the worst fold's
    squared deviation. The requirement is that every fold have a usable class
    ratio, which is a statement about the worst one; optimising the mean alone
    happily buys a large gain on four folds with a loss on the fifth, and then
    widening the size band can make the worst fold worse even as the mean
    improves. The mean term is kept because it still grades partitions that tie
    on their worst fold, which is what the descent needs to make progress.

    `fair` is passed in rather than derived from `rows` so the constructive
    pass scores partial partitions against the final fold size, not against a
    shrinking one. That pass also sets `strict=False`: folds are legitimately
    empty until every group is placed, and an empty fold scores as on-rate so
    only the size hinge pushes groups into it.
    """
    if strict and (rows == 0).any():
        return math.inf
    if fair <= 0:
        return math.inf

    filled = rows > 0
    rate = np.where(filled, pos / np.maximum(rows, 1), overall)
    deviation = np.square(rate - overall)
    imbalance = float(deviation.mean()) + float(deviation.max())

    lo = float(cfg["min_fold_row_frac"]) * fair
    hi = float(cfg["max_fold_row_frac"]) * fair
    under = np.maximum(0.0, lo - rows) / fair if strict else 0.0
    over = np.maximum(0.0, rows - hi) / fair
    band = float(np.sum(np.square(under)) + np.sum(np.square(over)))

    floor = max(int(cfg["min_class_rows"]), 0)
    if floor and strict:
        held = np.where(filled, np.minimum(pos, rows - pos), floor)
        deficit = np.maximum(0, floor - held) / floor
        scarcity = float(np.sum(np.square(deficit)))
    else:
        scarcity = 0.0

    return imbalance + _BAND_PENALTY * band + _FLOOR_PENALTY * scarcity


def _descend(
    assign: np.ndarray,
    rows_of_group: np.ndarray,
    pos_of_group: np.ndarray,
    n_folds: int,
    overall: float,
    cfg: dict,
    fair: float,
    sweeps: int,
) -> tuple[np.ndarray, float]:
    """
    Hill-climb to a local optimum by single moves, then pairwise swaps.

    Moves alone stall whenever two groups have to trade folds to improve - with
    a handful of dominant groups that happens constantly here - so each sweep
    follows the move pass with a swap pass, and stops once neither finds a gain.

    Fold totals are carried and patched by the delta of each candidate rather
    than recomputed from the assignment, which is what keeps the multi-restart,
    basin-hopping, band-annealed search above affordable.
    """
    assign = assign.copy()
    rows, pos = _fold_totals(rows_of_group, pos_of_group, assign, n_folds)
    best = partition_cost(rows, pos, overall, cfg, fair)
    n = len(assign)

    def shift(i: int, src: int, dst: int) -> None:
        rows[src] -= rows_of_group[i]
        pos[src] -= pos_of_group[i]
        rows[dst] += rows_of_group[i]
        pos[dst] += pos_of_group[i]

    for _ in range(max(int(sweeps), 1)):
        improved = False

        for i in range(n):
            home = int(assign[i])
            for k in range(n_folds):
                if k == home:
                    continue
                shift(i, home, k)
                c = partition_cost(rows, pos, overall, cfg, fair)
                if c < best - 1e-12:
                    best, improved = c, True
                    assign[i], home = k, k
                else:
                    shift(i, k, home)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = int(assign[i]), int(assign[j])
                if a == b:
                    continue
                shift(i, a, b)
                shift(j, b, a)
                c = partition_cost(rows, pos, overall, cfg, fair)
                if c < best - 1e-12:
                    best, improved = c, True
                    assign[i], assign[j] = b, a
                else:
                    shift(i, b, a)
                    shift(j, a, b)

        if not improved:
            break

    return assign, best


def assign_groups_to_folds(
    df: pd.DataFrame,
    target_col: str,
    label_col: str,
    group_of: dict[str, str],
    n_folds: int,
    seed: int,
    balance: dict | None = None,
) -> tuple[dict[str, int], np.ndarray, dict]:
    """
    Partition target groups into folds, balancing class ratio and row count.

    With balancing disabled this is the original largest-first packing by row
    count, kept so the baseline split stays reproducible. Enabled, it seeds the
    search with that packing plus a ratio-aware constructive pass and random
    restarts, and hill-climbs the best of them.

    Returns (fold_of_group, rows_per_fold, report).
    """
    cfg = {**DEFAULT_BALANCE, **(balance or {})}
    frame = df[[target_col, label_col]].copy()
    frame["_group"] = frame[target_col].map(group_of)
    stats = frame.groupby("_group")[label_col].agg(["size", "sum"])

    groups = list(stats.index)
    rows_of_group = stats["size"].to_numpy(dtype=np.int64)
    pos_of_group = stats["sum"].to_numpy(dtype=np.int64)
    overall = float(pos_of_group.sum() / max(rows_of_group.sum(), 1))

    rng = np.random.default_rng(seed)
    order = sorted(
        range(len(groups)), key=lambda i: (-int(rows_of_group[i]), rng.random())
    )

    # Largest-first into the lightest fold: the row-count-only baseline, and a
    # strong starting point even when the objective is wider than row count.
    by_size = np.zeros(len(groups), dtype=int)
    load = np.zeros(n_folds, dtype=np.int64)
    for i in order:
        k = int(np.argmin(load))
        by_size[i] = k
        load[k] += int(rows_of_group[i])

    if not cfg["enabled"]:
        rows, pos = _fold_totals(rows_of_group, pos_of_group, by_size, n_folds)
        report = _balance_report(rows, pos, overall, balanced=False)
        return {g: int(by_size[i]) for i, g in enumerate(groups)}, rows, report

    # A group larger than the ceiling makes the band unsatisfiable on its own,
    # and every partition then pays the same floor of penalty, which flattens
    # the ratio signal the search is supposed to follow. Widen to fit instead.
    fair = float(rows_of_group.sum()) / n_folds
    needed = float(rows_of_group.max()) / fair
    relaxed = None
    if needed > float(cfg["max_fold_row_frac"]):
        relaxed = {"requested": float(cfg["max_fold_row_frac"]),
                   "applied": round(needed, 4),
                   "forced_by_rows": int(rows_of_group.max())}
        cfg = {**cfg, "max_fold_row_frac": needed}

    def constructive(band: dict) -> np.ndarray:
        """Largest-first, each group into whichever fold leaves the partition
        cheapest - the ratio-aware counterpart of the by-size packing."""
        assign = np.full(len(groups), -1, dtype=int)
        for i in order:
            best_k, best_c = 0, math.inf
            for k in range(n_folds):
                assign[i] = k
                placed = assign >= 0
                rows, pos = _fold_totals(
                    rows_of_group[placed], pos_of_group[placed], assign[placed], n_folds
                )
                c = partition_cost(rows, pos, overall, band, fair, strict=False)
                if c < best_c:
                    best_c, best_k = c, k
            assign[i] = best_k
        return assign

    def optimize(band: dict, seeds: list[np.ndarray]) -> tuple[np.ndarray | None, float]:
        starts = list(seeds) + [by_size, constructive(band)]
        starts += [
            rng.integers(0, n_folds, len(groups))
            for _ in range(max(int(cfg["restarts"]), 0))
        ]
        best, best_c = None, math.inf
        for start in starts:
            assign, c = _descend(
                np.asarray(start, dtype=int), rows_of_group, pos_of_group, n_folds,
                overall, band, fair, int(cfg["sweeps"]),
            )
            if c < best_c:
                best, best_c = assign, c

        # Basin hopping: kick the incumbent and re-descend. Restarts alone
        # leave the incumbent stuck, because the good partitions here differ
        # from it by a coordinated move of several dominant groups at once.
        for _ in range(max(int(cfg["kicks"]), 0)):
            if best is None:
                break
            trial = best.copy()
            size = int(rng.integers(1, max(2, len(groups) // 3 + 1)))
            for i in rng.choice(len(groups), size=size, replace=False):
                trial[i] = rng.integers(n_folds)
            assign, c = _descend(
                trial, rows_of_group, pos_of_group, n_folds,
                overall, band, fair, int(cfg["sweeps"]),
            )
            if c < best_c:
                best, best_c = assign, c
        return best, best_c

    # Anneal the band from equal-size folds out to the requested width, seeding
    # each rung from the previous one. A tighter band is a subset of a wider
    # one, so a partition found under it stays feasible, and the tight rungs
    # are where the search is best behaved - the good wide-band partitions sit
    # in basins that restarts and kicks do not reach from cold. On dc50 a
    # single-rung search never finds the optimum however many kicks it is
    # given; three rungs reach it with fewer, from every seed tried.
    lo_req = float(cfg["min_fold_row_frac"])
    hi_req = float(cfg["max_fold_row_frac"])
    rungs = max(int(cfg["anneal_rungs"]), 1)
    best_assign, best_cost = None, math.inf
    seeds: list[np.ndarray] = []
    for step in range(rungs):
        t = 1.0 if rungs == 1 else step / (rungs - 1)
        band = {
            **cfg,
            "min_fold_row_frac": 1.0 + (lo_req - 1.0) * t,
            # The ceiling can never drop below the largest indivisible group.
            "max_fold_row_frac": max(1.0 + (hi_req - 1.0) * t, needed),
        }
        assign, c = optimize(band, seeds)
        if assign is not None:
            seeds = [assign]
            best_assign, best_cost = assign, c

    if best_assign is None:          # every start left a fold empty
        best_assign = by_size
        best_cost = math.inf

    rows, pos = _fold_totals(rows_of_group, pos_of_group, best_assign, n_folds)
    before_rows, before_pos = _fold_totals(rows_of_group, pos_of_group, by_size, n_folds)

    report = _balance_report(rows, pos, overall, balanced=True)
    report["size_band"] = [
        round(float(cfg["min_fold_row_frac"]), 4),
        round(float(cfg["max_fold_row_frac"]), 4),
    ]
    if relaxed:
        report["size_band_relaxed"] = relaxed
    report["cost"] = round(float(best_cost), 6)
    report["row_packing_baseline"] = _balance_report(
        before_rows, before_pos, overall, balanced=False
    )
    report["mixing"] = _mixing_report(
        groups, rows_of_group, pos_of_group, best_assign, n_folds
    )

    return {g: int(best_assign[i]) for i, g in enumerate(groups)}, rows, report


def _mixing_report(
    groups: list[str],
    rows_of_group: np.ndarray,
    pos_of_group: np.ndarray,
    assign: np.ndarray,
    n_folds: int,
) -> dict:
    """
    How much of each fold's class ratio is real, and how much is cancellation.

    A fold at the base rate can be one of two very different things. It can
    hold targets that are each near the base rate, or it can hold a 90%-positive
    target next to a 4%-positive one, whose rates average out. Both report the
    same fold ratio, but in the second the label is still read straight off
    target identity, so the balance is cosmetic and the fold is no harder.

    `spread` is the row-weighted mean absolute gap between a fold's constituent
    target rates and the fold's own rate: 0 means genuinely homogeneous, and
    anything approaching the fold's distance from 0/1 means cancellation. This
    diagnoses the split; it is not something the search can optimise away,
    because with a label this close to a function of the target, cancellation
    is the only mechanism available.
    """
    per_fold = []
    for k in range(n_folds):
        members = np.flatnonzero(assign == k)
        rows = int(rows_of_group[members].sum())
        if rows == 0:
            per_fold.append({"fold": k, "rows": 0, "spread": None, "targets": []})
            continue
        fold_rate = float(pos_of_group[members].sum() / rows)
        member_rates = pos_of_group[members] / np.maximum(rows_of_group[members], 1)
        weights = rows_of_group[members] / rows
        spread = float(np.sum(weights * np.abs(member_rates - fold_rate)))
        per_fold.append({
            "fold": k,
            "rows": rows,
            "positive_rate": round(fold_rate, 4),
            "spread": round(spread, 4),
            "targets": {
                groups[i]: round(float(pos_of_group[i] / max(rows_of_group[i], 1)), 4)
                for i in members[np.argsort(-rows_of_group[members])]
            },
        })

    spreads = [f["spread"] for f in per_fold if f["spread"] is not None]
    weightsum = sum(f["rows"] for f in per_fold if f["spread"] is not None)
    mean_spread = (
        sum(f["spread"] * f["rows"] for f in per_fold if f["spread"] is not None)
        / weightsum if weightsum else 0.0
    )
    return {
        "per_fold": per_fold,
        "mean_spread": round(float(mean_spread), 4),
        "max_spread": round(float(max(spreads)), 4) if spreads else None,
    }


def _balance_report(
    rows: np.ndarray, pos: np.ndarray, overall: float, balanced: bool
) -> dict:
    rate = np.where(rows > 0, pos / np.maximum(rows, 1), np.nan)
    finite = rate[np.isfinite(rate)]
    return {
        "balanced": balanced,
        "overall_positive_rate": round(overall, 4),
        "rows_per_fold": rows.tolist(),
        "positives_per_fold": pos.tolist(),
        "negatives_per_fold": (rows - pos).tolist(),
        "positive_rate_per_fold": [
            None if not np.isfinite(r) else round(float(r), 4) for r in rate
        ],
        "positive_rate_range": (
            [round(float(finite.min()), 4), round(float(finite.max()), 4)]
            if len(finite) else None
        ),
        "max_deviation_from_base_rate": (
            round(float(np.abs(finite - overall).max()), 4) if len(finite) else None
        ),
        "row_range": [int(rows.min()), int(rows.max())],
    }


def make_fold(
    df: pd.DataFrame,
    fold: int,
    fold_of_group: dict[str, int],
    group_of: dict[str, str],
    n_folds: int,
    target_col: str,
    scaffold_col: str,
    scaffold_disjoint: bool = False,
    tie_priority: tuple[str, ...] = SPLIT_NAMES,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build one split. test = fold, val = fold+1, train = the rest.

    Off by default, `scaffold_disjoint` assigns a scaffold straddling two
    splits to whichever split holds most of its rows (ties by `tie_priority`)
    and drops its rows in the others. It buys nothing here - a molecule in two
    folds is paired with a different target in each - and costs real rows, so
    `retention` is worth reading whenever it is switched on.
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
    require_scaffold_disjoint: bool = False,
) -> dict:
    """
    Verify disjointness and summarise the split. Raises on any leak.

    Target-disjointness is always enforced - it is the claim the whole CV rests
    on. Scaffold and SMILES overlap is always measured but only enforced under
    `require_scaffold_disjoint`, since a molecule recurring against a different
    target is the intended behaviour, not a leak.
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
            "n_positive": int(splits[n][label_col].sum()),
            "n_negative": int((1 - splits[n][label_col]).sum()),
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
    fold_of_group, load, balance_report = assign_groups_to_folds(
        frame, target_col, label_col, group_of, n_folds, int(cfg["seed"]),
        balance=cfg.get("balance"),
    )

    tie_priority = tuple(cfg.get("scaffold_tie_priority", SPLIT_NAMES))
    scaffold_disjoint = bool(cfg.get("scaffold_disjoint", False))

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
        "fold_balance": balance_report,
    }
    return folds, meta

#!/usr/bin/env python3
"""
Check for molecular-glue SMILES that leak across the train/test split.

Three questions this answers:
  1. How many exact-string Molecule_SMILES values appear in BOTH train and test?
  2. Of those, do any repeat with the same (SMILES, PrimaryTarget) pair, i.e. is the
     exact same glue-target assay duplicated across the split?
  3. For each SMILES that is shared between train and test, how much does its
     measured activity (both the raw `activity` column and the `Score` column
     used as the model's regression target) differ between its train
     occurrence(s) and its test occurrence(s)?

Usage:
    python analyze_train_test_overlap.py
    python analyze_train_test_overlap.py --train MG_data/train.csv --test MG_data/test.csv --out-dir MG_data/overlap_analysis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SMILES_COL = "Molecule_SMILES"
TARGET_COL = "PrimaryTarget"
TARGET_ID_COL = "PrimaryTarget_UniProtID"
E3_COL = "RecruitingProtein"
ACTIVITY_COLS = ["activity", "Score"]
LABEL_COL = "Score"  # the column actually used as the regression target during training
EPS = 1e-9


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
    return df


def find_duplicate_smiles(train: pd.DataFrame, test: pd.DataFrame):
    return sorted(set(train[SMILES_COL]) & set(test[SMILES_COL]))


def common_pairs(train: pd.DataFrame, test: pd.DataFrame, key_col: str):
    train_pairs = set(zip(train[SMILES_COL], train[key_col]))
    test_pairs = set(zip(test[SMILES_COL], test[key_col]))
    return train_pairs & test_pairs


def activity_variation(train: pd.DataFrame, test: pd.DataFrame, dup_smiles):
    rows = []
    for smi in dup_smiles:
        tr = train[train[SMILES_COL] == smi]
        te = test[test[SMILES_COL] == smi]

        tr_targets = sorted(tr[TARGET_COL].unique().tolist())
        te_targets = sorted(te[TARGET_COL].unique().tolist())

        row = {
            "smiles": smi,
            "n_train_rows": len(tr),
            "n_test_rows": len(te),
            "train_targets": "; ".join(tr_targets),
            "test_targets": "; ".join(te_targets),
            "same_target_in_both": bool(set(tr_targets) & set(te_targets)),
        }

        for col in ACTIVITY_COLS:
            tr_vals = pd.to_numeric(tr[col], errors="coerce").dropna()
            te_vals = pd.to_numeric(te[col], errors="coerce").dropna()

            tr_mean = tr_vals.mean() if len(tr_vals) else float("nan")
            te_mean = te_vals.mean() if len(te_vals) else float("nan")

            row[f"train_{col}_values"] = "; ".join(map(str, tr_vals.tolist()))
            row[f"test_{col}_values"] = "; ".join(map(str, te_vals.tolist()))
            row[f"train_{col}_mean"] = tr_mean
            row[f"test_{col}_mean"] = te_mean
            row[f"delta_{col}_mean"] = te_mean - tr_mean
            row[f"abs_delta_{col}_mean"] = abs(te_mean - tr_mean)

        rows.append(row)

    return pd.DataFrame(rows)


def regression_metrics(y_true, y_pred):
    """R2 / RMSE / MAE without an sklearn dependency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    sse = float((resid ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(np.abs(resid).mean())
    return r2, rmse, mae


def exact_score_match_count(var_df: pd.DataFrame):
    """How many of the shared molecules have an identical Score between splits."""
    mean_match = (var_df["train_Score_mean"] - var_df["test_Score_mean"]).abs() < EPS

    def all_values_equal(row):
        tr = [float(x) for x in str(row["train_Score_values"]).split(";")]
        te = [float(x) for x in str(row["test_Score_values"]).split(";")]
        return (max(tr + te) - min(tr + te)) < EPS

    strict_match = var_df.apply(all_values_equal, axis=1)
    return int(mean_match.sum()), int(strict_match.sum())


def memorization_leakage_report(train: pd.DataFrame, test: pd.DataFrame):
    """
    How much of the test set could be 'solved' by a model that ignores the
    target and just memorizes each molecule's typical train Score, versus a
    baseline that predicts the global train Score mean for everyone.
    """
    n_test = len(test)
    train_smiles_set = set(train[SMILES_COL])
    is_overlap = test[SMILES_COL].isin(train_smiles_set)
    n_overlap = int(is_overlap.sum())

    train_mean_by_smiles = train.groupby(SMILES_COL)[LABEL_COL].mean()
    global_train_mean = train[LABEL_COL].mean()

    overlap_test = test[is_overlap]
    y_true_overlap = overlap_test[LABEL_COL].values
    y_pred_memorized = overlap_test[SMILES_COL].map(train_mean_by_smiles).values
    y_pred_dumb_overlap = np.full_like(y_true_overlap, global_train_mean, dtype=float)

    y_true_full = test[LABEL_COL].values
    y_pred_full_mixed = test[SMILES_COL].map(train_mean_by_smiles).fillna(global_train_mean).values
    y_pred_full_dumb = np.full_like(y_true_full, global_train_mean, dtype=float)

    return {
        "n_test": n_test,
        "n_overlap": n_overlap,
        "pct_overlap": 100 * n_overlap / n_test,
        "global_train_mean": global_train_mean,
        "overlap_memorized": regression_metrics(y_true_overlap, y_pred_memorized),
        "overlap_dumb": regression_metrics(y_true_overlap, y_pred_dumb_overlap),
        "full_mixed": regression_metrics(y_true_full, y_pred_full_mixed),
        "full_dumb": regression_metrics(y_true_full, y_pred_full_dumb),
    }


def try_canonical_smiles_check(train: pd.DataFrame, test: pd.DataFrame):
    """Optional sanity check: are there additional duplicates hiding behind
    differently-written (non-canonical) SMILES strings for the same molecule?"""
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None

    def canon(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol is not None else None

    train_canon = train[SMILES_COL].map(canon)
    test_canon = test[SMILES_COL].map(canon)

    train_set = set(train_canon.dropna())
    test_set = set(test_canon.dropna())
    canon_common = train_set & test_set

    exact_common = set(train[SMILES_COL]) & set(test[SMILES_COL])
    exact_common_canon = {canon(s) for s in exact_common}

    extra = canon_common - exact_common_canon
    return len(canon_common), len(extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="MG_data/train.csv", type=Path)
    parser.add_argument("--test", default="MG_data/test.csv", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If given, write duplicate_smiles.csv and activity_variation.csv here.",
    )
    args = parser.parse_args()

    train = load(args.train)
    test = load(args.test)

    print("=" * 80)
    print(f"train: {len(train)} rows, {train[SMILES_COL].nunique()} unique SMILES")
    print(f"test:  {len(test)} rows, {test[SMILES_COL].nunique()} unique SMILES")

    # --- 1. exact-string SMILES overlap -------------------------------------------------
    dup_smiles = find_duplicate_smiles(train, test)
    print("\n" + "=" * 80)
    print(f"1) Exact-string Molecule_SMILES shared between train and test: {len(dup_smiles)}")

    canon_check = try_canonical_smiles_check(train, test)
    if canon_check is not None:
        n_canon, n_extra = canon_check
        print(f"   (sanity check via RDKit canonical SMILES: {n_canon} common molecules; "
              f"{n_extra} of those are NOT caught by exact-string matching)")
        if n_extra:
            print("   -> some duplicate molecules are written with different SMILES strings; "
                  "exact-string matching under-counts them.")
    else:
        print("   (RDKit not available - skipped canonical-SMILES sanity check)")

    # --- 2. common (SMILES, target) pairs -------------------------------------------------
    print("\n" + "=" * 80)
    print("2) Common (SMILES, target) pairs between train and test:")

    pairs_name = common_pairs(train, test, TARGET_COL)
    pairs_uid = common_pairs(train, test, TARGET_ID_COL)
    pairs_e3 = common_pairs(train, test, E3_COL)

    print(f"   - (SMILES, PrimaryTarget)          : {len(pairs_name)}")
    print(f"   - (SMILES, PrimaryTarget_UniProtID): {len(pairs_uid)}")
    print(f"   - (SMILES, RecruitingProtein / E3)  : {len(pairs_e3)}  (context, not the assay target)")

    if pairs_name:
        print("   Repeated (SMILES, PrimaryTarget) pairs:")
        for smi, tgt in sorted(pairs_name):
            print(f"     {tgt:15s} {smi}")
    else:
        print("   -> None of the shared molecules are tested against the same PrimaryTarget "
              "in both splits (every shared glue appears against a *different* target in "
              "train vs. test), even though nearly all still recruit the same E3 ligase.")

    # --- 3. activity variation for shared molecules ----------------------------------------
    print("\n" + "=" * 80)
    print("3) Activity variation (train vs test) for the shared SMILES:")

    var_df = activity_variation(train, test, dup_smiles)
    if len(var_df):
        for col in ACTIVITY_COLS:
            mean_abs_delta = var_df[f"abs_delta_{col}_mean"].mean()
            max_abs_delta = var_df[f"abs_delta_{col}_mean"].max()
            print(f"   [{col}] mean |train_mean - test_mean| across {len(var_df)} molecules: "
                  f"{mean_abs_delta:.3f} (max: {max_abs_delta:.3f})")

        print("\n   Top 10 molecules by |Score mean delta| (train vs test):")
        top = var_df.sort_values("abs_delta_Score_mean", ascending=False).head(10)
        for _, r in top.iterrows():
            print(f"     delta_Score={r['delta_Score_mean']:+.3f}  "
                  f"delta_activity={r['delta_activity_mean']:+.2f}  "
                  f"same_target={r['same_target_in_both']}  "
                  f"train_targets=[{r['train_targets']}]  test_targets=[{r['test_targets']}]  "
                  f"{r['smiles'][:50]}")

    # --- 4. exact-score matches + memorization / leakage simulation -----------------------
    print("\n" + "=" * 80)
    print("4) How identical are the scores, and how much would memorizing them help:")

    if len(var_df):
        n_mean_match, n_strict_match = exact_score_match_count(var_df)
        print(f"   - {n_mean_match}/{len(var_df)} shared molecules have an IDENTICAL mean "
              f"Score in train vs test (exact match, not just close)")
        print(f"   - {n_strict_match}/{len(var_df)} have ZERO spread at all: every train "
              f"occurrence and every test occurrence of that molecule has the exact same "
              f"Score")

    leak = memorization_leakage_report(train, test)
    print(f"\n   Of the {leak['n_test']} test rows, {leak['n_overlap']} "
          f"({leak['pct_overlap']:.1f}%) have a Molecule_SMILES that also appears in train.")
    print(f"   Global train Score mean: {leak['global_train_mean']:.3f}")

    def fmt(name, metrics):
        r2, rmse, mae = metrics
        print(f"     {name:55s} R2={r2:7.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    print(f"\n   -- On just the {leak['n_overlap']} overlapping test rows --")
    fmt("'memorize' (predict train mean Score for that molecule)", leak["overlap_memorized"])
    fmt("dumb baseline (predict global train Score mean)", leak["overlap_dumb"])

    print(f"\n   -- On the FULL {leak['n_test']}-row test set --")
    fmt("memorize overlap rows, dumb-baseline the rest", leak["full_mixed"])
    fmt("dumb baseline everywhere (no memorization credit at all)", leak["full_dumb"])

    r2_gap = leak["full_mixed"][0] - leak["full_dumb"][0]
    print(f"\n   -> Molecule-level memorization alone (with ZERO real learning about the "
          f"target) inflates whole-test-set R2 by {r2_gap:+.4f} "
          f"({leak['full_dumb'][0]:.4f} -> {leak['full_mixed'][0]:.4f}) just from the "
          f"{leak['pct_overlap']:.0f}% of test rows whose molecule was already seen in train.")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.Series(dup_smiles, name="smiles").to_csv(args.out_dir / "duplicate_smiles.csv", index=False)
        var_df.to_csv(args.out_dir / "activity_variation.csv", index=False)
        print(f"\nWrote detailed results to {args.out_dir}/")


if __name__ == "__main__":
    main()

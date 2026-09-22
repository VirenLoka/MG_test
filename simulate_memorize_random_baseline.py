#!/usr/bin/env python3
"""
Toy "shortcut" model on the real test set: perfectly memorize the score for
every test row whose molecule was already seen in train, and either (a) guess
uniformly at random in [0, 1], or (b) always guess a fixed constant (0.5 by
default) for the rest.

This is a more extreme version of the memorization baseline from
analyze_train_test_overlap.py (which predicted the train-mean Score for a
repeated molecule). Here the memorized rows are given the exact correct
answer, to see how much of the paper's reported test-set R2/RMSE/MAE could in
principle come from molecule recall alone, with no real target-conditioned
skill on the unseen rows.

Usage:
    python simulate_memorize_random_baseline.py --mode random
    python simulate_memorize_random_baseline.py --mode constant --constant-value 0.5
"""
import argparse

import numpy as np
import pandas as pd

SMILES_COL = "Molecule_SMILES"
LABEL_COL = "Score"

# Reported in Zhuang et al., bioRxiv 2026.08.08.739980 (MG2Act paper)
PAPER_REFERENCE = {
    "MG2Act, full test set": (0.5950, 0.1751, None),
    "MG2Act, de-redundant 79-pair subset": (0.4223, None, None),
    "Best baseline (XGBoost, MACCS+ESM-C), full test set": (0.5550, None, None),
    "Baselines, de-redundant subset (collapse)": (0.2710, None, None),
}


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    sse = float((resid ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt((resid ** 2).mean()))
    mae = float(np.abs(resid).mean())
    return r2, rmse, mae


def print_comparison(label, r2, rmse, mae):
    print("\n" + "=" * 80)
    print("Comparison to Zhuang et al. (MG2Act paper):")
    print(f"  {'Model':55s} R2       RMSE     MAE")
    print(f"  {'-'*80}")
    print(f"  {label:55s} {r2:<8.4f} {rmse:<8.4f} {mae:<8.4f}")
    for name, (p_r2, p_rmse, p_mae) in PAPER_REFERENCE.items():
        r2_str = f"{p_r2:.4f}" if p_r2 is not None else "n/a"
        rmse_str = f"{p_rmse:.4f}" if p_rmse is not None else "n/a"
        mae_str = f"{p_mae:.4f}" if p_mae is not None else "n/a"
        print(f"  {name:55s} {r2_str:<8s} {rmse_str:<8s} {mae_str:<8s}")

    gap = PAPER_REFERENCE["MG2Act, full test set"][0] - r2
    de_redundant_floor = PAPER_REFERENCE["Baselines, de-redundant subset (collapse)"][0]
    relation = "clears" if r2 > de_redundant_floor else "falls short of"
    print(f"\n-> {gap:.4f} below the paper's own trained MG2Act (R2=0.5950).")
    print(f"   It {relation} the reported de-redundant-subset baseline collapse "
          f"(R2<={de_redundant_floor:.4f}).")


def run_random_mode(y_true, is_memorized, n_random, trials, rng):
    print(f"\nSingle random draw:", end=" ")
    y_pred_single = y_true.copy()
    y_pred_single[~is_memorized] = rng.uniform(0, 1, size=n_random)
    r2_s, rmse_s, mae_s = regression_metrics(y_true, y_pred_single)
    print(f"R2={r2_s:.4f}  RMSE={rmse_s:.4f}  MAE={mae_s:.4f}")

    r2s, rmses, maes = np.zeros(trials), np.zeros(trials), np.zeros(trials)
    for i in range(trials):
        y_pred = y_true.copy()
        y_pred[~is_memorized] = rng.uniform(0, 1, size=n_random)
        r2s[i], rmses[i], maes[i] = regression_metrics(y_true, y_pred)

    print(f"\nMonte Carlo over {trials} random draws for the {n_random} non-memorized rows:")
    print(f"  R2   = {r2s.mean():.4f} +/- {r2s.std():.4f}   (range {r2s.min():.4f} to {r2s.max():.4f})")
    print(f"  RMSE = {rmses.mean():.4f} +/- {rmses.std():.4f}   (range {rmses.min():.4f} to {rmses.max():.4f})")
    print(f"  MAE  = {maes.mean():.4f} +/- {maes.std():.4f}   (range {maes.min():.4f} to {maes.max():.4f})")

    print_comparison("Memorize-then-random (mean of Monte Carlo)", r2s.mean(), rmses.mean(), maes.mean())


def run_constant_mode(y_true, is_memorized, n_random, constant_value):
    y_pred = y_true.copy()
    y_pred[~is_memorized] = constant_value
    r2, rmse, mae = regression_metrics(y_true, y_pred)

    print(f"\nMemorized rows predicted exactly; the {n_random} unseen rows all predicted "
          f"as a constant {constant_value}:")
    print(f"  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    print_comparison(f"Memorize-then-constant({constant_value})", r2, rmse, mae)


def exact_match_mask(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """True for test rows whose Score exactly equals >=1 train occurrence of the same SMILES."""
    train_scores_by_smiles = train.groupby(SMILES_COL)[LABEL_COL].apply(list)
    mask = np.zeros(len(test), dtype=bool)
    for i, (_, row) in enumerate(test.iterrows()):
        train_vals = train_scores_by_smiles.get(row[SMILES_COL])
        if train_vals is not None and any(abs(row[LABEL_COL] - tv) < 1e-9 for tv in train_vals):
            mask[i] = True
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="MG_data/train.csv")
    parser.add_argument("--test", default="MG_data/test.csv")
    parser.add_argument("--mode", choices=["random", "constant"], default="random",
                         help="How to predict the non-memorized rows.")
    parser.add_argument("--constant-value", type=float, default=0.5,
                         help="Value to predict for non-memorized rows in --mode constant.")
    parser.add_argument("--memorize-criterion", choices=["seen", "exact"], default="seen",
                         help="'seen': molecule's SMILES appears anywhere in train (83 rows). "
                              "'exact': this row's Score exactly matches a train occurrence of "
                              "the same molecule (59 rows) - the stricter, row-level definition.")
    parser.add_argument("--trials", type=int, default=2000,
                         help="Monte Carlo trials for --mode random, since those predictions are random.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    n_test = len(test)

    if args.memorize_criterion == "seen":
        is_memorized = test[SMILES_COL].isin(set(train[SMILES_COL])).values
    else:
        is_memorized = exact_match_mask(train, test)
    n_memorized = int(is_memorized.sum())
    n_random = n_test - n_memorized

    y_true = test[LABEL_COL].values.astype(float)

    print("=" * 80)
    pct = 100 * n_memorized / n_test
    print(f"Test set: {n_test} rows  ->  {n_memorized} ({pct:.1f}%) memorized "
          f"[criterion={args.memorize_criterion}], {n_random} unseen")
    print("(Note: the actual test.csv has 127 data rows / 128 lines including the header; "
          "using the real split as-is.)")

    if args.mode == "random":
        run_random_mode(y_true, is_memorized, n_random, args.trials, rng)
    else:
        run_constant_mode(y_true, is_memorized, n_random, args.constant_value)


if __name__ == "__main__":
    main()

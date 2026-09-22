#!/usr/bin/env python3
"""
Interrogate *why* the baselines score as well as they do.

A molecule-disjoint split removes exact duplicates but not close analogs, and
it does nothing about target-level label skew. Three diagnostics separate
genuine structure-activity signal from those two shortcuts:

  1. Feature ablation - train on target one-hot alone, on MACCS alone, and on
     both. If the target block alone nearly matches the full model, the model
     is largely predicting a per-target base rate.

  2. Analog leakage - for every test molecule, the maximum Morgan-fingerprint
     Tanimoto similarity to any training molecule. Test performance is then
     broken out by similarity band. High scores confined to the near-duplicate
     band mean the split is disjoint in name only.

  3. Per-target label skew - how lopsided each target's labels are, which
     bounds how much a target-identity feature can be worth.

Usage:
    python "BASELINE CHECK/src/diagnostics.py"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from data import load_config, prepare_all  # noqa: E402
from features import build_split_features  # noqa: E402
from splits import group_split  # noqa: E402

RDLogger.DisableLog("rdApp.*")
ROOT = SRC.parent.parent

# Bands for the test-vs-train nearest-neighbour similarity breakdown.
SIM_BANDS = [
    ("near-duplicate (>=0.9)", 0.90, 1.01),
    ("close analog [0.7,0.9)", 0.70, 0.90),
    ("related [0.5,0.7)", 0.50, 0.70),
    ("distinct (<0.5)", -0.01, 0.50),
]


def ablation(X, y, n_maccs: int, seed: int) -> dict:
    """Test ROC-AUC for target-only, MACCS-only, and the full feature set."""
    blocks = {
        "target_onehot_only": slice(n_maccs, None),
        "maccs_only": slice(0, n_maccs),
        "maccs_plus_target": slice(None),
    }
    out = {}
    for label, sl in blocks.items():
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
        model.fit(X["train"][:, sl], y["train"])
        row = {}
        for split in ("val", "test"):
            prob = model.predict_proba(X[split][:, sl])[:, 1]
            row[split] = round(float(roc_auc_score(y[split], prob)), 4)
        row["n_features"] = int(X["train"][:, sl].shape[1])
        out[label] = row
    return out


def morgan_fps(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        keep.append(i)
    return fps, np.asarray(keep)


def nearest_neighbour_similarity(test_smiles, train_smiles):
    """Max Tanimoto from each test molecule to the training set."""
    train_fps, _ = morgan_fps(train_smiles)
    test_fps, keep = morgan_fps(test_smiles)

    sims = np.full(len(test_smiles), np.nan)
    for pos, fp in zip(keep, test_fps):
        sims[pos] = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    return sims


def similarity_breakdown(sims, y_true, y_prob) -> dict:
    """Test ROC-AUC and accuracy within each similarity band."""
    out = {
        "median_max_similarity": round(float(np.nanmedian(sims)), 4),
        "mean_max_similarity": round(float(np.nanmean(sims)), 4),
        "frac_above_0.9": round(float(np.nanmean(sims >= 0.90)), 4),
        "frac_above_0.7": round(float(np.nanmean(sims >= 0.70)), 4),
        "bands": {},
    }
    y_pred = (y_prob >= 0.5).astype(int)
    for label, lo, hi in SIM_BANDS:
        mask = (sims >= lo) & (sims < hi)
        n = int(mask.sum())
        entry = {"n": n, "roc_auc": None, "accuracy": None}
        if n:
            entry["accuracy"] = round(float(np.mean(y_pred[mask] == y_true[mask])), 4)
            if len(np.unique(y_true[mask])) > 1:
                entry["roc_auc"] = round(
                    float(roc_auc_score(y_true[mask], y_prob[mask])), 4
                )
        out["bands"][label] = entry
    return out


def target_skew(frame, target_col, label_col) -> list[dict]:
    g = frame.groupby(target_col)[label_col]
    rows = [
        {
            "target": t,
            "n": int(len(s)),
            "positive_rate": round(float(s.mean()), 4),
            "skew": round(abs(float(s.mean()) - 0.5) * 2, 4),
        }
        for t, s in g
    ]
    return sorted(rows, key=lambda r: -r["n"])


def run(name, dataset, cfg, seed) -> dict:
    cols = cfg["columns"]
    smiles_col, target_col = cols["smiles"], cols["target"]
    label_col = dataset.label_col

    print(f"\n{'=' * 72}\n{name.upper()} diagnostics\n{'=' * 72}")

    sp = cfg["split"]
    splits = group_split(
        dataset.frame, smiles_col,
        sp["train"], sp["val"], sp["test"], seed,
    )
    X, feature_names, kept, feat_report = build_split_features(
        splits, smiles_col, target_col, cfg["features"]["drop_maccs_bit0"]
    )
    y = {k: kept[k][label_col].to_numpy() for k in kept}
    n_maccs = feat_report["n_maccs_bits"]

    print("\n  1. feature ablation (RandomForest, test ROC-AUC)")
    abl = ablation(X, y, n_maccs, seed)
    for label, row in abl.items():
        print(
            f"     {label:22s} feats={row['n_features']:4d}"
            f"  val={row['val']:.4f}  test={row['test']:.4f}"
        )

    print("\n  2. test-vs-train nearest-neighbour similarity (Morgan r=2)")
    sims = nearest_neighbour_similarity(
        kept["test"][smiles_col].tolist(), kept["train"][smiles_col].tolist()
    )
    full = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    ).fit(X["train"], y["train"])
    test_prob = full.predict_proba(X["test"])[:, 1]

    sim_report = similarity_breakdown(sims, y["test"], test_prob)
    print(
        f"     median max-similarity to train: {sim_report['median_max_similarity']}"
        f"   >=0.9: {sim_report['frac_above_0.9']:.1%}"
        f"   >=0.7: {sim_report['frac_above_0.7']:.1%}"
    )
    for label, entry in sim_report["bands"].items():
        print(
            f"     {label:24s} n={entry['n']:4d}"
            f"  roc_auc={entry['roc_auc']}  acc={entry['accuracy']}"
        )

    print("\n  3. per-target label skew (top 8 by row count)")
    skew = target_skew(dataset.frame, target_col, label_col)
    for row in skew[:8]:
        print(
            f"     {row['target']:10s} n={row['n']:4d}"
            f"  pos_rate={row['positive_rate']:.3f}  skew={row['skew']:.3f}"
        )

    return {
        "ablation": abl,
        "similarity": sim_report,
        "target_skew": skew,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(SRC.parent / "config" / "baseline.yaml"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["split"]["seed"]
    out_dir = ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets\n" + "-" * 72)
    datasets = prepare_all(cfg, ROOT)

    payload = {"seed": seed, "datasets": {}}
    for name, ds in datasets.items():
        payload["datasets"][name] = run(name, ds, cfg, seed)

    with open(out_dir / "diagnostics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_dir / 'diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

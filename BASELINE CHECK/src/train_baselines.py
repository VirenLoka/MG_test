#!/usr/bin/env python3
"""
Train RandomForest and XGBoost baselines on the dc50 and dmax datasets.

Features are MACCS keys (glue) concatenated with a one-hot block (target).
Labels are the binary columns already present in the source CSVs. Splits are
grouped on canonical_smiles so the test set contains only molecules never seen
in training.

Usage:
    python "BASELINE CHECK/src/train_baselines.py"
    python "BASELINE CHECK/src/train_baselines.py" --datasets dc50 --seed 7
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier  # noqa: E402

from data import load_config, prepare_all  # noqa: E402
from features import build_split_features  # noqa: E402
from splits import audit_split, group_split  # noqa: E402

# Repo root is two levels up from this file ("BASELINE CHECK/src/..").
ROOT = SRC.parent.parent


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Threshold-free and thresholded metrics for one split."""
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2

    metrics = {
        "n": int(len(y_true)),
        "positive_rate_true": round(float(np.mean(y_true)), 4),
        "positive_rate_pred": round(float(np.mean(y_pred)), 4),
        # ROC/PR are undefined when a split holds a single class.
        "roc_auc": None if single_class else round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": None if single_class else round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4) if not single_class else None,
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def majority_baseline(y_train, splits_y) -> dict:
    """
    Always-predict-the-training-majority reference.

    Without this it is easy to mistake class imbalance for signal.
    """
    majority = int(pd.Series(y_train).mode().iloc[0])
    out = {}
    for name, y in splits_y.items():
        y_pred = np.full(len(y), majority)
        out[name] = {
            "n": int(len(y)),
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred)), 4),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        }
    out["predicted_class"] = majority
    return out


def fit_random_forest(cfg, X, y, feature_names):
    params = dict(cfg["models"]["random_forest"])
    model = RandomForestClassifier(**params)
    model.fit(X["train"], y["train"])
    return model, params


def fit_xgboost(cfg, X, y, feature_names):
    params = dict(cfg["models"]["xgboost"])
    early = params.pop("early_stopping_rounds", None)

    # Imbalance handling mirrors RF's class_weight="balanced".
    pos = float(np.sum(y["train"] == 1))
    neg = float(np.sum(y["train"] == 0))
    params["scale_pos_weight"] = round(neg / pos, 4) if pos else 1.0

    model = XGBClassifier(**params, early_stopping_rounds=early)
    model.fit(
        X["train"],
        y["train"],
        eval_set=[(X["val"], y["val"])],
        verbose=False,
    )
    params["early_stopping_rounds"] = early
    params["best_iteration"] = int(getattr(model, "best_iteration", -1))
    return model, params


def top_features(model, feature_names, k: int = 20) -> list[dict]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    order = np.argsort(importances)[::-1][:k]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 5)}
        for i in order
        if importances[i] > 0
    ]


def run_dataset(name, dataset, cfg, out_dir: Path, seed: int) -> dict:
    cols = cfg["columns"]
    smiles_col, target_col = cols["smiles"], cols["target"]
    label_col = dataset.label_col

    print(f"\n{'=' * 72}\n{name.upper()}\n{'=' * 72}")

    sp_cfg = cfg["split"]
    splits = group_split(
        dataset.frame,
        group_col=smiles_col,
        train_frac=sp_cfg["train"],
        val_frac=sp_cfg["val"],
        test_frac=sp_cfg["test"],
        seed=seed,
    )
    split_report = audit_split(splits, smiles_col, target_col, label_col)

    X, feature_names, kept, feat_report = build_split_features(
        splits,
        smiles_col=smiles_col,
        target_col=target_col,
        drop_bit0=cfg["features"]["drop_maccs_bit0"],
        target_encoding=cfg["features"].get("target_encoding", "onehot"),
        sequence_col=cols["sequence"],
    )
    y = {k: kept[k][label_col].to_numpy() for k in kept}

    ds_out = out_dir / name
    (ds_out / "models").mkdir(parents=True, exist_ok=True)

    # Persist the exact splits so any result here can be reproduced or re-audited.
    keep_cols = [
        "compound_id", smiles_col, target_col, "target_uniprot",
        label_col, dataset.value_col, "source_patent",
    ]
    for split_name, frame in kept.items():
        present = [c for c in keep_cols if c in frame.columns]
        frame[present].to_csv(ds_out / f"split_{split_name}.csv", index=False)

    results = {}
    for model_key, fitter in (
        ("random_forest", fit_random_forest),
        ("xgboost", fit_xgboost),
    ):
        print(f"\n  -- {model_key} --")
        model, params = fitter(cfg, X, y, feature_names)

        per_split = {}
        for split_name in ("train", "val", "test"):
            prob = model.predict_proba(X[split_name])[:, 1]
            per_split[split_name] = evaluate(y[split_name], prob)
            m = per_split[split_name]
            print(
                f"     {split_name:5s} n={m['n']:5d}"
                f"  roc_auc={m['roc_auc']}"
                f"  pr_auc={m['pr_auc']}"
                f"  acc={m['accuracy']}"
                f"  bal_acc={m['balanced_accuracy']}"
                f"  f1={m['f1']}"
                f"  mcc={m['mcc']}"
            )

        # Save per-row test predictions for error analysis.
        test_pred = kept["test"][[smiles_col, target_col, label_col]].copy()
        test_pred["prob_positive"] = model.predict_proba(X["test"])[:, 1]
        test_pred["pred_label"] = (test_pred["prob_positive"] >= 0.5).astype(int)
        test_pred.to_csv(ds_out / f"test_predictions_{model_key}.csv", index=False)

        joblib.dump(
            {"model": model, "feature_names": feature_names},
            ds_out / "models" / f"{model_key}.joblib",
        )

        results[model_key] = {
            "params": params,
            "metrics": per_split,
            "top_features": top_features(model, feature_names),
        }

    results["majority_baseline"] = majority_baseline(
        y["train"], {k: y[k] for k in ("train", "val", "test")}
    )

    return {
        "dataset": name,
        "label_col": label_col,
        "data_report": dataset.report,
        "split_report": split_report,
        "feature_report": feat_report,
        "results": results,
    }


def write_summary(payload: dict, path: Path) -> None:
    """Flat per-(dataset, model, split) table, easiest thing to eyeball."""
    rows = []
    for ds_name, ds in payload["datasets"].items():
        for model_key in ("random_forest", "xgboost"):
            for split_name, m in ds["results"][model_key]["metrics"].items():
                rows.append({
                    "dataset": ds_name,
                    "model": model_key,
                    "split": split_name,
                    "n": m["n"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "mcc": m["mcc"],
                })
        mb = ds["results"]["majority_baseline"]
        for split_name in ("train", "val", "test"):
            rows.append({
                "dataset": ds_name,
                "model": "majority_baseline",
                "split": split_name,
                "n": mb[split_name]["n"],
                "roc_auc": None,
                "pr_auc": None,
                "accuracy": mb[split_name]["accuracy"],
                "balanced_accuracy": mb[split_name]["balanced_accuracy"],
                "precision": None,
                "recall": None,
                "f1": mb[split_name]["f1"],
                "mcc": None,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(SRC.parent / "config" / "baseline.yaml"),
        help="path to baseline.yaml",
    )
    ap.add_argument(
        "--datasets", nargs="*", default=None,
        help="subset of datasets to run (default: all in the config)",
    )
    ap.add_argument("--seed", type=int, default=None, help="override the split seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["split"]["seed"]

    out_dir = ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets\n" + "-" * 72)
    datasets = prepare_all(cfg, ROOT)
    if args.datasets:
        missing = set(args.datasets) - set(datasets)
        if missing:
            raise SystemExit(f"unknown dataset(s): {sorted(missing)}")
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config": cfg,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "datasets": {},
    }

    for name, ds in datasets.items():
        payload["datasets"][name] = run_dataset(name, ds, cfg, out_dir, seed)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    write_summary(payload, out_dir / "summary.csv")

    print(f"\n{'=' * 72}")
    print(f"wrote {out_dir / 'metrics.json'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

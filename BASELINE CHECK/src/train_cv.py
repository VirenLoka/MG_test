#!/usr/bin/env python3
"""
Doubly-disjoint cross-validated baselines: MACCS + AAC -> binary label.

Targets are split into K folds; for each fold the test set shares neither a
target nor a Bemis-Murcko scaffold with training. The target is described by
its 20-dimensional amino-acid composition rather than a one-hot identity,
because a one-hot block cannot represent a target it never saw in training.

Each fold trains a RandomForest and an XGBoost classifier, plus three
ablations (MACCS only / AAC only / both) so the contribution of the sequence
features under target-disjoint conditions is visible rather than assumed.

Usage:
    python "BASELINE CHECK/src/train_cv.py"
    python "BASELINE CHECK/src/train_cv.py" --datasets dmax --folds 5
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier  # noqa: E402

from cv_splits import add_scaffolds, audit_folds, make_fold, target_folds  # noqa: E402
from data import load_config, prepare_all  # noqa: E402
from features import build_split_features  # noqa: E402

ROOT = SRC.parent.parent
MODEL_KEYS = ("random_forest", "xgboost")


def metrics_for(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Metrics for one split; AUCs are None when the split is single-class."""
    y_pred = (y_prob >= threshold).astype(int)
    one_class = len(np.unique(y_true)) < 2
    return {
        "n": int(len(y_true)),
        "positive_rate_true": round(float(np.mean(y_true)), 4),
        "positive_rate_pred": round(float(np.mean(y_pred)), 4),
        "roc_auc": None if one_class else round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": None if one_class else round(float(average_precision_score(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": None if one_class else round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def build_rf(cfg, seed):
    params = dict(cfg["models"]["random_forest"])
    params["random_state"] = seed
    return RandomForestClassifier(**params)


def build_xgb(cfg, seed, y_train):
    params = dict(cfg["models"]["xgboost"])
    early = params.pop("early_stopping_rounds", None)
    if not cfg["cv"].get("xgb_early_stopping", True):
        early = None
    params["random_state"] = seed
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    params["scale_pos_weight"] = round(neg / pos, 4) if pos else 1.0
    return XGBClassifier(**params, early_stopping_rounds=early)


def fit_predict(model_key, cfg, seed, X, y, columns=None):
    """Fit one model on train and return probabilities for val and test."""
    sl = slice(None) if columns is None else columns

    def block(name):
        return X[name][:, sl]

    if model_key == "random_forest":
        model = build_rf(cfg, seed)
        model.fit(block("train"), y["train"])
    else:
        model = build_xgb(cfg, seed, y["train"])
        fit_kwargs = {"verbose": False}
        if getattr(model, "early_stopping_rounds", None):
            fit_kwargs["eval_set"] = [(block("val"), y["val"])]
        model.fit(block("train"), y["train"], **fit_kwargs)
    return model, {
        name: model.predict_proba(block(name))[:, 1] for name in ("val", "test")
    }


def aggregate(per_fold: list[dict]) -> dict:
    """Mean and std across folds, skipping folds where a metric is undefined."""
    keys = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1", "mcc")
    out = {}
    for k in keys:
        vals = [f[k] for f in per_fold if f.get(k) is not None]
        if vals:
            out[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "n_folds": len(vals),
            }
        else:
            out[k] = None
    return out


def run_dataset(name, dataset, cfg, out_dir: Path, n_folds: int, seed: int) -> dict:
    cols = cfg["columns"]
    smiles_col, target_col, seq_col = cols["smiles"], cols["target"], cols["sequence"]
    scaffold_col = cfg["cv"]["scaffold_col"]
    label_col = dataset.label_col
    encoding = cfg["cv"]["target_encoding"]

    print(f"\n{'=' * 74}\n{name.upper()}  (target-disjoint + scaffold-disjoint, K={n_folds})\n{'=' * 74}")

    frame, scaffold_report = add_scaffolds(dataset.frame, smiles_col, scaffold_col)
    assignment = target_folds(frame, target_col, n_folds, seed)

    ds_out = out_dir / name
    ds_out.mkdir(parents=True, exist_ok=True)

    fold_records, pooled = [], []
    for fold in range(n_folds):
        print(f"\n  --- fold {fold} ---")
        splits, split_meta = make_fold(
            frame, fold, assignment, n_folds, target_col, scaffold_col
        )
        if any(len(splits[s]) == 0 for s in ("train", "val", "test")):
            print("    skipped: a split came out empty")
            continue

        audit = audit_folds(splits, target_col, scaffold_col, smiles_col, label_col)
        print(
            f"    retention {split_meta['retention']:.1%}"
            f" ({split_meta['rows_before']} -> {split_meta['rows_after']} rows)"
        )

        X, feature_names, kept, feat_report = build_split_features(
            splits, smiles_col, target_col,
            drop_bit0=cfg["features"]["drop_maccs_bit0"],
            target_encoding=encoding, sequence_col=seq_col,
        )
        y = {k: kept[k][label_col].to_numpy() for k in kept}
        n_maccs = feat_report["n_maccs_bits"]

        blocks = {
            "maccs_only": slice(0, n_maccs),
            f"{encoding}_only": slice(n_maccs, None),
            f"maccs_plus_{encoding}": slice(None),
        }

        record = {
            "fold": fold,
            "split_meta": split_meta,
            "audit": audit,
            "feature_report": feat_report,
            "models": {},
            "ablation": {},
        }

        for model_key in MODEL_KEYS:
            model, probs = fit_predict(model_key, cfg, seed, X, y)
            record["models"][model_key] = {
                split: metrics_for(y[split], probs[split]) for split in ("val", "test")
            }
            m = record["models"][model_key]["test"]
            print(
                f"    {model_key:14s} test n={m['n']:4d}"
                f" roc_auc={m['roc_auc']} pr_auc={m['pr_auc']}"
                f" bal_acc={m['balanced_accuracy']} mcc={m['mcc']}"
            )
            if model_key == "random_forest":
                pooled.append(
                    pd.DataFrame({
                        "fold": fold,
                        "target": kept["test"][target_col].to_numpy(),
                        "smiles": kept["test"][smiles_col].to_numpy(),
                        "y_true": y["test"],
                        "prob": probs["test"],
                    })
                )

        # Ablations use RandomForest only; the two models track each other closely.
        for label, sl in blocks.items():
            _, probs = fit_predict("random_forest", cfg, seed, X, y, columns=sl)
            record["ablation"][label] = metrics_for(y["test"], probs["test"])
        abl = record["ablation"]
        print("    ablation (RF test roc_auc): " + "  ".join(
            f"{k}={abl[k]['roc_auc']}" for k in blocks
        ))

        fold_records.append(record)

    summary = {
        model_key: aggregate([r["models"][model_key]["test"] for r in fold_records])
        for model_key in MODEL_KEYS
    }
    ablation_summary = {
        label: aggregate([r["ablation"][label] for r in fold_records])
        for label in fold_records[0]["ablation"]
    } if fold_records else {}

    pooled_metrics = None
    if pooled:
        pooled_df = pd.concat(pooled, ignore_index=True)
        pooled_df.to_csv(ds_out / "cv_pooled_predictions_random_forest.csv", index=False)
        pooled_metrics = metrics_for(
            pooled_df["y_true"].to_numpy(), pooled_df["prob"].to_numpy()
        )
        # Per-target breakdown: which targets the model actually fails on.
        rows = []
        for tgt, grp in pooled_df.groupby("target"):
            single = grp["y_true"].nunique() < 2
            rows.append({
                "target": tgt,
                "n": len(grp),
                "positive_rate": round(float(grp["y_true"].mean()), 4),
                "mean_pred_prob": round(float(grp["prob"].mean()), 4),
                "roc_auc": None if single else round(
                    float(roc_auc_score(grp["y_true"], grp["prob"])), 4
                ),
                "accuracy": round(
                    float(accuracy_score(grp["y_true"], (grp["prob"] >= 0.5).astype(int))), 4
                ),
            })
        per_target = sorted(rows, key=lambda r: -r["n"])
        pd.DataFrame(per_target).to_csv(ds_out / "cv_per_target.csv", index=False)
    else:
        per_target = []

    print(f"\n  == {name} across {len(fold_records)} folds (held-out targets) ==")
    for model_key in MODEL_KEYS:
        s = summary[model_key]
        print(
            f"    {model_key:14s}"
            f" roc_auc={s['roc_auc']['mean']:.4f}+/-{s['roc_auc']['std']:.4f}"
            f" (min {s['roc_auc']['min']:.3f}, max {s['roc_auc']['max']:.3f})"
            f"  mcc={s['mcc']['mean']:.4f}+/-{s['mcc']['std']:.4f}"
        )
    print("    ablation means (RF test roc_auc):")
    for label, agg in ablation_summary.items():
        if agg["roc_auc"]:
            print(
                f"      {label:22s} {agg['roc_auc']['mean']:.4f}"
                f" +/-{agg['roc_auc']['std']:.4f}"
            )
    if pooled_metrics:
        print(
            f"    pooled out-of-fold (RF): roc_auc={pooled_metrics['roc_auc']}"
            f" bal_acc={pooled_metrics['balanced_accuracy']}"
            f" mcc={pooled_metrics['mcc']}"
        )

    return {
        "dataset": name,
        "label_col": label_col,
        "target_encoding": encoding,
        "data_report": dataset.report,
        "scaffold_report": scaffold_report,
        "target_fold_assignment": assignment,
        "folds": fold_records,
        "summary_across_folds": summary,
        "ablation_across_folds": ablation_summary,
        "pooled_out_of_fold_random_forest": pooled_metrics,
        "per_target_random_forest": per_target,
    }


def write_summary(payload: dict, path: Path) -> None:
    rows = []
    for ds_name, ds in payload["datasets"].items():
        for model_key in MODEL_KEYS:
            for r in ds["folds"]:
                m = r["models"][model_key]["test"]
                rows.append({
                    "dataset": ds_name, "model": model_key, "fold": r["fold"],
                    "test_targets": ";".join(r["audit"]["per_split"]["test"]["targets"]),
                    **{k: m[k] for k in
                       ("n", "positive_rate_true", "roc_auc", "pr_auc",
                        "accuracy", "balanced_accuracy", "f1", "mcc")},
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(SRC.parent / "config" / "baseline.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_folds = args.folds or cfg["cv"]["n_folds"]
    seed = args.seed if args.seed is not None else cfg["cv"]["seed"]

    out_dir = ROOT / cfg["paths"]["outputs"] / "cv_target_disjoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing datasets\n" + "-" * 74)
    datasets = prepare_all(cfg, ROOT)
    if args.datasets:
        missing = set(args.datasets) - set(datasets)
        if missing:
            raise SystemExit(f"unknown dataset(s): {sorted(missing)}")
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_folds": n_folds,
        "seed": seed,
        "split_design": "targets partitioned into K folds; scaffold-disjointness "
                        "imposed by dropping rows whose scaffold belongs to another split",
        "config": cfg,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "datasets": {},
    }
    for name, ds in datasets.items():
        payload["datasets"][name] = run_dataset(name, ds, cfg, out_dir, n_folds, seed)

    with open(out_dir / "cv_metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    write_summary(payload, out_dir / "cv_summary.csv")

    print(f"\n{'=' * 74}")
    print(f"wrote {out_dir / 'cv_metrics.json'}")
    print(f"wrote {out_dir / 'cv_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Binary classification metrics, matching the baseline package's definitions.

Threshold-free metrics (ROC-AUC, PR-AUC) are undefined on a single-class split
and return None rather than raising, because target-disjoint folds genuinely
produce them: dmax `CDK2` is 1.8% positive, and a small fold can hold one class.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
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

ALL_METRICS = (
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
    "precision", "recall", "f1", "mcc",
)


def compute_metrics(
    y_true, y_prob, threshold: float = 0.5, which=ALL_METRICS
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2

    out: dict = {
        "n": int(len(y_true)),
        "positive_rate_true": round(float(y_true.mean()), 4) if len(y_true) else None,
        "positive_rate_pred": round(float(y_pred.mean()), 4) if len(y_pred) else None,
    }
    funcs = {
        "roc_auc": lambda: None if single_class else roc_auc_score(y_true, y_prob),
        "pr_auc": lambda: None if single_class else average_precision_score(y_true, y_prob),
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(y_true, y_pred),
        "precision": lambda: precision_score(y_true, y_pred, zero_division=0),
        "recall": lambda: recall_score(y_true, y_pred, zero_division=0),
        "f1": lambda: f1_score(y_true, y_pred, zero_division=0),
        "mcc": lambda: None if single_class else matthews_corrcoef(y_true, y_pred),
    }
    for name in which:
        if name not in funcs:
            raise ValueError(f"unknown metric {name!r}; choose from {ALL_METRICS}")
        value = funcs[name]()
        out[name] = None if value is None else round(float(value), 4)

    if not single_class or len(np.unique(y_pred)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return out


def aggregate_folds(per_fold: list[dict], which=ALL_METRICS) -> dict:
    """
    Mean/std/min/max across folds, skipping folds where a metric is undefined.

    The std is the number that matters for these splits: the baselines'
    target-disjoint ROC-AUC had a std of ~0.20 on a [0,1] metric, which made
    the mean close to meaningless on its own.
    """
    out = {}
    for name in which:
        values = [f[name] for f in per_fold if f.get(name) is not None]
        out[name] = (
            {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "n_folds": len(values),
            }
            if values else None
        )
    return out

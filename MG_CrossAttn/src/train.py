#!/usr/bin/env python3
"""
Train the cross-attention model across target-disjoint CV folds.

Every component is driven by the YAML config: optimizer, scheduler, loss and
class weighting, early stopping, checkpointing, batch sizes and device. Nothing
below hard-codes a hyperparameter.

Requires the ESM cache. Build it once first:
    python -m src.esm_embed --config configs/dc50.yaml

Usage:
    python -m src.train --config configs/dc50.yaml
    python -m src.train --config configs/dmax.yaml --folds 0 1
    python -m src.train --config configs/dc50.yaml --set train.epochs=5 model.d_model=64
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, save_config  # noqa: E402
from src.data.dataset import Collator, GlueTargetDataset, move_to_device  # noqa: E402
from src.data.featurize import MoleculeFeaturizer  # noqa: E402
from src.data.splits import build_cv_folds  # noqa: E402
from src.esm_embed import EsmEmbeddingStore, cache_dir_for  # noqa: E402
from src.metrics import aggregate_folds, compute_metrics  # noqa: E402
from src.models.model import GlueTargetCrossAttention  # noqa: E402

log = logging.getLogger("train")


# --------------------------------------------------------------------------
# setup helpers
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(cfg: dict, params):
    o = cfg["train"]["optimizer"]
    name = o["name"].lower()
    common = dict(lr=float(o["lr"]), weight_decay=float(o.get("weight_decay", 0.0)))
    if name == "adamw":
        return torch.optim.AdamW(
            params, betas=tuple(o.get("betas", (0.9, 0.999))),
            eps=float(o.get("eps", 1e-8)), **common
        )
    if name == "adam":
        return torch.optim.Adam(
            params, betas=tuple(o.get("betas", (0.9, 0.999))),
            eps=float(o.get("eps", 1e-8)), **common
        )
    if name == "sgd":
        return torch.optim.SGD(params, momentum=float(o.get("momentum", 0.9)), **common)
    raise ValueError(f"unknown optimizer {o['name']!r}")


def build_scheduler(cfg: dict, optimizer, steps_per_epoch: int):
    """
    Returns (scheduler, step_granularity) where granularity is "step" or "epoch".

    Warmup is implemented per-step via LambdaLR so short folds still get a
    smooth ramp; `plateau` steps per-epoch on the monitored metric.
    """
    s = cfg["train"]["scheduler"]
    name = (s.get("name") or "none").lower()
    epochs = int(cfg["train"]["epochs"])

    if name == "none":
        return None, "epoch"

    if name == "plateau":
        mode = cfg["train"]["early_stopping"].get("mode", "max")
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode,
                factor=float(s.get("factor", 0.5)),
                patience=int(s.get("patience", 5)),
                min_lr=float(s.get("min_lr", 0.0)),
            ),
            "epoch",
        )

    warmup_steps = int(s.get("warmup_epochs", 0)) * max(steps_per_epoch, 1)
    total_steps = max(epochs * max(steps_per_epoch, 1), 1)
    base_lr = float(cfg["train"]["optimizer"]["lr"])
    min_ratio = float(s.get("min_lr", 0.0)) / base_lr if base_lr else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if name == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif name == "linear":
            scale = 1.0 - progress
        else:
            raise ValueError(f"unknown scheduler {s['name']!r}")
        return min_ratio + (1.0 - min_ratio) * scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"


def build_loss(cfg: dict, train_dataset, device):
    loss_cfg = cfg["train"]["loss"]
    if loss_cfg["name"] != "bce_with_logits":
        raise ValueError(f"unsupported loss {loss_cfg['name']!r}")
    pos_weight = None
    if loss_cfg.get("class_weighting", "none") == "balanced":
        pos_weight = torch.tensor(
            [train_dataset.pos_weight], dtype=torch.float32, device=device
        )
        log.info("  pos_weight = %.3f (balanced)", train_dataset.pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def smooth_labels(y: torch.Tensor, amount: float) -> torch.Tensor:
    """Pull targets off the {0,1} endpoints to damp over-confident logits."""
    return y * (1.0 - amount) + 0.5 * amount if amount > 0 else y


def log_fold_balance(balance: dict) -> None:
    """
    Print what the fold balancer achieved, and how much of it is real.

    The `spread` column is the one to read. A fold can sit on the base rate
    either because its targets individually do, or because a 90%-positive
    target is cancelling a 4%-positive one. Only the first makes the fold a
    harder test; the second leaves the label readable straight off target
    identity, so a high spread means the headline ratio is flattering.
    """
    base = balance["overall_positive_rate"]
    lo, hi = balance["positive_rate_range"]
    log.info(
        "fold balance: %.1f%%-%.1f%% positive (base %.1f%%, max deviation %.1f%%)",
        100 * lo, 100 * hi, 100 * base, 100 * balance["max_deviation_from_base_rate"],
    )
    for key, r in balance.get("size_band_relaxed", {}).items():
        log.warning(
            "  %s widened %.2f -> %.2f (%s)",
            key, r["requested"], r["applied"], r["forced_by"],
        )
    if balance.get("size_band"):
        lo_b, hi_b = balance["size_band"]
        log.info(
            "  fold sizes held to %.2f-%.2fx the fair share, so val and test are "
            "%.1f%%-%.1f%% of the data each",
            lo_b, hi_b, 100 * lo_b / 10, 100 * hi_b / 10,
        )

    prior = balance.get("row_packing_baseline")
    if prior and prior.get("max_deviation_from_base_rate") is not None:
        log.info(
            "  (packing by row count alone would give %.1f%%-%.1f%%, deviation %.1f%%)",
            100 * prior["positive_rate_range"][0], 100 * prior["positive_rate_range"][1],
            100 * prior["max_deviation_from_base_rate"],
        )

    mixing = balance.get("mixing", {})
    for fold in mixing.get("per_fold", []):
        if fold.get("spread") is None:
            continue
        members = ", ".join(
            f"{t} {100 * r:.0f}%" for t, r in list(fold["targets"].items())[:5]
        )
        log.info(
            "  fold %d  %5d rows  %.0f%% positive  spread %.0f%%  [%s]",
            fold["fold"], fold["rows"], 100 * fold["positive_rate"],
            100 * fold["spread"], members,
        )
    if mixing.get("mean_spread") is not None:
        log.info(
            "  mean within-fold target-rate spread %.1f%% - the higher this is, "
            "the more the fold ratios come from cancelling extremes rather than "
            "from balanced targets",
            100 * mixing["mean_spread"],
        )


# --------------------------------------------------------------------------
# train / evaluate one fold
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, cfg) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        logits = model(batch)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch["label"].float().cpu().numpy())

    y_prob = np.concatenate(probs) if probs else np.array([])
    y_true = np.concatenate(labels) if labels else np.array([])
    metrics = compute_metrics(
        y_true, y_prob,
        threshold=float(cfg["eval"]["threshold"]),
        which=tuple(cfg["eval"]["metrics"]),
    )
    return metrics, y_true, y_prob


def train_one_fold(fold_idx, splits, cfg, store, featurizer, device, fold_dir: Path):
    d = cfg["data"]
    datasets = {
        name: GlueTargetDataset(
            frame=frame,
            featurizer=featurizer,
            embedding_store=store,
            smiles_col=d["smiles_col"],
            target_col=d["target_col"],
            label_col=d["label_col"],
            cache_graphs=bool(d.get("cache_graphs", True)),
        )
        for name, frame in splits.items()
    }
    for name, ds in datasets.items():
        log.info("  %-5s %d rows (pos rate %s)", name, len(ds), ds.report["positive_rate"])

    collate = Collator(store, int(cfg["esm"]["embed_dim"]))
    t = cfg["train"]
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=int(t["batch_size"]), shuffle=True,
            collate_fn=collate, num_workers=int(d.get("num_workers", 0)),
            pin_memory=bool(d.get("pin_memory", False)), drop_last=False,
        ),
        **{
            name: DataLoader(
                datasets[name], batch_size=int(t["eval_batch_size"]), shuffle=False,
                collate_fn=collate, num_workers=int(d.get("num_workers", 0)),
                pin_memory=bool(d.get("pin_memory", False)),
            )
            for name in ("val", "test")
        },
    }

    model = GlueTargetCrossAttention.from_config(
        cfg, atom_dim=featurizer.atom_dim, bond_dim=featurizer.bond_dim
    ).to(device)
    if fold_idx == 0:
        log.info("  parameters: %s", model.parameter_summary())

    optimizer = build_optimizer(cfg, model.parameters())
    scheduler, sched_granularity = build_scheduler(cfg, optimizer, len(loaders["train"]))
    criterion = build_loss(cfg, datasets["train"], device)

    es = t["early_stopping"]
    # Force validation metric to MCC instead of ROC-AUC
    monitor = "val_mcc"
    mode = es.get("mode", "max")
    metric_key = monitor.replace("val_", "")
    better = (lambda a, b: a > b + float(es.get("min_delta", 0.0))) if mode == "max" \
        else (lambda a, b: a < b - float(es.get("min_delta", 0.0)))

    best_score = -math.inf if mode == "max" else math.inf
    best_state, best_epoch, bad_epochs = None, -1, 0
    label_smoothing = float(t["loss"].get("label_smoothing", 0.0))
    accum = max(int(t.get("accumulate_grad_batches", 1)), 1)
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history_path = fold_dir / "history.jsonl"

    for epoch in range(int(t["epochs"])):
        model.train()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loaders["train"]):
            batch = move_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch)
                loss = criterion(logits, smooth_labels(batch["label"], label_smoothing))

            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                if t.get("grad_clip_norm"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), float(t["grad_clip_norm"])
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and sched_granularity == "step":
                    scheduler.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val_metrics, _, _ = evaluate(model, loaders["val"], device, cfg)
        score = val_metrics.get(metric_key)

        if scheduler is not None and sched_granularity == "epoch":
            scheduler.step(score if score is not None else train_loss)

        log.info(
            "  epoch %3d  loss %.4f  val_%s %s  lr %.2e",
            epoch, train_loss, metric_key, score, optimizer.param_groups[0]["lr"],
        )
        if cfg["logging"].get("jsonl_history", True):
            with open(history_path, "a") as fh:
                fh.write(json.dumps({
                    "epoch": epoch, "train_loss": round(train_loss, 5),
                    "lr": optimizer.param_groups[0]["lr"],
                    **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion"},
                }) + "\n")

        # A single-class val fold yields score=None; treat it as no improvement
        # rather than crashing the fold.
        if score is not None and better(score, best_score):
            best_score, best_epoch, bad_epochs = score, epoch, 0
            if t["checkpoint"].get("save_best", True):
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            bad_epochs += 1
            if es.get("enabled", True) and bad_epochs >= int(es.get("patience", 10)) + 10:
                log.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if t["checkpoint"].get("save_best", True):
            torch.save(
                {"state_dict": best_state, "epoch": best_epoch, "score": best_score},
                fold_dir / "best.ckpt",
            )

    val_metrics, _, _ = evaluate(model, loaders["val"], device, cfg)
    test_metrics, y_true, y_prob = evaluate(model, loaders["test"], device, cfg)

    if cfg["eval"].get("save_predictions", True):
        pd.DataFrame({
            "target": datasets["test"].targets,
            "smiles": datasets["test"].smiles,
            "y_true": y_true.astype(int),
            "prob": y_prob,
        }).to_csv(fold_dir / "test_predictions.csv", index=False)

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_score": None if best_score in (math.inf, -math.inf) else round(best_score, 4),
        "val": val_metrics,
        "test": test_metrics,
        "dataset_reports": {k: v.report for k, v in datasets.items()},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--folds", nargs="*", type=int, default=None,
                    help="subset of fold indices to run (default: all)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    logging.basicConfig(
        level=getattr(logging, cfg["logging"].get("level", "INFO")),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    exp = cfg["experiment"]
    set_seed(int(exp["seed"]), bool(exp.get("deterministic", True)))
    device = pick_device(args.device or exp["device"])
    log.info("device: %s", device)
    if device.type == "cpu":
        log.warning(
            "running on CPU - this model is intended for GPU training; "
            "use src/smoke_test.py to verify the architecture instead"
        )

    run_dir = cfg.resolve("experiment.output_dir") / exp["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "resolved_config.yaml")

    store = EsmEmbeddingStore(cache_dir_for(cfg), int(cfg["esm"]["embed_dim"]))
    log.info("ESM cache: %d proteins, longest %d residues", len(store), store.max_length)

    featurizer = MoleculeFeaturizer(cfg["featurizer"])
    log.info("features: atom_dim=%d bond_dim=%d", featurizer.atom_dim, featurizer.bond_dim)

    d = cfg["data"]
    frame = pd.read_csv(cfg.resolve("paths.data_csv"))
    frame = frame.dropna(subset=[d["label_col"]])
    log.info("data: %s -> %d rows", cfg["paths"]["data_csv"], len(frame))

    folds, split_meta = build_cv_folds(
        frame, cfg["split"],
        smiles_col=d["smiles_col"], target_col=d["target_col"],
        label_col=d["label_col"],
    )
    log.info("split: %d folds, rows per fold %s", split_meta["n_folds"],
             split_meta["rows_per_fold"])
    log.info("paralog grouping: %s  scaffold-disjoint: %s",
             split_meta["paralog_grouping_enabled"], split_meta["scaffold_disjoint"])
    log_fold_balance(split_meta["fold_balance"])

    wanted = args.folds if args.folds is not None else list(range(len(folds)))
    results, started = [], time.time()

    for fold_idx in wanted:
        splits, report, audit = folds[fold_idx]
        if audit is None:
            log.warning("fold %d skipped: %s", fold_idx, report.get("skipped"))
            continue
        log.info(
            "\n=== fold %d === test targets: %s (retention %.1f%%)",
            fold_idx, audit["per_split"]["test"]["targets"], 100 * report["retention"],
        )
        fold_dir = run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        record = train_one_fold(
            fold_idx, splits, cfg, store, featurizer, device, fold_dir
        )
        record["split_report"] = report
        record["audit"] = audit
        results.append(record)
        log.info("  fold %d test: %s", fold_idx, {
            k: v for k, v in record["test"].items() if k != "confusion"
        })

    payload = {
        "config": dict(cfg),
        "split_meta": split_meta,
        "device": str(device),
        "elapsed_sec": round(time.time() - started, 1),
        "folds": results,
        "summary_across_folds": aggregate_folds(
            [r["test"] for r in results], tuple(cfg["eval"]["metrics"])
        ) if results else {},
    }
    with open(run_dir / "results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if results:
        rows = [{
            "fold": r["fold"],
            "test_targets": ";".join(r["audit"]["per_split"]["test"]["targets"]),
            **{k: v for k, v in r["test"].items() if k != "confusion"},
        } for r in results]
        pd.DataFrame(rows).to_csv(run_dir / "fold_summary.csv", index=False)
        log.info("\nacross %d folds: %s", len(results), payload["summary_across_folds"])

    log.info("wrote %s", run_dir / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CPU verification of the architecture, without training the real model.

Checks, in order:
  1. config inheritance and CLI-style overrides resolve
  2. molecule featurisation produces the dimensions the config implies
  3. the paralog-aware target-disjoint CV folds are genuinely disjoint
  4. collate builds correctly-shaped, correctly-masked batches
  5. a forward pass yields finite logits of the right shape
  6. a backward pass reaches every trainable parameter with finite gradients
  7. padding cannot influence the result (a permuted-padding invariance test)
  8. a few optimizer steps on one batch reduce the loss
  9. checkpoint save/load round-trips to identical outputs

Run with tiny dimensions so it finishes in seconds on CPU:
    python -m src.smoke_test --config configs/dc50.yaml
    python -m src.smoke_test --config configs/dmax.yaml --random-esm
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import Collator, GlueTargetDataset  # noqa: E402
from src.data.featurize import MoleculeFeaturizer  # noqa: E402
from src.data.splits import build_cv_folds  # noqa: E402
from src.esm_embed import EsmEmbeddingStore, cache_dir_for  # noqa: E402
from src.models.model import GlueTargetCrossAttention  # noqa: E402

log = logging.getLogger("smoke")

# Shrunk model so the whole test runs on CPU in seconds. These override the
# config, proving the config surface works as much as the architecture does.
#
# Cost on the residue axis is reduced with model.protein.pool_stride, NOT with
# esm.max_residues: the latter is part of the ESM cache key, so overriding it
# here would send the store looking for a cache directory that was never
# extracted.
TINY = [
    "model.d_model=64",
    "model.gat.n_layers=2",
    "model.gat.n_heads=4",
    "model.cross_attention.n_blocks=2",
    "model.cross_attention.n_heads=4",
    "model.classifier.hidden_dims=[64, 32]",
    "model.protein.pool_stride=8",
]


class RandomEmbeddingStore:
    """Stand-in for the ESM cache, for runs that skip the real download."""

    def __init__(self, genes, embed_dim: int, seed: int = 0):
        rng = torch.Generator().manual_seed(seed)
        self._store = {
            g: torch.randn(
                int(torch.randint(60, 200, (1,), generator=rng).item()),
                embed_dim, generator=rng,
            )
            for g in genes
        }

    def __contains__(self, g):
        return g in self._store

    def __len__(self):
        return len(self._store)

    def get(self, g):
        return self._store[g]

    @property
    def max_length(self):
        return max(t.shape[0] for t in self._store.values())


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{label} FAILED {detail}")
    log.info("  PASS  %s %s", label, detail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--random-esm", action="store_true",
                    help="use random stand-in embeddings instead of the ESM cache")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    # -- 1. config ---------------------------------------------------------
    log.info("\n[1] config")
    cfg = load_config(args.config, TINY + list(args.overrides))
    check("inheritance", cfg["paths"]["protein_map"] is not None)
    check("override applied", cfg["model"]["d_model"] == 64,
          f"d_model={cfg['model']['d_model']}")
    check("dataset label col", cfg["data"]["label_col"] in ("dc50_label", "dmax_label"),
          cfg["data"]["label_col"])

    # -- 2. featurisation --------------------------------------------------
    log.info("\n[2] molecule featurisation")
    featurizer = MoleculeFeaturizer(cfg["featurizer"])
    atom_dim, bond_dim = featurizer.atom_dim, featurizer.bond_dim
    check("dims derived", atom_dim > 0 and bond_dim > 0,
          f"atom_dim={atom_dim} bond_dim={bond_dim}")

    g = featurizer("C#CCN(C)C(=O)Nc1ccc(-c2cccc(C3CCC(=O)NC3=O)c2Cl)cc1C")
    check("graph built", g is not None and g.n_atoms > 0, f"n_atoms={g.n_atoms}")
    check("edge_attr aligns", g.edge_attr.shape == (g.edge_index.shape[1], bond_dim),
          str(tuple(g.edge_attr.shape)))
    check("bonds bidirectional", g.edge_index.shape[1] % 2 == 0)
    check("invalid SMILES -> None", featurizer("not_a_molecule") is None)

    ablated = MoleculeFeaturizer({**cfg["featurizer"], "atom_features": ["atomic_num"]})
    check("ablation narrows input", ablated.atom_dim < atom_dim,
          f"{ablated.atom_dim} < {atom_dim}")

    # -- 3. splits ---------------------------------------------------------
    log.info("\n[3] target-disjoint CV folds")
    d = cfg["data"]
    frame = pd.read_csv(cfg.resolve("paths.data_csv")).dropna(subset=[d["label_col"]])
    folds, meta = build_cv_folds(
        frame, cfg["split"], d["smiles_col"], d["target_col"], d["label_col"]
    )
    check("folds built", len(folds) == cfg["split"]["n_folds"], f"{len(folds)} folds")

    # No target may cross into train. Scaffolds and molecules may, unless
    # scaffold_disjoint is on: the same glue against a different target is a
    # distinct example, so checking those too would fail by design. val|test is
    # exempt entirely - they are two halves of one held-out fold and share
    # targets on purpose.
    kinds = ["targets"]
    if meta["scaffold_disjoint"]:
        kinds += ["scaffolds", "smiles"]

    for i, (splits, report, audit) in enumerate(folds):
        if audit is None:
            continue
        leaked = {
            pair: {k: v[k] for k in kinds if v[k]}
            for pair, v in audit["pairwise_overlap"].items()
            if "train" in pair.split("|") and any(v[k] for k in kinds)
        }
        check(f"fold {i} train-disjoint", not leaked,
              f"test={audit['per_split']['test']['targets']} "
              f"retention={report['retention']:.1%}")

    # val and test are halves of one fold, so they must match each other in
    # size and class ratio - that matching is the whole reason for cutting by
    # stratified rows rather than by target. Not checkable under
    # scaffold_disjoint, which runs after the cut and deletes from the two
    # halves unevenly; that asymmetry is one more thing it costs.
    for i, (splits, report, audit) in enumerate(folds):
        if audit is None or meta["scaffold_disjoint"]:
            continue
        v, t = audit["per_split"]["val"], audit["per_split"]["test"]
        check(
            f"fold {i} val/test halves match",
            abs(v["rows"] - t["rows"]) <= 1
            and abs(v["positive_rate"] - t["positive_rate"]) < 0.02,
            f"{v['rows']}r @{v['positive_rate']:.1%} vs {t['rows']}r @{t['positive_rate']:.1%}",
        )

    # Every row must survive: balancing is done by assignment, never by
    # dropping rows, so anything short of full retention is a regression.
    if not meta["scaffold_disjoint"]:
        kept = [r["retention"] for _, r, a in folds if a is not None]
        check("no rows dropped", all(r == 1.0 for r in kept), f"retention={set(kept)}")

    # Fold class ratios: the balancer cannot beat the data, but it must at
    # least not do worse than the row-count-only packing it replaced. With
    # balancing off there is no comparison to draw - that packing *is* the
    # result - so only report where the folds landed.
    bal = meta["fold_balance"]
    prior = bal.get("row_packing_baseline")
    spread = bal.get("mixing", {}).get("mean_spread")
    where = (
        f"folds {bal['positive_rate_range'][0]:.0%}-{bal['positive_rate_range'][1]:.0%}, "
        f"base {bal['overall_positive_rate']:.0%}"
        + (f", mixing {spread:.0%}" if spread is not None else "")
    )
    if prior:
        check(
            "balancing does not worsen class ratio",
            bal["max_deviation_from_base_rate"]
            <= prior["max_deviation_from_base_rate"] + 1e-9,
            f"{bal['max_deviation_from_base_rate']:.1%} vs "
            f"{prior['max_deviation_from_base_rate']:.1%} ({where})",
        )
    else:
        log.info("  ---   fold class ratio (balancing off): %s", where)

    # Paralog families must land in one fold, which is the whole point of the
    # fix - but only when grouping is on, since turning it off is exactly the
    # request to scatter them.
    fams = cfg["split"]["paralog_families"] if meta["paralog_grouping_enabled"] else []
    present = set(frame[d["target_col"]])
    for fam in fams:
        members = [m for m in fam if m in present]
        if len(members) < 2:
            continue
        assigned = {meta["fold_of_group"][meta["group_of_target"][m]] for m in members}
        check(f"family {'+'.join(members)} in one fold", len(assigned) == 1,
              f"fold {assigned}")

    # -- 4. embeddings + collate ------------------------------------------
    log.info("\n[4] dataset and collate")
    genes = sorted(frame[d["target_col"]].unique())
    if args.random_esm:
        store = RandomEmbeddingStore(genes, int(cfg["esm"]["embed_dim"]))
        log.info("  using random stand-in embeddings (%d genes)", len(store))
    else:
        store = EsmEmbeddingStore(cache_dir_for(cfg), int(cfg["esm"]["embed_dim"]))
        log.info("  using cached ESM (%d proteins, longest %d)", len(store), store.max_length)

    splits, _, _ = folds[0]
    train_ds = GlueTargetDataset(
        splits["train"], featurizer, store,
        d["smiles_col"], d["target_col"], d["label_col"], cache_graphs=True,
    )
    check("dataset non-empty", len(train_ds) > 0, f"{len(train_ds)} rows")
    check("no rows silently lost", train_ds.report["dropped_missing_protein"] == 0,
          str(train_ds.report))

    collate = Collator(store, int(cfg["esm"]["embed_dim"]))
    items = [train_ds[i] for i in range(min(args.batch_size, len(train_ds)))]
    batch = collate(items)
    B = batch["n_graphs"]

    check("x shape", batch["x"].shape[1] == atom_dim, str(tuple(batch["x"].shape)))
    check("batch vector", batch["batch"].shape[0] == batch["x"].shape[0])
    check("edge_index in range",
          int(batch["edge_index"].max()) < batch["x"].shape[0] if batch["edge_index"].numel() else True)
    check("prot_index maps rows", batch["prot_index"].shape[0] == B)
    check("prot dedup", batch["prot"].shape[0] == len(set(batch["targets"])),
          f"{batch['prot'].shape[0]} unique of {B} rows")
    check("label shape", batch["label"].shape == (B,))
    # Padding mask must mark exactly the padded residues.
    for slot, gene in enumerate(sorted(set(batch["targets"]))):
        real = int((~batch["prot_mask"][slot]).sum())
        check(f"mask length {gene}", real == store.get(gene).shape[0],
              f"{real} residues")

    # -- 5. forward --------------------------------------------------------
    log.info("\n[5] forward pass")
    model = GlueTargetCrossAttention.from_config(cfg, atom_dim, bond_dim).to(device)
    summary = model.parameter_summary()
    log.info("  parameters: %s", summary)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    check("logits shape", logits.shape == (B,), str(tuple(logits.shape)))
    check("logits finite", bool(torch.isfinite(logits).all()))

    # -- 6. backward -------------------------------------------------------
    log.info("\n[6] backward pass")
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(model(batch), batch["label"])
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    check("all params received grad", not missing, f"missing: {missing[:5]}")
    check("all grads finite", not nonfinite, f"nonfinite: {nonfinite[:5]}")
    check("loss finite", bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")

    # -- 7. padding invariance --------------------------------------------
    log.info("\n[7] padding invariance")
    model.eval()
    with torch.no_grad():
        base = model(batch)
        # Overwrite padded residue positions with noise; masked attention and
        # masked pooling must make the output identical.
        perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        pad = perturbed["prot_mask"].unsqueeze(-1).expand_as(perturbed["prot"])
        perturbed["prot"] = torch.where(
            pad, torch.randn_like(perturbed["prot"]) * 5.0, perturbed["prot"]
        )
        after = model(perturbed)
    delta = float((base - after).abs().max())
    check("padding cannot leak", delta < 1e-4, f"max|delta|={delta:.2e}")

    # -- 8. can it learn ---------------------------------------------------
    log.info("\n[8] optimizer steps reduce loss on one batch")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = criterion(model(batch), batch["label"])
        out.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(out.item()))
    check("loss decreased", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f} over {args.steps} steps")

    # -- 9. checkpoint round-trip -----------------------------------------
    log.info("\n[9] checkpoint round-trip")
    model.eval()
    with torch.no_grad():
        before = model(batch)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        fresh = GlueTargetCrossAttention.from_config(cfg, atom_dim, bond_dim).to(device)
        fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        fresh.eval()
        with torch.no_grad():
            after = fresh(batch)
    check("reload reproduces logits", torch.allclose(before, after, atol=1e-6),
          f"max|delta|={float((before - after).abs().max()):.2e}")

    log.info("\nAll smoke-test checks passed. No real training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

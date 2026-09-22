#!/usr/bin/env python3
"""
Extract and cache per-residue ESM-2 embeddings for every target protein.

There are only 27 distinct proteins in this dataset, so ESM is run exactly
once here and never again: the training loop loads the cache and never
instantiates the protein language model. That keeps ESM's ~2.6 GB of weights
out of the training process entirely and makes training feasible without a GPU
in the loop.

Per-residue embeddings (not mean-pooled) are required, because cross-attention
attends over the residue axis. At 650M/1280-dim the full cache for all 27
proteins is roughly 104 MB in fp32 and 52 MB in fp16.

Usage:
    python -m src.esm_embed --config configs/dc50.yaml
    python -m src.esm_embed --config configs/dc50.yaml --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

log = logging.getLogger("esm_embed")

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def cache_key(checkpoint: str, layer: int, max_residues: int | None) -> str:
    """A short digest so different checkpoints/layers never share a cache dir."""
    raw = f"{checkpoint}|layer={layer}|max={max_residues}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return f"{checkpoint.split('/')[-1]}_L{layer}_{digest}"


def cache_dir_for(cfg) -> Path:
    esm = cfg["esm"]
    key = cache_key(esm["checkpoint"], int(esm["layer"]), esm.get("max_residues"))
    return cfg.resolve("paths.esm_cache") / key


def load_sequences(cfg) -> dict[str, str]:
    """Gene symbol -> sequence, from the protein map."""
    path = cfg.resolve("paths.protein_map")
    df = pd.read_csv(path)
    key_col = cfg["data"]["protein_key_col"]
    seq_col = cfg["data"]["protein_seq_col"]

    df = df[[key_col, seq_col]].dropna()
    df[key_col] = df[key_col].astype(str).str.strip()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=[key_col], keep="first")
    return dict(zip(df[key_col], df[seq_col]))


def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract(cfg, force: bool = False, device: str | None = None) -> Path:
    """
    Run ESM over every target sequence and write one tensor file per protein.

    Each file holds a [L, embed_dim] tensor whose row count equals the
    sequence length: the tokenizer's BOS/EOS positions are stripped so residue
    index i in the sequence is row i in the tensor.
    """
    from transformers import AutoTokenizer, EsmModel

    esm_cfg = cfg["esm"]
    out_dir = cache_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    sequences = load_sequences(cfg)
    log.info("%d target sequences from the protein map", len(sequences))

    existing = {}
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text()).get("proteins", {})
        todo = {g: s for g, s in sequences.items() if g not in existing}
        if not todo:
            log.info("cache already complete at %s", out_dir)
            return out_dir
        log.info("%d already cached, %d to extract", len(existing), len(todo))
    else:
        todo = dict(sequences)

    dev = pick_device(device or cfg["experiment"]["device"])
    dtype = DTYPES[esm_cfg.get("cache_dtype", "float16")]
    layer = int(esm_cfg["layer"])
    max_residues = esm_cfg.get("max_residues")
    strip = bool(esm_cfg.get("strip_special_tokens", True))

    log.info("loading %s onto %s", esm_cfg["checkpoint"], dev)
    tokenizer = AutoTokenizer.from_pretrained(esm_cfg["checkpoint"])
    model = EsmModel.from_pretrained(
        esm_cfg["checkpoint"], output_hidden_states=(layer != -1)
    )
    model.eval().to(dev)

    embed_dim = int(model.config.hidden_size)
    if embed_dim != int(esm_cfg["embed_dim"]):
        raise ValueError(
            f"config esm.embed_dim={esm_cfg['embed_dim']} but checkpoint "
            f"{esm_cfg['checkpoint']} has hidden_size={embed_dim}"
        )

    records = dict(existing)
    for i, (gene, seq) in enumerate(sorted(todo.items()), 1):
        if max_residues:
            seq = seq[: int(max_residues)]

        batch = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)

        hidden = (
            out.last_hidden_state
            if layer == -1
            else out.hidden_states[layer]
        )[0]

        if strip:
            # ESM tokenizers prepend <cls> and append <eos>; drop both so the
            # cached row count equals len(seq).
            hidden = hidden[1 : 1 + len(seq)]

        if hidden.shape[0] != len(seq):
            raise RuntimeError(
                f"{gene}: cached {hidden.shape[0]} rows for a {len(seq)}-residue "
                "sequence; special-token handling is wrong"
            )

        tensor = hidden.to(torch.float32).cpu().to(dtype).contiguous()
        torch.save(tensor, out_dir / f"{gene}.pt")

        records[gene] = {
            "length": len(seq),
            "embed_dim": embed_dim,
            "dtype": str(dtype).replace("torch.", ""),
            "file": f"{gene}.pt",
        }
        log.info("[%d/%d] %-10s L=%4d -> %s", i, len(todo), gene, len(seq), tensor.shape)

    manifest = {
        "checkpoint": esm_cfg["checkpoint"],
        "layer": layer,
        "embed_dim": embed_dim,
        "cache_dtype": esm_cfg.get("cache_dtype", "float16"),
        "max_residues": max_residues,
        "strip_special_tokens": strip,
        "n_proteins": len(records),
        "proteins": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    total_mb = sum(
        (out_dir / r["file"]).stat().st_size for r in records.values()
    ) / 1e6
    log.info("cached %d proteins (%.1f MB) in %s", len(records), total_mb, out_dir)
    return out_dir


class EsmEmbeddingStore:
    """
    Read-only access to the cached embeddings, held in memory.

    The whole cache is well under a hundred megabytes and every batch needs
    several proteins, so loading once up front beats touching the filesystem
    per sample.
    """

    def __init__(self, cache_dir: Path, embed_dim: int, dtype=torch.float32):
        self.cache_dir = Path(cache_dir)
        self.embed_dim = int(embed_dim)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no ESM cache at {self.cache_dir}. Run:\n"
                "    python -m src.esm_embed --config <your config>"
            )
        self.manifest = json.loads(manifest_path.read_text())
        if int(self.manifest["embed_dim"]) != self.embed_dim:
            raise ValueError(
                f"cache embed_dim={self.manifest['embed_dim']} != config {self.embed_dim}"
            )
        self._store: dict[str, torch.Tensor] = {}
        for gene, rec in self.manifest["proteins"].items():
            tensor = torch.load(self.cache_dir / rec["file"], map_location="cpu", weights_only=True)
            self._store[gene] = tensor.to(dtype)

    def __contains__(self, gene: str) -> bool:
        return gene in self._store

    def __len__(self) -> int:
        return len(self._store)

    def get(self, gene: str) -> torch.Tensor:
        """[L, embed_dim] for one gene symbol."""
        try:
            return self._store[gene]
        except KeyError as exc:
            raise KeyError(
                f"no cached embedding for {gene!r}; cached: {sorted(self._store)}"
            ) from exc

    @property
    def max_length(self) -> int:
        return max(t.shape[0] for t in self._store.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--force", action="store_true", help="re-extract everything")
    ap.add_argument("--device", default=None, help="override experiment.device")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_config(args.config, args.overrides)
    extract(cfg, force=args.force, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

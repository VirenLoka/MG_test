#!/usr/bin/env python3
"""
Load the base dc50 / dmax datasets and attach target protein sequences.

The clean CSVs (patglue_dc50_clean.csv, patglue_dmax_clean.csv) carry a gene
symbol in `target` but no sequence, so every sequence is resolved by looking
the gene symbol up in protein_structures_mapped.csv. Rows whose target has no
mapped sequence are dropped and counted, never silently kept with a null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class Dataset:
    """One prepared dataset, plus the bookkeeping from preparing it."""

    name: str
    frame: pd.DataFrame
    label_col: str
    value_col: str
    report: dict = field(default_factory=dict)


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_protein_map(cfg: dict, root: Path) -> pd.DataFrame:
    """Gene symbol -> sequence lookup, de-duplicated on the gene symbol."""
    pm_cfg = cfg["protein_map"]
    pm = pd.read_csv(root / cfg["paths"]["protein_map"])

    key, seq, uni = pm_cfg["key"], pm_cfg["sequence"], pm_cfg["uniprot"]
    pm = pm[[key, uni, seq]].copy()
    pm[key] = pm[key].astype(str).str.strip()
    pm[seq] = pm[seq].astype(str).str.strip().str.upper()

    before = len(pm)
    pm = pm.drop_duplicates(subset=[key], keep="first")
    if len(pm) != before:
        print(f"  [protein map] dropped {before - len(pm)} duplicate gene symbols")

    return pm.rename(
        columns={key: "target", uni: "target_uniprot", seq: "target_seq"}
    )


def prepare(name: str, cfg: dict, root: Path, protein_map: pd.DataFrame) -> Dataset:
    """Read one dataset, join sequences, and normalise the key columns."""
    ds_cfg = cfg["datasets"][name]
    cols = cfg["columns"]
    smiles_col, target_col = cols["smiles"], cols["target"]
    label_col, value_col = ds_cfg["label_col"], ds_cfg["value_col"]

    df = pd.read_csv(root / ds_cfg["csv"])
    n_raw = len(df)

    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()

    # Drop anything without the two things every row needs: a structure and a label.
    df = df[df[smiles_col].ne("") & df[smiles_col].ne("nan")]
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    n_labelled = len(df)

    # Every sequence comes from the protein map — the source CSVs have none.
    merged = df.merge(protein_map, on="target", how="left")
    unmapped = merged[merged["target_seq"].isna()]
    unmapped_targets = sorted(unmapped["target"].unique().tolist())
    merged = merged[merged["target_seq"].notna()].reset_index(drop=True)

    report = {
        "rows_raw": n_raw,
        "rows_after_label_filter": n_labelled,
        "rows_final": len(merged),
        "rows_dropped_unmapped_target": int(len(unmapped)),
        "unmapped_targets": unmapped_targets,
        "n_unique_smiles": int(merged[smiles_col].nunique()),
        "n_unique_targets": int(merged["target"].nunique()),
        "n_unique_pairs": int(merged.groupby([smiles_col, "target"]).ngroups),
        "label_counts": merged[label_col].value_counts().sort_index().to_dict(),
        "positive_rate": round(float(merged[label_col].mean()), 4),
    }

    print(f"[{name}] {n_raw} raw -> {len(merged)} usable rows")
    print(
        f"  {report['n_unique_smiles']} unique SMILES,"
        f" {report['n_unique_targets']} targets,"
        f" {report['n_unique_pairs']} unique (SMILES, target) pairs"
    )
    print(f"  label balance: {report['label_counts']}"
          f" (positive rate {report['positive_rate']})")
    if unmapped_targets:
        print(
            f"  WARNING: dropped {len(unmapped)} rows whose target had no mapped"
            f" sequence: {unmapped_targets}"
        )
    else:
        print("  all targets resolved to a sequence from the protein map")

    return Dataset(
        name=name,
        frame=merged,
        label_col=label_col,
        value_col=value_col,
        report=report,
    )


def prepare_all(cfg: dict, root: Path) -> dict[str, Dataset]:
    protein_map = load_protein_map(cfg, root)
    print(f"[protein map] {len(protein_map)} gene symbols with sequences\n")
    return {
        name: prepare(name, cfg, root, protein_map)
        for name in cfg["datasets"]
    }

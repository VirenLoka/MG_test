#!/usr/bin/env python3
"""
YAML configuration loading with inheritance and CLI overrides.

A config file may declare `defaults: base.yaml` to inherit from another file in
the same directory; the child is deep-merged over the parent. Leaves can then
be overridden from the command line with dotted keys, so no experiment needs a
code change:

    python -m src.train --config configs/dc50.yaml \
        --set model.gat.n_layers=6 train.optimizer.lr=1e-4 split.n_folds=3
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Repository root: MG_CrossAttn/src/config.py -> MG2Act/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Config(dict):
    """A dict that also supports attribute and dotted-path access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into non-mapping at {dotted!r}")
        node[parts[-1]] = value

    def resolve(self, dotted: str) -> Path:
        """A path config value, resolved against the repository root."""
        raw = self.get_path(dotted)
        if raw is None:
            raise ValueError(f"config path {dotted!r} is not set")
        p = Path(raw)
        return p if p.is_absolute() else REPO_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML rules (ints, floats, bools, lists)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(
    path: str | Path, overrides: list[str] | None = None, _seen: set | None = None
) -> Config:
    """
    Load a YAML config, following `defaults` inheritance and applying overrides.

    `overrides` entries are `dotted.key=value` strings.
    """
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path) if (REPO_ROOT / path).exists() else path.resolve()

    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular `defaults` inheritance at {path}")
    _seen.add(path)

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    parent_name = raw.pop("defaults", None)
    merged = raw
    if parent_name:
        parent_path = path.parent / parent_name
        parent = load_config(parent_path, overrides=None, _seen=_seen)
        merged = deep_merge(dict(parent), raw)

    cfg = Config(merged)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects dotted.key=value, got {item!r}")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce(value.strip()))

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the fully-resolved config beside the run's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, default_flow_style=False)

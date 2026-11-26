"""Configuration loading: ``core.yaml`` deep-merged with asset-class and pair overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict | None) -> dict:
    """Return a new dict: ``base`` with ``override`` merged in recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    """Read a YAML file into a dict (empty file -> empty dict)."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(asset_class: str | None = None, config_dir: str | Path = "config") -> dict:
    """Load ``core.yaml`` and deep-merge ``<asset_class>.yaml``'s ``overrides`` on top.

    The returned dict holds every core section (``calendar``, ``split``, ``ou``, ...).
    When ``asset_class`` is given, the full asset YAML is also stored under ``cfg["asset"]``
    and the name under ``cfg["asset_class"]``.
    """
    config_dir = Path(config_dir)
    cfg = _read_yaml(config_dir / "core.yaml")
    if asset_class is None:
        return cfg
    asset = _read_yaml(config_dir / f"{asset_class}.yaml")
    cfg = deep_merge(cfg, asset.get("overrides"))
    cfg["asset"] = asset
    cfg["asset_class"] = asset_class
    return cfg


def pair_config(cfg: dict, pair: dict) -> dict:
    """Effective config for one pair: merge the pair's ``overrides`` and ``in_sample_start_override``.

    The pair definition itself is stored under ``cfg["pair"]``.
    """
    out = deep_merge(cfg, pair.get("overrides"))
    start = pair.get("in_sample_start_override")
    if start is not None:
        out.setdefault("split", {})["in_sample_start"] = str(start)
    out["pair"] = copy.deepcopy(dict(pair))
    return out


def get(cfg: dict, dotted: str) -> Any:
    """Fetch ``cfg["a"]["b"]`` via ``get(cfg, "a.b")``; raises ``KeyError`` naming the missing key."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing config key '{dotted}'")
        node = node[part]
    return node

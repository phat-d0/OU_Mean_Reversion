"""Equities asset module: KO/PEP, XOM/CVX, HD/LOW, V/MA, XLF/KBE hedged log spreads.

``run()`` loads adjusted closes (cached), cleans and aligns each pair, builds the spread via
``core.backtest.build_pair_spread`` and evaluates it with ``core.pipeline.evaluate_pair``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ou_model.assets.equities.clean import align_legs, cleaning_notes
from ou_model.assets.equities.data import load_prices
from ou_model.assets.equities.pairs import build_pair
from ou_model.core.config import load_config, pair_config
from ou_model.core.pipeline import evaluate_pair

ASSET_CLASS = "equities"
log = logging.getLogger(__name__)


def run(config_dir: str | Path = "config", results_root: str | Path = "results") -> list[Path]:
    """Download/clean data, build every configured pair, call core.pipeline.evaluate_pair, return result dirs."""
    cfg = load_config(ASSET_CLASS, config_dir)
    asset = cfg["asset"]
    data_cfg = asset["data"]
    pairs = asset["pairs"]
    raw = load_prices([t for p in pairs for t in p["legs"]], data_cfg)
    out: list[Path] = []
    for pair in pairs:
        pcfg = pair_config(cfg, pair)
        prices, report = align_legs(raw, list(pair["legs"]), float(data_cfg["min_price"]))
        built = build_pair(prices, pair, pcfg, notes=cleaning_notes(report),
                           non_positive=report["non_positive_prices"])
        log.info("evaluating %s (%d obs)", built.name, len(built.legs))
        out.append(evaluate_pair(built.name, ASSET_CLASS, built.legs, built.spread, built.leg_weights,
                                 pcfg, meta=built.meta, hedge_history=built.hedge_history,
                                 results_root=results_root))
    return out

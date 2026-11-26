"""Commodities entry point: load, clean, flag, construct and evaluate every configured pair."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ou_model.core.config import get, load_config, pair_config
from ou_model.core.pipeline import evaluate_pair
from ou_model.core.results import validate_pair_results

from .cleaning import (align_legs, drop_incomplete_last_bar, find_non_positive, flags_from,
                       mask_non_positive, roll_gap_flags)
from .construction import build_spread, scale_legs
from .data import load_prices

log = logging.getLogger(__name__)
ASSET_CLASS = "commodities"


def prepare_pair(prices: pd.DataFrame, pair: dict, cfg: dict
                 ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame | None, dict]:
    """Legs (scaled, non-positive masked), spread, leg weights, hedge history and meta for one pair."""
    data_cfg, roll_cfg = cfg["asset"]["data"], cfg["asset"]["roll_gap"]
    raw = align_legs(prices, list(pair["legs"]), int(data_cfg.get("max_ffill_days", 0)))
    start = get(cfg, "split.in_sample_start")  # flags before it are warm-up, not written
    non_pos = flags_from(find_non_positive(raw), start)
    gaps = flags_from(roll_gap_flags(raw, roll_cfg), start)  # detection still uses warm-up std
    legs = mask_non_positive(raw)
    scale = pair.get("scale") or [1.0] * legs.shape[1]
    legs = scale_legs(legs, scale)
    spread, weights, hedge_hist = build_spread(legs, pair, cfg["hedge"],
                                               fit_end=get(cfg, "split.in_sample_end"))
    notes = [
        "yfinance continuous front-month futures are not back-adjusted; roll_gaps lists "
        f"|log return| / prior-{roll_cfg['window']}d std > {roll_cfg['zscore_threshold']} "
        "(contract rolls and genuine shocks alike); P&L across those dates includes the jump.",
    ]
    if any(s != 1.0 for s in scale):
        notes.append(f"legs scaled by {list(scale)} (RB/HO $/gal x42 -> $/bbl); "
                     "spread and P&L use scaled prices")
    if non_pos:
        notes.append(f"{len(non_pos)} non-positive price(s) masked to NaN before spread construction")
    meta = {"flags": {"roll_gaps": gaps, "non_positive_prices": non_pos, "notes": notes},
            "construction": pair["construction"], "transform": pair.get("transform", "level")}
    return legs, spread, weights, hedge_hist, meta


def run(config_dir: str | Path = "config", results_root: str | Path = "results",
        refresh: bool = False) -> list[Path]:
    """Download/clean data, build every configured pair, call ``evaluate_pair``; return result dirs."""
    cfg = load_config(ASSET_CLASS, config_dir)
    pairs = cfg["asset"]["pairs"]
    tickers = [t for p in pairs for t in p["legs"]]
    prices = load_prices(tickers, cfg["asset"]["data"], refresh=refresh)
    if cfg["asset"]["data"].get("drop_incomplete_last_bar", True):
        prices = drop_incomplete_last_bar(prices)
    out = []
    for pair in pairs:
        pcfg = pair_config(cfg, pair)
        legs, spread, weights, hist, meta = prepare_pair(prices, pair, pcfg)
        path = evaluate_pair(pair["name"], ASSET_CLASS, legs, spread, weights, pcfg, meta,
                             hedge_history=hist, results_root=results_root)
        errs = validate_pair_results(path)
        if errs:
            raise RuntimeError(f"{pair['name']}: invalid results: {errs}")
        log.info("wrote %s", path)
        out.append(path)
    return out


"""FX asset module: AUD/NZD, EUR/CHF (full and post-SNB) and USD/CAD vs crude."""

from __future__ import annotations

import logging
from pathlib import Path

from ou_model.core.config import load_config, pair_config
from ou_model.core.pipeline import evaluate_pair

from .data import Fetcher, load_prices
from .pairs import build_pair

ASSET_CLASS = "fx"
log = logging.getLogger(__name__)


def run(config_dir: str | Path = "config", results_root: str | Path = "results",
        fetch: Fetcher | None = None) -> list[Path]:
    """Download/clean data, build every configured FX pair, run ``core.pipeline.evaluate_pair``.

    Returns the result directories (one per pair, failures included). ``fetch`` overrides the
    yfinance downloader (tests).
    """
    cfg = load_config(ASSET_CLASS, config_dir)
    pairs = cfg["asset"]["pairs"]
    raw = load_prices([t for p in pairs for t in p["legs"]], cfg["asset"]["data"], fetch)
    out: list[Path] = []
    for pair in pairs:
        pcfg = pair_config(cfg, pair)
        inputs = build_pair(pair, raw, pcfg)
        log.info("evaluating %s (%d rows)", pair["name"], len(inputs.legs))
        out.append(evaluate_pair(pair["name"], ASSET_CLASS, inputs.legs, inputs.spread,
                                 inputs.leg_weights, pcfg, inputs.meta,
                                 hedge_history=inputs.hedge_history, results_root=results_root))
    return out


__all__ = ["run"]

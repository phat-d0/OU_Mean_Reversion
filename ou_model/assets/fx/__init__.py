"""FX asset module: AUD/NZD, EUR/CHF (full and post-SNB) and USD/CAD vs crude."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ou_model.core.config import load_config, pair_config
from ou_model.core.pipeline import evaluate_pair

from .clean import drop_incomplete_last_bar
from .data import Fetcher, load_prices
from .pairs import build_pair

ASSET_CLASS = "fx"
log = logging.getLogger(__name__)


def run(config_dir: str | Path = "config", results_root: str | Path = "results",
        fetch: Fetcher | None = None, today: str | pd.Timestamp | None = None) -> list[Path]:
    """Download/clean data, build every configured FX pair, run ``core.pipeline.evaluate_pair``.

    Returns the result directories (one per pair, failures included). ``fetch`` overrides the
    yfinance downloader (tests); ``today`` overrides the UTC date used by
    ``data.drop_incomplete_last_bar`` (rows dated on or after it are dropped).
    """
    cfg = load_config(ASSET_CLASS, config_dir)
    pairs = cfg["asset"]["pairs"]
    raw = load_prices([t for p in pairs for t in p["legs"]], cfg["asset"]["data"], fetch)
    dropped: dict[str, int] = {}
    if cfg["asset"]["data"].get("drop_incomplete_last_bar", True):
        for ticker, s in raw.items():
            raw[ticker], dropped[ticker] = drop_incomplete_last_bar(s, today)
    out: list[Path] = []
    for pair in pairs:
        pcfg = pair_config(cfg, pair)
        inputs = build_pair(pair, raw, pcfg)
        if dropped:
            inputs.meta["flags"]["notes"].insert(0, "incomplete bars (dated >= today UTC) dropped "
                                                 f"{ {t: dropped[t] for t in pair['legs']} }")
        log.info("evaluating %s (%d rows)", pair["name"], len(inputs.legs))
        out.append(evaluate_pair(pair["name"], ASSET_CLASS, inputs.legs, inputs.spread,
                                 inputs.leg_weights, pcfg, inputs.meta,
                                 hedge_history=inputs.hedge_history, results_root=results_root))
    return out


__all__ = ["run"]

"""Pair construction for equities: hedged 2-leg spreads via core.backtest.build_pair_spread."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ou_model.core.backtest import build_pair_spread
from ou_model.core.config import get

SURVIVORSHIP_NOTE = ("survivorship/selection bias: tickers were chosen with hindsight as today's "
                     "large caps and well-known peer pairs; results overstate what an ex-ante "
                     "selection would have achieved")
ADJ_CLOSE_NOTE = ("prices are yfinance 'Adj Close' (auto_adjust=False): split- and dividend-adjusted "
                  "backwards from the download date, so historical levels change between downloads")


@dataclass
class BuiltPair:
    """Everything core.pipeline.evaluate_pair needs for one pair."""

    name: str
    legs: pd.DataFrame
    spread: pd.Series
    leg_weights: pd.DataFrame
    hedge_history: pd.DataFrame
    meta: dict = field(default_factory=dict)


def build_pair(prices: pd.DataFrame, pair: dict, cfg: dict, notes: list[str] | None = None,
               non_positive: list[str] | None = None) -> BuiltPair:
    """Build the spread for one configured pair from cleaned, aligned leg prices.

    ``prices`` columns must include ``pair["legs"]`` (legs[0] = y). ``cfg`` is the effective pair
    config from ``pair_config``; its ``hedge`` section drives the hedge ratio. ``static_ols`` is
    fitted on in-sample data only (``split.in_sample_end``).
    """
    construction = pair["construction"]
    if construction != "hedged_spread":
        raise ValueError(f"equities supports construction 'hedged_spread', got '{construction}'")
    if len(pair["legs"]) != 2:
        raise ValueError(f"hedged_spread needs exactly 2 legs, got {pair['legs']}")
    y_name, x_name = pair["legs"]
    legs = prices[[y_name, x_name]].astype(float)
    spread, weights, hist = build_pair_spread(legs[y_name], legs[x_name], pair["transform"],
                                              get(cfg, "hedge"), fit_end=get(cfg, "split.in_sample_end"))
    meta = {
        "construction": construction,
        "transform": pair["transform"],
        "flags": {"non_positive_prices": list(non_positive or []),
                  "notes": [SURVIVORSHIP_NOTE, ADJ_CLOSE_NOTE] + list(notes or [])},
    }
    return BuiltPair(pair["name"], legs, spread, weights, hist, meta)

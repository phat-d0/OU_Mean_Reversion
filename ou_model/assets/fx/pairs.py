"""Pair construction for FX: hedged log spreads and single series, plus data flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ou_model.core.backtest import build_pair_spread

from .clean import align_legs, bad_ticks, mask_non_positive


@dataclass
class PairInputs:
    """Everything ``core.pipeline.evaluate_pair`` needs for one pair."""

    legs: pd.DataFrame
    spread: pd.Series
    leg_weights: pd.DataFrame
    hedge_history: pd.DataFrame | None
    meta: dict


def structural_breaks_for(pair_name: str, breaks_cfg: list[dict] | None) -> list[dict]:
    """Configured structural breaks that apply to ``pair_name`` (``date`` and ``label``)."""
    return [{"date": str(b["date"]), "label": str(b["label"])}
            for b in breaks_cfg or [] if pair_name in (b.get("pairs") or [])]


def single_series(legs: pd.DataFrame, transform: str) -> tuple[pd.Series, pd.DataFrame]:
    """One-leg spread ``T(P)`` with weights in shares per spread unit (``1/P`` for log, 1 for level)."""
    p = legs.iloc[:, 0]
    if transform == "log":
        spread, w = np.log(p), 1.0 / p
    elif transform == "level":
        spread, w = p, pd.Series(1.0, index=p.index).where(p.notna())
    else:
        raise ValueError(f"unknown transform '{transform}'")
    return spread.rename("spread"), w.to_frame(legs.columns[0])


def build_pair(pair: dict, raw: dict[str, pd.Series], cfg: dict) -> PairInputs:
    """Align, clean and construct one configured pair from raw ticker series.

    ``cfg`` is the effective pair config (``core.config.pair_config``); its ``asset`` key holds
    ``config/fx.yaml``.
    """
    asset, data_cfg = cfg["asset"], cfg["asset"]["data"]
    tickers = list(pair["legs"])
    legs, stats = align_legs({t: raw[t] for t in tickers}, int(data_cfg["max_ffill_days"]),
                             bool(data_cfg.get("drop_weekends", True)))
    ticks = bad_ticks(legs, float(data_cfg["bad_tick_max_abs_log_return"]))
    legs, non_pos = mask_non_positive(legs)
    notes = stats.notes()
    notes += [f"non-positive price masked to NaN: {f['leg']} {f['price']} on {f['date']}"
              for f in non_pos]
    notes += [f"large one-day move (|log r| > {data_cfg['bad_tick_max_abs_log_return']}): "
              f"{t['leg']} {t['log_return']:+.4f} on {t['date']}"
              + (" (reverses next day: possible bad tick)" if t["reverses_next_day"] else "")
              for t in ticks]
    construction, transform = pair["construction"], pair.get("transform", "log")
    hedge_history = None
    if construction == "hedged_spread":
        if len(tickers) != 2:
            raise ValueError(f"{pair['name']}: hedged_spread needs 2 legs")
        spread, weights, hedge_history = build_pair_spread(
            legs[tickers[0]], legs[tickers[1]], transform, cfg["hedge"],
            fit_end=cfg["split"]["in_sample_end"])
    elif construction == "single":
        if len(tickers) != 1:
            raise ValueError(f"{pair['name']}: single needs 1 leg")
        spread, weights = single_series(legs, transform)
    else:
        raise ValueError(f"{pair['name']}: unknown construction '{construction}'")
    meta = {"construction": construction, "transform": transform,
            "flags": {"roll_gaps": [],
                      "structural_breaks": structural_breaks_for(pair["name"],
                                                                 asset.get("structural_breaks")),
                      "non_positive_prices": non_pos, "notes": notes,
                      "bad_ticks": ticks,
                      "alignment": {"raw_rows": stats.raw_rows, "nan_dropped": stats.nan_dropped,
                                    "weekend_dropped": stats.weekend_dropped,
                                    "ffilled": stats.ffilled, "union_rows": stats.union_rows,
                                    "join_dropped": stats.join_dropped,
                                    "final_rows": stats.final_rows}}}
    return PairInputs(legs, spread, weights, hedge_history, meta)

"""Spread construction for the commodity pairs (fixed weights, ratio, rolling-OLS hedge)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ou_model.core.backtest import build_pair_spread


def scale_legs(prices: pd.DataFrame, scale: list[float]) -> pd.DataFrame:
    """Multiply each leg by its scale (e.g. RB/HO $/gal x 42 -> $/bbl)."""
    if len(scale) != prices.shape[1]:
        raise ValueError("scale must have one entry per leg")
    return prices * np.asarray(scale, dtype=float)


def fixed_weight_spread(legs: pd.DataFrame, weights: list[float]) -> tuple[pd.Series, pd.DataFrame]:
    """``sum_i w_i * leg_i`` on (already scaled) leg prices; constant shares ``w`` per spread unit.

    Dates where any leg is NaN (e.g. a masked non-positive price) get NaN spread and weights.
    """
    w = pd.Series(np.asarray(weights, dtype=float), index=legs.columns)
    spread = (legs * w).sum(axis=1, min_count=legs.shape[1]).where(legs.notna().all(axis=1))
    lw = pd.DataFrame(np.tile(w.to_numpy(), (len(legs), 1)), index=legs.index, columns=legs.columns)
    lw = lw.where(legs.notna().all(axis=1), axis=0)
    return spread.rename("spread"), lw


def ratio_spread(legs: pd.DataFrame, transform: str = "log") -> tuple[pd.Series, pd.DataFrame]:
    """``T(leg0 / leg1)``; for ``log`` the shares per unit are ``(1/P0, -1/P1)`` ($1 vs $1)."""
    a, b = legs.iloc[:, 0], legs.iloc[:, 1]
    ok = (a > 0) & (b > 0)
    if transform == "log":
        spread = np.log(a.where(ok) / b.where(ok))
        lw = pd.DataFrame({legs.columns[0]: 1.0 / a, legs.columns[1]: -1.0 / b})
    elif transform == "level":
        spread = a.where(ok) / b.where(ok)
        lw = pd.DataFrame({legs.columns[0]: 1.0 / b, legs.columns[1]: -a / b ** 2})
    else:
        raise ValueError(f"unknown transform '{transform}'")
    return spread.rename("spread"), lw.where(ok, axis=0)


def hedged_spread(legs: pd.DataFrame, transform: str, hedge_cfg: dict, fit_end=None
                  ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """2-leg ``T(y) - beta_t T(x) - alpha_t`` via ``core.backtest.build_pair_spread``."""
    y, x = legs.iloc[:, 0], legs.iloc[:, 1]
    return build_pair_spread(y, x, transform, hedge_cfg, fit_end=fit_end)


def build_spread(legs: pd.DataFrame, pair: dict, hedge_cfg: dict, fit_end=None
                 ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame | None]:
    """Dispatch on ``pair["construction"]``; returns ``(spread, leg_weights, hedge_history)``."""
    kind, transform = pair["construction"], pair.get("transform", "level")
    if kind == "fixed_weights":
        if transform != "level":
            raise ValueError("fixed_weights supports transform 'level' only")
        s, w = fixed_weight_spread(legs, pair["weights"])
        return s, w, None
    if kind == "ratio":
        s, w = ratio_spread(legs, transform)
        return s, w, None
    if kind == "hedged_spread":
        return hedged_spread(legs, transform, hedge_cfg, fit_end)
    raise ValueError(f"unknown construction '{kind}'")

"""Performance metrics on daily returns (on fixed notional) and round-trip trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .utils import json_safe

METRIC_KEYS = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "n_trades",
               "avg_holding_days", "half_life_days", "holding_to_half_life",
               "total_return", "total_gross_return", "total_cost", "n_obs"]


def max_drawdown(pnl: pd.Series) -> float | None:
    """Largest peak-to-trough fall of ``1 + cumsum(pnl)``, as a non-positive fraction of the peak."""
    if len(pnl) == 0:
        return None
    equity = 1.0 + pd.Series(pnl, dtype=float).fillna(0.0).cumsum()
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    return float(((equity.to_numpy() - peak) / peak).min())


def _ratio(num: float | None, den: float | None) -> float | None:
    """``num / den`` or ``None`` if undefined."""
    if num is None or den is None or not math.isfinite(den) or den == 0:
        return None
    return num / den


def compute_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    half_life_days: float | None,
    trading_days_per_year: int,
) -> dict:
    """Annualised return (mean daily ``pnl`` x N), vol, Sharpe, max drawdown, hit rate,
    trade count, average holding days, half-life days and holding/half-life ratio.

    Also ``total_return`` / ``total_gross_return`` / ``total_cost`` (sums over the period) and
    ``n_obs``. Undefined values (no trades, zero vol, infinite half-life) are ``None``.
    """
    pnl = pd.Series(daily["pnl"], dtype=float).fillna(0.0) if len(daily) else pd.Series(dtype=float)
    n_days = len(pnl)
    mean = float(pnl.mean()) if n_days else None
    std = float(pnl.std(ddof=1)) if n_days > 1 else None
    ann_ret = mean * trading_days_per_year if mean is not None else None
    ann_vol = std * math.sqrt(trading_days_per_year) if std is not None else None
    n_trades = int(len(trades))
    hold = float(trades["holding_days"].mean()) if n_trades else None
    hl = half_life_days if half_life_days is not None and math.isfinite(half_life_days) else None
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": _ratio(ann_ret, ann_vol) if ann_vol and ann_vol > 1e-15 else None,
        "max_drawdown": max_drawdown(pnl),
        "hit_rate": float((trades["pnl"] > 0).mean()) if n_trades else None,
        "n_trades": n_trades,
        "avg_holding_days": hold,
        "half_life_days": hl,
        "holding_to_half_life": _ratio(hold, hl),
        "total_return": float(pnl.sum()) if n_days else None,
        "total_gross_return": float(daily["gross_pnl"].sum()) if n_days else None,
        "total_cost": (float(daily["cost"].sum()) if n_days and "cost" in daily
                       else (float((daily["gross_pnl"] - daily["pnl"]).sum()) if n_days else None)),
        "n_obs": n_days,
    }
    return json_safe(out)

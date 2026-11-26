"""Cleaning, non-positive price masking and roll-gap (jump) detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_incomplete_last_bar(prices: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop rows dated today or later: yfinance returns the live, unsettled bar for today."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    return prices[prices.index < today]


def align_legs(prices: pd.DataFrame, legs: list[str], max_ffill_days: int = 0) -> pd.DataFrame:
    """Columns ``legs`` on dates where every leg has a quote (after an optional bounded ffill)."""
    frame = prices[legs].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if max_ffill_days > 0:
        frame = frame.ffill(limit=max_ffill_days)
    return frame.dropna(how="any")


def find_non_positive(prices: pd.DataFrame) -> list[dict]:
    """Every (date, leg, price) with price <= 0, sorted by date."""
    out = []
    for leg in prices.columns:
        bad = prices[leg][prices[leg] <= 0]
        out += [{"date": d.strftime("%Y-%m-%d"), "leg": str(leg), "price": float(p)}
                for d, p in bad.items()]
    return sorted(out, key=lambda r: (r["date"], r["leg"]))


def mask_non_positive(prices: pd.DataFrame) -> pd.DataFrame:
    """Replace non-positive prices with NaN (rows are kept so the date shows a NaN spread)."""
    return prices.where(prices > 0)


def detect_roll_gaps(prices: pd.Series, zscore_threshold: float, window: int) -> pd.DataFrame:
    """Flag jumps: ``|r_t| / std(r_{t-window} .. r_{t-1}) > zscore_threshold``, r = daily log return.

    The rolling std uses only *prior* returns so the jump does not inflate its own yardstick.
    Non-positive prices are masked first. Returns ``date``-indexed ``log_return``, ``zscore``.
    """
    s = pd.Series(prices, dtype=float)
    r = np.log(s.where(s > 0)).diff()
    sd = r.rolling(window, min_periods=window).std().shift(1)
    z = r.abs() / sd
    hit = z > zscore_threshold
    return pd.DataFrame({"log_return": r[hit], "zscore": z[hit]})


def roll_gap_flags(prices: pd.DataFrame, roll_cfg: dict) -> list[dict]:
    """Roll-gap flags for every leg as ``{"date", "leg", "log_return", "zscore"}`` dicts."""
    if not roll_cfg.get("detect", True):
        return []
    out = []
    for leg in prices.columns:
        gaps = detect_roll_gaps(prices[leg], float(roll_cfg["zscore_threshold"]),
                                int(roll_cfg["window"]))
        out += [{"date": d.strftime("%Y-%m-%d"), "leg": str(leg),
                 "log_return": round(float(row.log_return), 6), "zscore": round(float(row.zscore), 3)}
                for d, row in gaps.iterrows()]
    return sorted(out, key=lambda r: (r["date"], r["leg"]))


def flags_from(flags: list[dict], start: str | pd.Timestamp) -> list[dict]:
    """Keep flags dated on or after ``start`` (drops warm-up dates that are not in the results)."""
    start = pd.Timestamp(start)
    return [f for f in flags if pd.Timestamp(f["date"]) >= start]

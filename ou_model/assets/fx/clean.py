"""Cleaning and FX / futures calendar alignment. Every step reports what it removed."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AlignmentStats:
    """Row counts touched by each alignment step (per leg where relevant)."""

    raw_rows: dict[str, int] = field(default_factory=dict)
    nan_dropped: dict[str, int] = field(default_factory=dict)
    weekend_dropped: dict[str, int] = field(default_factory=dict)
    ffilled: dict[str, int] = field(default_factory=dict)
    union_rows: int = 0
    join_dropped: int = 0
    final_rows: int = 0

    def notes(self) -> list[str]:
        """Human-readable lines for ``data.flags.notes``."""
        legs = list(self.raw_rows)
        lines = [f"alignment: raw rows {self.raw_rows}",
                 f"alignment: NaN price rows dropped {self.nan_dropped}",
                 f"alignment: weekend rows dropped {self.weekend_dropped}"]
        if len(legs) > 1:
            lines += [f"alignment: rows forward-filled (holiday mismatch) {self.ffilled}",
                      f"alignment: union of dates {self.union_rows}, rows dropped by inner join "
                      f"{self.join_dropped}, final rows {self.final_rows}"]
        else:
            lines.append(f"alignment: single series, final rows {self.final_rows}")
        return lines


def utc_today() -> pd.Timestamp:
    """Today's date in UTC (tz-naive, midnight)."""
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def drop_incomplete_last_bar(s: pd.Series, today: pd.Timestamp | str | None = None
                             ) -> tuple[pd.Series, int]:
    """Drop rows dated on or after ``today`` (UTC by default): yfinance returns the live,
    unsettled bar for the current session. Returns the series and the number of rows removed."""
    cutoff = utc_today() if today is None else pd.Timestamp(today).normalize()
    keep = s.index < cutoff
    return s[keep], int((~keep).sum())


def drop_weekends(s: pd.Series) -> tuple[pd.Series, int]:
    """Remove Saturday/Sunday stamps; return the series and the number of rows removed."""
    keep = s.index.day_of_week < 5
    return s[keep], int((~keep).sum())


def align_legs(series: dict[str, pd.Series], max_ffill_days: int, drop_weekend_rows: bool = True
               ) -> tuple[pd.DataFrame, AlignmentStats]:
    """Align legs on common dates: drop NaN and weekend rows, forward-fill each leg at most
    ``max_ffill_days`` rows over the union of dates, then inner-join (drop any row still NaN).

    Columns keep the order of ``series``.
    """
    stats = AlignmentStats()
    cleaned: dict[str, pd.Series] = {}
    for name, s in series.items():
        s = pd.Series(s, dtype=float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        stats.raw_rows[name] = int(len(s))
        stats.nan_dropped[name] = int(s.isna().sum())
        s = s.dropna()
        n_wk = 0
        if drop_weekend_rows:
            s, n_wk = drop_weekends(s)
        stats.weekend_dropped[name] = n_wk
        cleaned[name] = s
    union = pd.DatetimeIndex(sorted(set().union(*(s.index for s in cleaned.values()))))
    stats.union_rows = int(len(union))
    frame = pd.DataFrame({n: s.reindex(union) for n, s in cleaned.items()})
    filled = frame.ffill(limit=max_ffill_days) if max_ffill_days > 0 else frame
    for name in frame:
        stats.ffilled[name] = int((frame[name].isna() & filled[name].notna()).sum())
    out = filled.dropna(how="any")
    stats.join_dropped = int(len(filled) - len(out))
    stats.final_rows = int(len(out))
    return out, stats


def mask_non_positive(legs: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Set prices <= 0 to NaN (they have no log); return the masked frame and flag records."""
    bad = legs <= 0
    flags = [{"date": d.strftime("%Y-%m-%d"), "leg": str(c), "price": float(legs.at[d, c])}
             for c in legs for d in legs.index[bad[c].to_numpy()]]
    return legs.where(~bad), flags


def bad_ticks(legs: pd.DataFrame, max_abs_log_return: float) -> list[dict]:
    """Days where a leg's one-day |log return| exceeds ``max_abs_log_return`` (flag, don't drop).

    Each record also says whether the next day reverses most of the move (a spike).
    """
    out: list[dict] = []
    for c in legs:
        p = legs[c].where(legs[c] > 0)
        r = np.log(p).diff()
        nxt = r.shift(-1)
        for d in r.index[(r.abs() > max_abs_log_return).to_numpy()]:
            spike = bool(np.isfinite(nxt[d]) and np.sign(nxt[d]) == -np.sign(r[d])
                         and abs(nxt[d]) > 0.5 * abs(r[d]))
            out.append({"date": d.strftime("%Y-%m-%d"), "leg": str(c),
                        "log_return": round(float(r[d]), 6), "reverses_next_day": spike})
    return out

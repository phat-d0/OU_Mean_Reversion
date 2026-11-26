"""FX alignment and cleaning on synthetic data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ou_model.assets.fx.clean import align_legs, bad_ticks, drop_weekends, mask_non_positive


def _s(dates: list[str], values: list[float], name: str = "x") -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=name, dtype=float)


def test_drop_weekends_counts_rows() -> None:
    # 2024-01-05 Fri, 06 Sat, 07 Sun, 08 Mon
    s = _s(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], [1, 2, 3, 4])
    out, n = drop_weekends(s)
    assert n == 2
    assert list(out.index.day_of_week) == [4, 0]


def test_align_ffill_limit_and_inner_join() -> None:
    days = pd.bdate_range("2024-01-01", periods=8)
    fx = pd.Series(np.arange(1.0, 9.0), index=days, name="FX")
    # futures leg misses day 2 (1-day holiday: filled) and days 5-6 (2-day gap: day 6 dropped)
    fut = fx.drop(days[[2, 5, 6]]).rename("FUT") * 10
    out, st = align_legs({"FX": fx, "FUT": fut}, max_ffill_days=1)
    assert list(out.columns) == ["FX", "FUT"]
    assert st.ffilled == {"FX": 0, "FUT": 2}          # day 2 and day 5
    assert st.join_dropped == 1                         # day 6 exceeds the 1-day limit
    assert days[6] not in out.index
    assert out.loc[days[2], "FUT"] == fut[days[1]]      # carried from the previous day
    assert out.loc[days[5], "FUT"] == fut[days[4]]
    assert st.final_rows == len(out) == 7
    assert not out.isna().any().any()


def test_align_zero_ffill_drops_all_mismatches() -> None:
    days = pd.bdate_range("2024-01-01", periods=5)
    a = pd.Series(1.0, index=days)
    b = pd.Series(2.0, index=days.delete(2))
    out, st = align_legs({"A": a, "B": b}, max_ffill_days=0)
    assert st.join_dropped == 1 and len(out) == 4


def test_align_drops_weekend_and_nan_rows_before_join() -> None:
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"])
    a = pd.Series([1.0, 1.1, np.nan, 1.3], index=idx)
    b = pd.Series([2.0, 2.0, 2.1, 2.2], index=idx)
    out, st = align_legs({"A": a, "B": b}, max_ffill_days=1)
    assert st.weekend_dropped == {"A": 1, "B": 1}
    assert st.nan_dropped == {"A": 1, "B": 0}
    assert st.ffilled["A"] == 1                          # the NaN Monday is filled from Friday
    assert (out.index.day_of_week < 5).all()
    assert len(out) == 3
    assert any("weekend" in n for n in st.notes())


def test_align_leading_period_trimmed_to_common_start() -> None:
    days = pd.bdate_range("2024-01-01", periods=6)
    a = pd.Series(1.0, index=days)
    b = pd.Series(2.0, index=days[3:])
    out, st = align_legs({"A": a, "B": b}, max_ffill_days=1)
    assert out.index[0] == days[3] and st.join_dropped == 3


def test_mask_non_positive_flags_and_masks() -> None:
    days = pd.bdate_range("2020-04-16", periods=4)  # includes Mon 2020-04-20
    legs = pd.DataFrame({"CAD=X": 1.4, "CL=F": [19.9, 18.3, -37.63, 10.0]}, index=days)
    out, flags = mask_non_positive(legs)
    assert flags == [{"date": "2020-04-20", "leg": "CL=F", "price": pytest.approx(-37.63)}]
    assert np.isnan(out.loc["2020-04-20", "CL=F"]) and out["CAD=X"].notna().all()


def test_bad_ticks_flags_spike_without_dropping() -> None:
    days = pd.bdate_range("2024-01-01", periods=5)
    legs = pd.DataFrame({"X": [1.0, 1.0, 1.5, 1.0, 1.0]}, index=days)
    ticks = bad_ticks(legs, 0.10)
    assert [t["date"] for t in ticks] == [days[2].strftime("%Y-%m-%d"), days[3].strftime("%Y-%m-%d")]
    assert ticks[0]["reverses_next_day"] is True
    assert bad_ticks(legs, 1.0) == []


def test_drop_incomplete_last_bar_with_injected_today() -> None:
    from ou_model.assets.fx.clean import drop_incomplete_last_bar
    s = pd.Series(1.0, index=pd.bdate_range("2026-11-23", periods=5))  # Mon 23 .. Fri 27
    out, n = drop_incomplete_last_bar(s, today="2026-11-26")
    assert n == 2 and out.index.max() == pd.Timestamp("2026-11-25")
    out, n = drop_incomplete_last_bar(s, today=pd.Timestamp("2026-12-01"))
    assert n == 0 and len(out) == 5

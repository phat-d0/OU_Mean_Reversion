"""Commodities cleaning: roll-gap detection, non-positive masking, alignment (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from ou_model.assets.commodities.cleaning import (align_legs, detect_roll_gaps,
                                                  drop_incomplete_last_bar, find_non_positive,
                                                  mask_non_positive, roll_gap_flags)


def _walk(n=300, seed=0, start=50.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_roll_gap_detects_injected_jump():
    s = _walk()
    s.iloc[200:] *= 1.08  # 8% splice jump ~ 8 sigma
    gaps = detect_roll_gaps(s, zscore_threshold=5.0, window=60)
    assert list(gaps.index) == [s.index[200]]
    assert gaps["log_return"].iloc[0] == pytest.approx(np.log(1.08), abs=0.04)


def test_roll_gap_quiet_series_and_warmup():
    s = _walk(seed=3)
    assert detect_roll_gaps(s, 5.0, 60).empty
    s.iloc[30:] *= 1.5  # inside the warm-up window: not enough history to judge
    assert detect_roll_gaps(s, 5.0, 60).empty


def test_roll_gap_flags_format_and_disable():
    df = pd.DataFrame({"A": _walk(seed=1), "B": _walk(seed=2)})
    df.iloc[150:, 1] *= 0.9
    flags = roll_gap_flags(df, {"detect": True, "zscore_threshold": 5.0, "window": 60})
    assert flags == [{"date": df.index[150].strftime("%Y-%m-%d"), "leg": "B",
                      "log_return": pytest.approx(flags[0]["log_return"]),
                      "zscore": pytest.approx(flags[0]["zscore"])}]
    assert roll_gap_flags(df, {"detect": False, "zscore_threshold": 5.0, "window": 60}) == []


def test_non_positive_masked_and_flagged():
    idx = pd.bdate_range("2020-04-15", periods=5)
    df = pd.DataFrame({"CL=F": [20.0, 19.0, 18.0, -37.63, 10.0],
                       "BZ=F": [28.0, 27.0, 26.0, 25.0, 0.0]}, index=idx)
    flags = find_non_positive(df)
    assert flags == [{"date": idx[3].strftime("%Y-%m-%d"), "leg": "CL=F", "price": -37.63},
                     {"date": idx[4].strftime("%Y-%m-%d"), "leg": "BZ=F", "price": 0.0}]
    masked = mask_non_positive(df)
    assert np.isnan(masked.loc[idx[3], "CL=F"]) and np.isnan(masked.loc[idx[4], "BZ=F"])
    assert len(masked) == 5 and masked.loc[idx[3], "BZ=F"] == 25.0


def test_roll_gap_ignores_non_positive_price():
    s = _walk(n=200)
    s.iloc[150] = -37.63
    # log return undefined around the masked point -> no crash, no flag on the NaN day itself
    gaps = detect_roll_gaps(s, 5.0, 60)
    assert s.index[150] not in gaps.index


def test_align_inner_join_no_ffill():
    idx = pd.bdate_range("2021-01-01", periods=4)
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0, 4.0], "B": [1.0, 2.0, 3.0, np.nan],
                       "C": [9.0] * 4}, index=idx)
    out = align_legs(df, ["A", "B"], max_ffill_days=0)
    assert list(out.index) == [idx[0], idx[2]] and list(out.columns) == ["A", "B"]
    assert len(align_legs(df, ["A", "B"], max_ffill_days=1)) == 4


def test_drop_incomplete_last_bar():
    idx = pd.date_range("2026-11-24", periods=3)
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    assert list(drop_incomplete_last_bar(df, today="2026-11-26").index) == list(idx[:2])

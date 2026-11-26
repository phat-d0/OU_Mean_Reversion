"""Commodities spread construction: x42 crack, Brent-WTI, ratio, hedged (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from ou_model.assets.commodities.construction import (build_spread, fixed_weight_spread,
                                                      ratio_spread, scale_legs)
from ou_model.assets.commodities.runner import prepare_pair
from ou_model.core.config import load_config, pair_config

HEDGE = {"method": "rolling_ols", "window": 60, "min_periods": 30, "add_constant": True}


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cl = 60 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "CL=F": cl,
        "RB=F": cl / 42 * 1.25 * np.exp(rng.normal(0, 0.01, n)),
        "HO=F": cl / 42 * 1.35 * np.exp(rng.normal(0, 0.01, n)),
        "BZ=F": cl + 4 + rng.normal(0, 0.5, n),
        "GC=F": 1500 * np.exp(np.cumsum(rng.normal(0, 0.008, n))),
        "SI=F": 18 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
    }, index=idx)


def test_crack_x42_conversion_and_spread():
    p = _prices()[["CL=F", "RB=F", "HO=F"]]
    legs = scale_legs(p, [1.0, 42.0, 42.0])
    assert np.allclose(legs["RB=F"], p["RB=F"] * 42) and np.allclose(legs["CL=F"], p["CL=F"])
    spread, w = fixed_weight_spread(legs, [-3.0, 2.0, 1.0])
    expected = 2 * p["RB=F"] * 42 + p["HO=F"] * 42 - 3 * p["CL=F"]
    assert np.allclose(spread, expected)
    # $/bbl crack margin around (2*1.25 + 1.35 - 3) * CL -> positive and same order as CL
    assert spread.mean() == pytest.approx(0.85 * p["CL=F"].mean(), rel=0.05)
    assert (w == pd.Series([-3.0, 2.0, 1.0], index=legs.columns)).all().all()


def test_scale_length_mismatch():
    with pytest.raises(ValueError):
        scale_legs(_prices()[["CL=F", "RB=F"]], [1.0])


def test_fixed_weights_nan_on_masked_leg():
    legs = _prices()[["BZ=F", "CL=F"]].copy()
    legs.iloc[100, 1] = np.nan
    spread, w = fixed_weight_spread(legs, [1.0, -1.0])
    assert np.isnan(spread.iloc[100]) and w.iloc[100].isna().all()
    assert spread.iloc[101] == pytest.approx(legs.iloc[101, 0] - legs.iloc[101, 1])


def test_ratio_log_shares():
    legs = _prices()[["GC=F", "SI=F"]]
    spread, w = ratio_spread(legs, "log")
    assert np.allclose(spread, np.log(legs["GC=F"] / legs["SI=F"]))
    assert np.allclose(w["GC=F"] * legs["GC=F"], 1.0)
    assert np.allclose(w["SI=F"] * legs["SI=F"], -1.0)


def test_hedged_dispatch_returns_history():
    legs = _prices()[["BZ=F", "CL=F"]]
    spread, w, hist = build_spread(legs, {"construction": "hedged_spread", "transform": "level"}, HEDGE)
    assert hist is not None and {"alpha", "beta"} <= set(hist.columns)
    ok = hist["beta"].notna()
    assert hist["beta"][ok].mean() == pytest.approx(1.0, abs=0.1)
    assert np.allclose(w["CL=F"][ok], -hist["beta"][ok])


def test_unknown_construction():
    with pytest.raises(ValueError):
        build_spread(_prices()[["BZ=F", "CL=F"]], {"construction": "nope"}, HEDGE)


def test_prepare_pair_masks_negative_wti_in_level_spread():
    prices = _prices()
    bad_day = prices.index[200]
    prices.loc[bad_day, "CL=F"] = -37.63
    cfg = load_config("commodities", "config")
    pair = next(p for p in cfg["asset"]["pairs"] if p["name"] == "crack_321")
    legs, spread, w, hist, meta = prepare_pair(prices, pair, pair_config(cfg, pair))
    assert np.isnan(legs.loc[bad_day, "CL=F"]) and np.isnan(spread.loc[bad_day])
    assert w.loc[bad_day].isna().all() and hist is None
    assert meta["flags"]["non_positive_prices"][0]["price"] == -37.63
    assert np.allclose(legs["HO=F"], prices["HO=F"] * 42)

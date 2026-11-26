"""Stationarity / cointegration wrappers and hedge ratios."""

import json

import numpy as np
import pandas as pd
import pytest

from ou_model.core.hedge import kalman_hedge_ratio, ols_hedge_ratio, rolling_hedge_ratio
from ou_model.core.ou import simulate_ou
from ou_model.core.stationarity import adf_test, engle_granger, johansen_test
from ou_model.core.synthetic import synthetic_cointegrated_pair

DT = 1.0 / 252


@pytest.fixture(scope="module")
def pair():
    return synthetic_cointegrated_pair(2000, beta=0.8, kappa=15.0, sigma_spread=0.1,
                                       sigma_x=0.25, dt=DT, start="2015-01-01", x0=50.0,
                                       alpha=0.3, seed=5)


def test_adf_ou_vs_random_walk():
    ou = adf_test(simulate_ou(20.0, 0, 1, 0, 2000, DT, seed=0), "c", "AIC")
    rw = adf_test(np.random.default_rng(0).standard_normal(2000).cumsum(), "c", "AIC")
    assert ou.pvalue < 0.01 and rw.pvalue > 0.05
    json.dumps(ou.to_dict(), allow_nan=False)


def test_adf_degenerate_does_not_raise():
    r = adf_test(np.ones(100), "c", "AIC")
    assert "error" in r.extra or np.isfinite(r.statistic)
    json.dumps(r.to_dict(), allow_nan=False)


def test_engle_granger(pair):
    ly, lx = np.log(pair["Y"]), np.log(pair["X"])
    r = engle_granger(ly, lx, "c")
    assert r.pvalue < 0.01
    assert r.extra["hedge_ratio"] == pytest.approx(0.8, abs=0.05)
    # independent random-walk pairs: false rejections at 5% should be rare (<= 3 of 20 seeds)
    rejections = 0
    for seed in range(20):
        steps = np.random.default_rng(seed).standard_normal((2000, 2)).cumsum(axis=0)
        rejections += engle_granger(pd.Series(steps[:, 0]), pd.Series(steps[:, 1]), "c").pvalue < 0.05
    assert rejections <= 3


def test_johansen_three_legs():
    rng = np.random.default_rng(2)
    n = 3000
    a, b = rng.standard_normal(n).cumsum(), rng.standard_normal(n).cumsum()
    c = 0.5 * a + 0.5 * b + simulate_ou(30.0, 0, 1, 0, n, DT, seed=3)
    r = johansen_test(pd.DataFrame({"C": c, "A": a, "B": b}), 0, 1)
    assert r.rank >= 1 and r.weights[0] == 1.0
    assert r.weights[1] == pytest.approx(-0.5, abs=0.05)
    assert r.weights[2] == pytest.approx(-0.5, abs=0.05)
    with pytest.raises(ValueError):
        johansen_test(pd.DataFrame({"A": a}), 0, 1)


def test_ols_and_rolling_hedge_causal(pair):
    ly, lx = np.log(pair["Y"]), np.log(pair["X"])
    assert ols_hedge_ratio(ly, lx).beta == pytest.approx(0.8, abs=0.05)
    roll = rolling_hedge_ratio(ly, lx, window=100, min_periods=50)
    assert roll["beta"].iloc[:49].isna().all() and roll["beta"].iloc[49:].notna().all()
    t = 500
    static = ols_hedge_ratio(ly.iloc[t - 99:t + 1], lx.iloc[t - 99:t + 1])
    assert roll["beta"].iloc[t] == pytest.approx(static.beta, rel=1e-9)
    assert roll["alpha"].iloc[t] == pytest.approx(static.alpha, rel=1e-9)
    ly2 = ly.copy()
    ly2.iloc[t + 1:] *= 3
    pd.testing.assert_frame_equal(roll.iloc[:t + 1],
                                  rolling_hedge_ratio(ly2, lx, 100, 50).iloc[:t + 1])


def test_rolling_no_constant(pair):
    ly, lx = np.log(pair["Y"]), np.log(pair["X"])
    roll = rolling_hedge_ratio(ly, lx, window=100, min_periods=50, add_constant=False)
    t = 300
    w = slice(t - 99, t + 1)
    expected = (ly.iloc[w] * lx.iloc[w]).sum() / (lx.iloc[w] ** 2).sum()
    assert roll["beta"].iloc[t] == pytest.approx(expected)
    assert (roll["alpha"].dropna() == 0).all()


def test_kalman_tracks_beta(pair):
    ly, lx = np.log(pair["Y"]), np.log(pair["X"])
    k = kalman_hedge_ratio(ly, lx, delta=1e-5, obs_var=1e-3, init_var=1.0)
    assert k["beta"].iloc[-500:].mean() == pytest.approx(0.8, abs=0.1)

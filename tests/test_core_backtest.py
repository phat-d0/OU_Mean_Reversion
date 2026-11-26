"""Quality gate 3-4: no look-ahead and costs; plus sizing and OU gating."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ou_model.core.backtest import (BacktestConfig, build_pair_spread, execute_target,
                                    run_backtest, size_position)
from ou_model.core.config import load_config
from ou_model.core.synthetic import synthetic_cointegrated_pair

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def cfg():
    return load_config(None, CONFIG_DIR)


@pytest.fixture(scope="module")
def bcfg(cfg):
    return BacktestConfig.from_config(cfg)


@pytest.fixture(scope="module")
def prices(cfg):
    return synthetic_cointegrated_pair(2000, beta=0.8, kappa=12.0, sigma_spread=0.12,
                                       sigma_x=0.25, dt=cfg["calendar"]["dt"], start="2015-01-01",
                                       x0=50.0, alpha=0.5, seed=7)


def _run(prices, cfg, bcfg, cost=None):
    spread, w, _ = build_pair_spread(prices["Y"], prices["X"], "log", cfg["hedge"])
    return run_backtest(prices, spread, w, bcfg, cost_bps_per_leg=cost)


@pytest.fixture(scope="module")
def base(prices, cfg, bcfg):
    return _run(prices, cfg, bcfg)


def test_backtest_trades_and_schema(base):
    d = base.daily
    assert len(base.trades) > 10
    assert set(d["position"].unique()) <= {-1, 0, 1}
    assert d["equity"].iloc[0] == 1.0
    assert np.allclose(d["equity"], 1 + d["pnl"].cumsum())
    assert not d[["pnl", "gross_pnl", "cost"]].isna().any().any()
    assert np.isclose(base.trades["pnl"].sum(), d["pnl"].sum())  # all P&L is inside trades


# ---- 3a: perturbing prices after t leaves everything up to t unchanged
@pytest.mark.parametrize("cut", [700, 1200, 1700])
def test_future_perturbation_does_not_change_past(prices, cfg, bcfg, base, cut):
    noisy = prices.copy()
    shock = np.exp(np.random.default_rng(cut).normal(0, 0.2, size=noisy.iloc[cut + 1:].shape))
    noisy.iloc[cut + 1:] = noisy.iloc[cut + 1:].to_numpy() * shock
    alt = _run(noisy, cfg, bcfg)
    cols = ["spread", "zscore", "target", "position", "event", "pnl", "gross_pnl", "cost", "equity"]
    pd.testing.assert_frame_equal(base.daily[cols].iloc[:cut + 1], alt.daily[cols].iloc[:cut + 1])
    t = prices.index[cut]
    done = base.trades[base.trades["exit_date"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(done, alt.trades[alt.trades["exit_date"] <= t].reset_index(drop=True))
    assert not base.daily["pnl"].iloc[cut + 1:].equals(alt.daily["pnl"].iloc[cut + 1:])


# ---- 3b: position is the target shifted by execution_lag
@pytest.mark.parametrize("lag", [0, 1, 2, 5])
def test_position_is_lagged_target(prices, cfg, bcfg, lag):
    res = _run(prices, cfg, dataclasses.replace(bcfg, execution_lag=lag))
    d = res.daily
    expected = d["target"].shift(lag).fillna(0).astype(int)
    assert (d["position"] == expected).all()
    entries = d.index[d["event"] == "entry"]
    assert list(res.trades["entry_date"]) == list(entries)


# ---- 3c: shift test with a signal that peeks one day ahead
def test_peeking_signal_only_profits_without_lag(bcfg):
    rng = np.random.default_rng(42)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    logp = np.log(50) + rng.normal(0, 0.015, size=(n, 2)).cumsum(axis=0)
    legs = pd.DataFrame(np.exp(logp), index=idx, columns=["A", "B"])
    spread = np.log(legs["A"]) - np.log(legs["B"])
    weights = pd.DataFrame({"A": 1 / legs["A"], "B": -1 / legs["B"]})
    peek = np.sign(spread.shift(-1) - spread).fillna(0).astype(int)  # uses tomorrow's close
    cfg0, cfg1 = (dataclasses.replace(bcfg, execution_lag=k, cost_bps_per_leg=0.0) for k in (0, 1))

    cheat, _ = execute_target(legs, weights, peek, cfg0)          # fills on the signal close
    honest, _ = execute_target(legs, weights, peek, cfg1)         # engine timing (lag 1)
    shifted, _ = execute_target(legs, weights, peek.shift(1).fillna(0).astype(int), cfg0)

    sharpe = lambda p: p.mean() / p.std() * np.sqrt(252)  # noqa: E731
    assert sharpe(cheat["pnl"]) > 10          # the peek really is informative
    assert abs(sharpe(honest["pnl"])) < 1.5   # ...but worthless once the lag is respected
    # Shift test: lag-1 engine == lag-0 engine fed the target shifted by one day.
    np.testing.assert_array_equal(honest["pnl"].to_numpy(), shifted["pnl"].to_numpy())
    np.testing.assert_array_equal(honest["position"].to_numpy(), shifted["position"].to_numpy())


# ---- 4: costs reduce returns
def test_costs_reduce_returns(prices, cfg, bcfg):
    runs = {c: _run(prices, cfg, bcfg, cost=c) for c in (0.0, 2.5, 5.0, 10.0)}
    totals = [runs[c].daily["pnl"].sum() for c in sorted(runs)]
    assert len(runs[0.0].trades) > 0
    assert all(a > b for a, b in zip(totals, totals[1:]))
    zero = runs[0.0].daily
    assert (zero["pnl"] == zero["gross_pnl"]).all() and (zero["cost"] == 0).all()
    for c, r in runs.items():  # gross is cost-independent
        np.testing.assert_array_equal(r.daily["gross_pnl"], zero["gross_pnl"])
        pd.testing.assert_series_equal(r.daily["position"], zero["position"])


def test_cost_amount_is_bps_of_traded_notional(prices, cfg, bcfg):
    r = _run(prices, cfg, bcfg, cost=10.0)
    first = r.trades.iloc[0]
    # dollar-neutral: entry trades notional/2 per leg (2 legs) = notional; exit ~ notional too
    entry_cost = r.daily.loc[first["entry_date"], "cost"]
    assert entry_cost == pytest.approx(10.0 / 1e4 * bcfg.notional)
    assert first["cost"] == pytest.approx(10.0 / 1e4 * bcfg.notional * 2, rel=0.2)


# ---- sizing and shares fixed at entry
def test_size_position_dollar_neutral_and_units():
    p = np.array([100.0, 20.0, 50.0])
    w = np.array([1.0, -2.0, -1.0])
    s = size_position(1, w, p, 1.0, "dollar_neutral")
    dollars = s * p
    assert dollars[0] == pytest.approx(0.5)
    assert dollars[1:].sum() == pytest.approx(-0.5)
    assert dollars[1] / dollars[2] == pytest.approx((2 * 20) / (1 * 50))
    single = size_position(-1, np.array([1.0]), np.array([10.0]), 1.0, "dollar_neutral")
    assert single * 10.0 == pytest.approx(-1.0)
    u = size_position(1, w, p, 1.0, "units")
    assert np.abs(u * p).sum() == pytest.approx(1.0)
    assert u[1] / u[0] == pytest.approx(-2.0)


def test_shares_fixed_during_trade(prices, cfg, bcfg, base):
    """With fixed shares, daily gross P&L during a trade equals shares . dP / notional where
    shares are those set at the entry fill (later weight/price changes are ignored)."""
    spread, w, _ = build_pair_spread(prices["Y"], prices["X"], "log", cfg["hedge"])
    tr = base.trades.iloc[2]
    i0 = prices.index.get_loc(tr["entry_date"])
    i1 = prices.index.get_loc(tr["exit_date"])
    shares = size_position(int(tr["direction"]), w.iloc[i0 - bcfg.execution_lag].to_numpy(),
                           prices.iloc[i0].to_numpy(), bcfg.notional, bcfg.sizing)
    expected = (prices.diff().iloc[i0 + 1:i1 + 1].to_numpy() @ shares) / bcfg.notional
    np.testing.assert_allclose(base.daily["gross_pnl"].iloc[i0 + 1:i1 + 1], expected, rtol=1e-12)


# ---- rolling OU re-estimation gates entries
def test_ou_gating_blocks_entries_when_not_mean_reverting(bcfg):
    rng = np.random.default_rng(3)
    n = 1500
    x = np.zeros(n)
    for i in range(1, n):  # b = -0.5: z-score still spikes, but OU fit is flagged
        x[i] = -0.5 * x[i - 1] + rng.standard_normal()
    idx = pd.bdate_range("2015-01-01", periods=n)
    spread = pd.Series(x, index=idx)
    legs = pd.DataFrame({"S": 100 + x}, index=idx)
    w = pd.DataFrame({"S": 1.0}, index=idx)
    gated = run_backtest(legs, spread, w, bcfg)
    free = run_backtest(legs, spread, w, dataclasses.replace(bcfg, require_mean_reverting=False))
    assert len(gated.trades) == 0 and (gated.daily["position"] == 0).all()
    assert len(free.trades) > 0
    assert not gated.ou_history["mean_reverting"].any()
    assert (np.diff(gated.ou_history.index) >= 0).all()


def test_trade_start_blocks_warmup_entries(prices, cfg, bcfg):
    spread, w, _ = build_pair_spread(prices["Y"], prices["X"], "log", cfg["hedge"])
    start = prices.index[1000]
    r = run_backtest(prices, spread, w, bcfg, trade_start=start)
    assert (r.daily.loc[:start, "position"].iloc[:-1] == 0).all()
    assert (r.trades["entry_date"] > start).all()


def test_time_stop_exits(prices, cfg, bcfg):
    r = _run(prices, cfg, dataclasses.replace(bcfg, max_holding_multiple=0.5))
    assert "time" in set(r.trades["exit_reason"])

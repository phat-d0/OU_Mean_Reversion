"""Metrics and config loading."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from ou_model.core.config import deep_merge, load_config, pair_config
from ou_model.core.metrics import METRIC_KEYS, compute_metrics, max_drawdown

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_metrics_known_values():
    daily = pd.DataFrame({"pnl": [0.01, -0.02, 0.03, 0.0], "gross_pnl": [0.01, -0.01, 0.03, 0.0],
                          "cost": [0.0, 0.01, 0.0, 0.0]})
    trades = pd.DataFrame({"pnl": [0.02, -0.01], "holding_days": [3, 5]})
    m = compute_metrics(daily, trades, half_life_days=8.0, trading_days_per_year=252)
    assert set(m) == set(METRIC_KEYS)
    assert m["ann_return"] == pytest.approx(0.005 * 252)
    assert m["ann_vol"] == pytest.approx(daily["pnl"].std() * math.sqrt(252))
    assert m["sharpe"] == pytest.approx(m["ann_return"] / m["ann_vol"])
    assert m["max_drawdown"] == pytest.approx((0.99 - 1.01) / 1.01)
    assert m["hit_rate"] == 0.5 and m["n_trades"] == 2 and m["avg_holding_days"] == 4.0
    assert m["holding_to_half_life"] == 0.5
    assert m["total_return"] == pytest.approx(0.02) and m["total_cost"] == pytest.approx(0.01)


def test_metrics_undefined_are_null():
    daily = pd.DataFrame({"pnl": [0.0] * 5, "gross_pnl": [0.0] * 5})
    m = compute_metrics(daily, pd.DataFrame(columns=["pnl", "holding_days"]), math.inf, 252)
    assert m["n_trades"] == 0
    for k in ("sharpe", "hit_rate", "avg_holding_days", "half_life_days", "holding_to_half_life"):
        assert m[k] is None
    json.dumps(m, allow_nan=False)
    empty = compute_metrics(daily.iloc[:0], pd.DataFrame(columns=["pnl", "holding_days"]), None, 252)
    json.dumps(empty, allow_nan=False)
    assert max_drawdown(pd.Series(dtype=float)) is None


def test_load_and_pair_config():
    core = load_config(None, CONFIG_DIR)
    assert core["backtest"]["execution_lag"] == 1 and "asset" not in core
    fx = load_config("fx", CONFIG_DIR)
    assert fx["asset_class"] == "fx" and fx["asset"]["pairs"]
    pair = next(p for p in fx["asset"]["pairs"] if p["name"] == "eurchf_post_snb")
    pc = pair_config(fx, dict(pair, overrides={"signals": {"entry_z": 1.5}}))
    assert pc["split"]["in_sample_start"] == "2015-02-01"
    assert pc["signals"]["entry_z"] == 1.5 and pc["signals"]["exit_z"] == 0.5
    assert fx["split"]["in_sample_start"] == "2015-01-01"  # original untouched
    assert pc["pair"]["name"] == "eurchf_post_snb"


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4}) == {"a": {"b": 1, "c": 3}, "d": 4}

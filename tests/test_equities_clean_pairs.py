"""Equities cleaning, pair construction and run() on synthetic data (no network)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ou_model.assets.equities as eq
from ou_model.assets.equities.clean import align_legs, cleaning_notes, drop_bad_prices
from ou_model.assets.equities.pairs import SURVIVORSHIP_NOTE, build_pair
from ou_model.core.config import load_config, pair_config
from ou_model.core.results import validate_pair_results

REPO = Path(__file__).resolve().parents[1]


def test_drop_bad_prices_counts():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([10.0, np.nan, -1.0, 0.0, 12.0], index=idx)
    out, counts = drop_bad_prices(s, min_price=0.01)
    assert list(out.values) == [10.0, 12.0]
    assert counts["missing"] == 1 and counts["non_positive"] == 2
    assert counts["non_positive_dates"] == ["2020-01-03", "2020-01-06"]


def test_align_legs_inner_join_no_fill():
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=4))
    b = pd.Series([5.0, -1.0, 7.0], index=pd.bdate_range("2020-01-02", periods=3))
    prices, report = align_legs({"A": a, "B": b}, ["A", "B"], min_price=0.01)
    assert list(prices.columns) == ["A", "B"]
    assert list(prices.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    assert not prices.isna().any().any()
    assert report["dates_dropped_by_join"] == 2
    assert report["non_positive_prices"] == ["B 2020-01-03"]
    assert any("inner join" in n for n in cleaning_notes(report))


def _synthetic_pair(n: int, seed: int = 0, start: str = "2014-01-02") -> pd.DataFrame:
    """Cointegrated pair: log y = 0.8 log x + OU noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    log_x = np.log(50.0) + np.cumsum(rng.normal(0, 0.01, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.01)
    log_y = 0.5 + 0.8 * log_x + ou
    return pd.DataFrame({"AAA": np.exp(log_y), "BBB": np.exp(log_x)}, index=idx)


def test_build_pair_log_weights_and_notes():
    cfg = load_config("equities", REPO / "config")
    pair = {"name": "aaa_bbb", "legs": ["AAA", "BBB"], "construction": "hedged_spread", "transform": "log"}
    prices = _synthetic_pair(600)
    built = build_pair(prices, pair, pair_config(cfg, pair), notes=["x"], non_positive=[])
    assert list(built.legs.columns) == ["AAA", "BBB"]
    last = built.leg_weights.dropna().index[-1]
    beta = built.hedge_history.loc[last, "beta"]
    assert built.leg_weights.loc[last, "AAA"] == pytest.approx(1.0 / prices.loc[last, "AAA"])
    assert built.leg_weights.loc[last, "BBB"] == pytest.approx(-beta / prices.loc[last, "BBB"])
    alpha = built.hedge_history.loc[last, "alpha"]
    expected = np.log(prices.loc[last, "AAA"]) - beta * np.log(prices.loc[last, "BBB"]) - alpha
    assert built.spread.loc[last] == pytest.approx(expected)
    assert SURVIVORSHIP_NOTE in built.meta["flags"]["notes"] and "x" in built.meta["flags"]["notes"]


def test_build_pair_rejects_bad_construction():
    cfg = load_config("equities", REPO / "config")
    pair = {"name": "p", "legs": ["AAA", "BBB"], "construction": "single_series", "transform": "log"}
    with pytest.raises(ValueError):
        build_pair(_synthetic_pair(300), pair, pair_config(cfg, pair))


def test_run_end_to_end_synthetic(tmp_path, monkeypatch):
    """run() with a one-pair config and a stubbed loader writes valid results."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(REPO / "config" / "core.yaml", cfg_dir / "core.yaml")
    text = (REPO / "config" / "equities.yaml").read_text().split("pairs:")[0]
    text += ("pairs:\n  - {name: aaa_bbb, legs: [\"AAA\", \"BBB\"], "
             "construction: hedged_spread, transform: log}\n")
    (cfg_dir / "equities.yaml").write_text(text)
    prices = _synthetic_pair(2600)  # 2014-01 .. ~2023-12: covers IS and some OOS
    monkeypatch.setattr(eq, "load_prices", lambda tickers, data_cfg: {t: prices[t] for t in tickers})

    paths = eq.run(config_dir=cfg_dir, results_root=tmp_path / "results")
    assert [p.name for p in paths] == ["aaa_bbb"]
    assert validate_pair_results(paths[0]) == []
    summary = json.loads((paths[0] / "summary.json").read_text())
    assert summary["legs"] == ["AAA", "BBB"]
    assert SURVIVORSHIP_NOTE in summary["data"]["flags"]["notes"]

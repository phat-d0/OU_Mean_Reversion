"""Unit tests for ou_model.ui.data / charts on a tiny synthetic results directory."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ou_model.ui import charts as C
from ou_model.ui import data as D

UI_CFG = D.load_ui_config(Path(__file__).resolve().parents[1] / "config" / "ui.yaml")
MISSING = UI_CFG.get("missing_value", "—")


def _metrics(sharpe, half_life=None, n_trades=3):
    return {"sharpe": sharpe, "ann_return": 0.01, "n_trades": n_trades, "half_life_days": half_life,
            "holding_to_half_life": None if half_life is None else 0.5, "total_return": 0.02, "total_cost": 0.001}


def _summary(asset_class, pair, oos_sharpe, *, flags=None, eg=True, hedge=0.9):
    return {
        "asset_class": asset_class, "pair": pair, "legs": ["A", "B"], "construction": "hedged_spread",
        "transform": "log",
        "data": {"start": "2020-01-01", "end": "2020-03-31", "n_obs": 60, "flags": flags or {}},
        "split": {"in_sample": ["2020-01-01", "2020-02-14"], "out_of_sample": ["2020-02-17", "2020-03-31"]},
        "hedge_ratio_in_sample": hedge,
        "ou_params": {"in_sample": {"ols": {"kappa": 5.0, "mu": 0.1, "sigma": 0.2, "half_life": 0.1386,
                                            "n_obs": 40, "mean_reverting": True},
                                    "mle": {"kappa": 5.1, "half_life": None}},
                      "out_of_sample_diagnostic": {"ols": {}}},
        "tests": {"in_sample": {"adf": {"statistic": -3.5, "pvalue": 0.01},
                                "engle_granger": {"statistic": -3.9, "pvalue": 0.02} if eg else None,
                                "johansen": None},
                  "out_of_sample_diagnostic": {"adf": {"statistic": -1.0, "pvalue": 0.7}}},
        "verdict": {"cointegrated_in_sample": eg, "mean_reverting_in_sample": True},
        "metrics": {"in_sample": _metrics(1.0, 10.0), "out_of_sample": _metrics(oos_sharpe, 10.0)},
        "cost_sensitivity": {"10.0": {"in_sample": _metrics(0.5), "out_of_sample": _metrics(0.4)},
                             "0.0": {"in_sample": _metrics(1.5), "out_of_sample": _metrics(None, n_trades=0)},
                             "5.0": {}},
        "config": {"signals": {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0}},
    }


def _spread(n=60):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    events = [""] * n
    events[10], events[15], events[30], events[35] = "entry", "exit", "entry", "stop"
    return pd.DataFrame({
        "date": dates, "spread": rng.standard_normal(n), "spread_mean": np.zeros(n), "spread_std": np.ones(n),
        "zscore": rng.standard_normal(n), "hedge_ratio": np.nan, "target": 0, "position": 0, "event": events,
        "pnl": 0.0, "gross_pnl": 0.0, "equity": 1.0, "sample": ["IS"] * 32 + ["OOS"] * (n - 32),
    })


@pytest.fixture()
def results_root(tmp_path: Path) -> Path:
    flags = {
        "roll_gaps": [{"date": "2020-01-10", "leg": "A", "log_return": -0.1, "zscore": 8.0}, "2020-02-03",
                      {"leg": "no-date"}, 42],
        "structural_breaks": [{"date": "2020-01-20", "label": "policy change"}],
        "notes": ["note one", "note two"],
    }
    specs = [("equities", "aa_bb", 0.5, flags, True), ("equities", "cc_dd", None, None, True),
             ("fx", "ee", 1.2, None, False)]
    for ac, pair, sr, fl, eg in specs:
        d = tmp_path / ac / pair
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(_summary(ac, pair, sr, flags=fl, eg=eg,
                                                            hedge=None if not eg else 0.9)))
        _spread().to_parquet(d / "spread.parquet")
        if pair != "cc_dd":  # cc_dd: zero trades, no trades file
            pd.DataFrame({"trade_id": [1], "entry_date": ["2020-01-15"], "exit_date": ["2020-01-22"],
                          "pnl": [0.01]}).to_parquet(d / "trades.parquet")
    (tmp_path / "commodities" / "empty_dir").mkdir(parents=True)  # no summary -> ignored
    return tmp_path


def test_discover_pairs_uses_directory_names(results_root):
    assert D.discover_pairs(results_root) == {"equities": ["aa_bb", "cc_dd"], "fx": ["ee"]}
    assert D.discover_pairs(results_root / "nope") == {}


def test_loaders_handle_missing_files(results_root, tmp_path):
    d = D.pair_dir(results_root, "equities", "aa_bb")
    assert D.load_summary(d)["pair"] == "aa_bb"
    spread = D.load_spread(d)
    assert pd.api.types.is_datetime64_any_dtype(spread["date"])
    assert D.load_trades(D.pair_dir(results_root, "equities", "cc_dd")).empty
    empty = tmp_path / "missing"
    assert D.load_summary(empty) == {} and D.load_spread(empty).empty and D.load_trades(empty).empty


def test_fmt_renders_nulls_as_placeholder():
    assert D.fmt(None, "{:.2f}", MISSING) == MISSING
    assert D.fmt(float("nan"), "{:.2f}", MISSING) == MISSING
    assert D.fmt(math.inf, "{:.2f}", MISSING) == MISSING
    assert D.fmt(1.234, "{:.2f}", MISSING) == "1.23"
    assert D.fmt(True, "{}", MISSING) == "yes"


def test_entry_bands_and_thresholds(results_root):
    s = D.load_summary(D.pair_dir(results_root, "equities", "aa_bb"))
    th = D.signal_thresholds(s)
    assert th == {"entry": 2.0, "exit": 0.5, "stop": 4.0}
    bands = D.entry_bands(_spread(), th["entry"])
    assert np.allclose(bands["upper"], 2.0) and np.allclose(bands["lower"], -2.0)
    assert D.entry_bands(pd.DataFrame(), 2.0).empty
    assert D.signal_thresholds({}) == {"entry": None, "exit": None, "stop": None}


def test_event_points_and_oos_start(results_root):
    pts = D.event_points(_spread())
    assert set(pts) == {"entry", "exit", "stop"} and len(pts["entry"]) == 2
    assert D.event_points(pd.DataFrame()) == {}
    s = D.load_summary(D.pair_dir(results_root, "equities", "aa_bb"))
    assert D.oos_start(s) == pd.Timestamp("2020-02-17")
    assert D.oos_start({}) is None


def test_flag_records_are_read_defensively(results_root):
    s = D.load_summary(D.pair_dir(results_root, "equities", "aa_bb"))
    gaps = D.flag_records(s, "roll_gaps")
    assert len(gaps) == 2 and list(gaps["date"]) == [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-02-03")]
    assert D.flag_records(s, "structural_breaks").loc[0, "label"] == "policy change"
    assert D.flag_records(s, "bad_ticks").empty
    assert D.flag_notes(s) == ["note one", "note two"]
    assert D.flag_counts(s)["roll_gaps"] == 2


def test_tables_never_crash_on_nulls(results_root):
    s = D.load_summary(D.pair_dir(results_root, "fx", "ee"))
    ou = D.ou_params_table(s, UI_CFG)
    assert ou.loc["half_life (years)", "IS · MLE"] == MISSING
    assert ou.loc["kappa", "OOS · OLS (diagnostic)"] == MISSING
    tests = D.tests_table(s, UI_CFG)
    eg = tests[(tests["test"] == "engle_granger")]
    assert eg["p-value / rank"].iloc[0] == MISSING
    assert D.verdict_table(s, UI_CFG).loc["hedge ratio (IS)", "value"] == MISSING
    sens = D.cost_sensitivity_table(s, UI_CFG)
    assert list(sens["cost (bps/leg)"]) == ["0.0", "5.0", "10.0"]
    assert sens.loc[0, "OOS sharpe"] == MISSING and sens.loc[1, "IS sharpe"] == MISSING
    assert D.metrics_table({}, UI_CFG).eq(MISSING).all().all()
    for fn in (D.ou_params_table, D.tests_table, D.verdict_table, D.cost_sensitivity_table):
        fn({}, UI_CFG)


def test_cross_asset_ranked_by_oos_sharpe(results_root):
    df = D.cross_asset_frame(results_root)
    assert list(df["pair"]) == ["ee", "aa_bb", "cc_dd"]  # null Sharpe last
    assert not df.loc[0, "cointegrated_is"]
    out = D.format_cross_asset(df, UI_CFG)
    assert out.loc[2, "OOS Sharpe"] == MISSING and list(out["rank"]) == [1, 2, 3]


def test_implied_half_life():
    assert D.implied_half_life_days(math.log(2), 252) == pytest.approx(252)
    assert D.implied_half_life_days(0.0, 252) is None


def test_figures_build_on_synthetic_and_empty(results_root):
    d = D.pair_dir(results_root, "equities", "aa_bb")
    s, sp = D.load_summary(d), D.load_spread(d)
    for fig in (C.spread_figure(sp, s, UI_CFG), C.zscore_figure(sp, s, UI_CFG), C.equity_figure(sp, s, UI_CFG),
                C.spread_figure(pd.DataFrame(), {}, UI_CFG), C.equity_figure(pd.DataFrame(), {}, UI_CFG)):
        assert fig.to_dict()
    names = {t.name for t in C.spread_figure(sp, s, UI_CFG).data}
    assert "roll gap" in names
    names_off = {t.name for t in C.spread_figure(sp, s, UI_CFG, show_roll_gaps=False).data}
    assert "roll gap" not in names_off
    assert C.cross_asset_figure(D.cross_asset_frame(results_root), UI_CFG).to_dict()


def test_ui_imports_only_simulate_ou_from_core():
    ui_dir = Path(D.__file__).parent
    for path in ui_dir.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ou_model.core"):
                assert node.module == "ou_model.core.ou" and [a.name for a in node.names] == ["simulate_ou"], path
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("ou_model.core") for a in node.names), path
                assert not any(a.name.split(".")[0] in {"yfinance", "requests", "urllib"} for a in node.names)

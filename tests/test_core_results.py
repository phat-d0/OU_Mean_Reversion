"""Quality gate 5: evaluate_pair -> write_pair_results -> validate_pair_results round trip."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ou_model.core.backtest import build_pair_spread
from ou_model.core.config import load_config, pair_config
from ou_model.core.metrics import METRIC_KEYS
from ou_model.core.ou import simulate_ou
from ou_model.core.pipeline import evaluate_pair
from ou_model.core.results import validate_pair_results, write_pair_results
from ou_model.core.synthetic import synthetic_cointegrated_pair

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture(scope="module")
def cfg():
    return load_config(None, CONFIG_DIR)


@pytest.fixture(scope="module")
def result_dir(cfg, tmp_path_factory):
    pc = pair_config(cfg, {"name": "syn", "legs": ["Y", "X"], "construction": "hedged_spread",
                           "transform": "log"})
    px = synthetic_cointegrated_pair(2900, beta=0.8, kappa=12.0, sigma_spread=0.12, sigma_x=0.25,
                                     dt=cfg["calendar"]["dt"], start="2014-01-01", x0=50.0,
                                     alpha=0.5, seed=7)
    spread, w, hist = build_pair_spread(px["Y"], px["X"], "log", pc["hedge"])
    root = tmp_path_factory.mktemp("results")
    return evaluate_pair("syn", "synthetic", px, spread, w, pc,
                         {"flags": {"roll_gaps": ["2016-01-04"]}}, hedge_history=hist,
                         results_root=root)


def test_round_trip_valid(result_dir):
    assert result_dir.name == "syn" and result_dir.parent.name == "synthetic"
    assert validate_pair_results(result_dir) == []


def test_summary_content(result_dir):
    s = json.loads((result_dir / "summary.json").read_text())
    assert s["verdict"] == {"cointegrated_in_sample": True, "mean_reverting_in_sample": True}
    assert s["hedge_ratio_in_sample"] == pytest.approx(0.8, abs=0.05)
    assert s["split"]["in_sample"] == ["2015-01-01", "2021-12-31"]
    assert s["data"]["start"] >= "2015-01-01"
    assert s["data"]["flags"]["roll_gaps"] == ["2016-01-04"]
    assert set(s["ou_params"]["in_sample"]) == {"ols", "mle"}
    assert s["ou_params"]["in_sample"]["ols"]["kappa"] == pytest.approx(12.0, rel=0.3)
    assert s["tests"]["in_sample"]["engle_granger"]["pvalue"] < 0.05
    assert s["tests"]["in_sample"]["johansen"]["rank"] >= 1
    assert set(s["cost_sensitivity"]) == {"0.0", "5.0", "10.0"}
    for block in ("in_sample", "out_of_sample"):
        assert set(METRIC_KEYS) <= set(s["metrics"][block])
        assert s["metrics"][block]["n_trades"] > 0
        tot = [s["cost_sensitivity"][c][block]["total_return"] for c in ("0.0", "5.0", "10.0")]
        assert tot[0] > tot[1] > tot[2]
        assert s["metrics"][block] == s["cost_sensitivity"]["5.0"][block]
    assert "asset" not in s["config"]


def test_parquet_schema(result_dir):
    sp = pd.read_parquet(result_dir / "spread.parquet")
    tr = pd.read_parquet(result_dir / "trades.parquet")
    assert list(sp.columns) == ["date", "spread", "spread_mean", "spread_std", "zscore",
                                "hedge_ratio", "target", "position", "event", "pnl",
                                "gross_pnl", "equity", "sample"]
    assert sp["equity"].iloc[0] == 1.0
    assert set(sp["sample"]) == {"IS", "OOS"}
    assert sp["hedge_ratio"].notna().all()
    assert (sp.loc[sp["sample"] == "IS", "date"] <= "2021-12-31").all()
    assert set(tr["sample"]) == {"IS", "OOS"}
    assert set(tr["exit_reason"]) <= {"exit", "stop", "time", "end_of_data"}


def _copy(src: Path, dst: Path) -> Path:
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_validator_catches_problems(result_dir, tmp_path):
    bad = _copy(result_dir, tmp_path / "bad1")
    sp = pd.read_parquet(bad / "spread.parquet")
    sp.loc[100, "equity"] += 1.0
    sp.loc[200, "position"] = 1 - sp.loc[200, "target"]  # breaks lag consistency
    sp.drop(columns=["zscore"]).to_parquet(bad / "spread.parquet")
    assert any("zscore" in e for e in validate_pair_results(bad))
    sp.to_parquet(bad / "spread.parquet")
    errs = validate_pair_results(bad)
    assert any("equity" in e for e in errs) and any("execution_lag" in e for e in errs)

    bad2 = _copy(result_dir, tmp_path / "bad2")
    (bad2 / "summary.json").write_text('{"pair": NaN}')
    assert validate_pair_results(bad2)
    (bad2 / "trades.parquet").unlink()
    assert validate_pair_results(bad2) == ["missing file trades.parquet"]


def test_single_leg_and_three_leg_pipelines(cfg, tmp_path):
    n = 2000
    idx = pd.bdate_range("2015-01-01", periods=n)
    dt = cfg["calendar"]["dt"]
    one = pd.DataFrame({"S": np.exp(0.1 * simulate_ou(10.0, 0.0, 1.0, 0.0, n, dt, seed=1))}, index=idx)
    pc = pair_config(cfg, {"name": "single", "legs": ["S"], "construction": "single", "transform": "log"})
    out = evaluate_pair("single", "synthetic", one, np.log(one["S"]), pd.DataFrame({"S": 1.0}, index=idx),
                        pc, {}, results_root=tmp_path)
    assert validate_pair_results(out) == []
    s = json.loads((out / "summary.json").read_text())
    assert s["tests"]["in_sample"]["engle_granger"] is None and s["tests"]["in_sample"]["johansen"] is None
    assert s["hedge_ratio_in_sample"] is None and s["verdict"]["cointegrated_in_sample"] is True

    rng = np.random.default_rng(4)
    a, b = 100 + rng.standard_normal(n).cumsum(), 100 + rng.standard_normal(n).cumsum()
    c = 0.5 * a + 0.5 * b + simulate_ou(30.0, 0, 3.0, 0, n, dt, seed=5)
    legs = pd.DataFrame({"C": c, "A": a, "B": b}, index=idx)
    weights = pd.DataFrame({"C": 1.0, "A": -0.5, "B": -0.5}, index=idx)
    spread = (legs * weights).sum(axis=1)
    pc3 = pair_config(cfg, {"name": "three", "legs": list(legs), "construction": "fixed_weights",
                            "transform": "level"})
    out3 = evaluate_pair("three", "synthetic", legs, spread, weights, pc3, {}, results_root=tmp_path)
    assert validate_pair_results(out3) == []
    s3 = json.loads((out3 / "summary.json").read_text())
    assert s3["tests"]["in_sample"]["johansen"]["rank"] >= 1
    assert s3["verdict"]["cointegrated_in_sample"] is True


def test_write_empty_trades(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=3)
    sp = pd.DataFrame({"spread": [1.0, 2, 3], "spread_mean": np.nan, "spread_std": np.nan,
                       "zscore": np.nan, "hedge_ratio": np.nan, "target": 0, "position": 0,
                       "event": "", "pnl": 0.0, "gross_pnl": 0.0, "equity": 1.0,
                       "sample": "IS"}, index=idx)
    out = write_pair_results(tmp_path / "x", sp, pd.DataFrame(), {"a": float("nan")})
    assert json.loads((out / "summary.json").read_text()) == {"a": None}
    assert len(pd.read_parquet(out / "trades.parquet")) == 0


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_level_shift_fails_mean_reverting_verdict(cfg, tmp_path, seed):
    """A single large level shift with slow partial recovery (plus a small random walk and
    iid noise): the lag-0 Dickey-Fuller in fit_ou rejects a unit root, the AIC-lag ADF does
    not, so the verdict must be False while ou_params keep the fitted values."""
    n = 2000
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shift = np.where(t >= 300, -0.15 + 0.075 * (1 - np.exp(-(t - 300) / 400.0)), 0.0)
    s = shift + rng.normal(0, 0.002, n).cumsum() + rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    legs = pd.DataFrame({"S": np.exp(s)}, index=idx)
    pc = pair_config(cfg, {"name": "shift", "legs": ["S"], "construction": "single",
                           "transform": "log"})
    out = evaluate_pair("shift", "synthetic", legs, np.log(legs["S"]),
                        pd.DataFrame({"S": 1.0}, index=idx), pc, {}, results_root=tmp_path)
    assert validate_pair_results(out) == []
    summary = json.loads((out / "summary.json").read_text())
    sig = cfg["stationarity"]["significance"]
    ols = summary["ou_params"]["in_sample"]["ols"]
    assert ols["mean_reverting"] is True and ols["unit_root_pvalue"] < sig   # lag-0 DF rejects
    assert summary["tests"]["in_sample"]["adf"]["pvalue"] >= sig             # ADF(AIC) does not
    assert summary["verdict"]["mean_reverting_in_sample"] is False
    for block in ("in_sample", "out_of_sample"):
        assert summary["metrics"][block]["half_life_days"] is None
        assert summary["metrics"][block]["holding_to_half_life"] is None

"""FX pair construction, cache/retry and end-to-end run on synthetic data (no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ou_model.assets import fx
from ou_model.assets.fx.data import cache_path, fetch_with_retry, load_prices
from ou_model.assets.fx.pairs import single_series, structural_breaks_for
from ou_model.core.results import validate_pair_results

REPO = Path(__file__).resolve().parents[1]


def _synthetic_fetch(seed: int = 0):
    """Fetcher producing daily (incl. weekend stamps) FX and weekday futures with holidays."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2014-01-01", "2023-06-30", freq="D")
    n = len(days)
    common = np.cumsum(rng.normal(0, 0.005, n))
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.95 * ou[i - 1] + rng.normal(0, 0.003)
    series = {
        "AUDUSD=X": 0.75 * np.exp(common + ou),
        "NZDUSD=X": 0.70 * np.exp(common),
        "EURCHF=X": 1.10 * np.exp(ou),
        "CAD=X": 1.30 * np.exp(-0.1 * common + ou),
        "CL=F": 60.0 * np.exp(common),
    }
    holidays = days[(days.day_of_week < 5)][::50]

    def fetch(ticker: str, start: str, end: str | None) -> pd.Series:
        s = pd.Series(series[ticker], index=days, name=ticker)
        if ticker == "CL=F":
            s = s[(s.index.day_of_week < 5) & ~s.index.isin(holidays)]
            s.loc["2020-04-20"] = -37.63
            s = s.sort_index()
        return s

    return fetch


@pytest.fixture()
def cfg_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    shutil.copytree(REPO / "config", d)
    fx_cfg = yaml.safe_load((d / "fx.yaml").read_text())
    fx_cfg["data"]["cache_dir"] = str(tmp_path / "cache")
    (d / "fx.yaml").write_text(yaml.safe_dump(fx_cfg))
    return d


def test_fetch_with_retry_backs_off() -> None:
    calls, waits = [], []

    def flaky() -> pd.Series:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("HTTP 429")
        return pd.Series([1.0])

    out = fetch_with_retry(flaky, 5, 2.0, sleep=waits.append)
    assert len(calls) == 3 and waits == [2.0, 4.0] and out.iloc[0] == 1.0
    with pytest.raises(RuntimeError):
        fetch_with_retry(lambda: (_ for _ in ()).throw(RuntimeError("429")), 2, 0.0,
                         sleep=lambda s: None)


def test_load_prices_uses_cache(tmp_path: Path) -> None:
    data_cfg = {"cache_dir": str(tmp_path), "download_start": "2014-01-01", "download_end": None,
                "max_retries": 1, "backoff_seconds": 0.0}
    calls: list[str] = []
    base = _synthetic_fetch()

    def fetch(t: str, s: str, e: str | None) -> pd.Series:
        calls.append(t)
        return base(t, s, e)

    first = load_prices(["EURCHF=X", "EURCHF=X"], data_cfg, fetch)
    assert calls == ["EURCHF=X"] and cache_path(tmp_path, "EURCHF=X").is_file()
    second = load_prices(["EURCHF=X"], data_cfg, fetch)
    assert calls == ["EURCHF=X"]
    pd.testing.assert_series_equal(first["EURCHF=X"], second["EURCHF=X"], check_freq=False)


def test_single_series_weights_are_inverse_price() -> None:
    legs = pd.DataFrame({"EURCHF=X": [1.0, 1.25, 2.0]},
                        index=pd.bdate_range("2024-01-01", periods=3))
    spread, w = single_series(legs, "log")
    np.testing.assert_allclose(spread, np.log(legs["EURCHF=X"]))
    np.testing.assert_allclose(w["EURCHF=X"], [1.0, 0.8, 0.5])


def test_structural_breaks_for_pair() -> None:
    cfg = [{"date": "2015-01-15", "label": "SNB", "pairs": ["eurchf", "eurchf_post_snb"]}]
    assert structural_breaks_for("eurchf", cfg) == [{"date": "2015-01-15", "label": "SNB"}]
    assert structural_breaks_for("aud_nzd", cfg) == []


def test_run_end_to_end_synthetic(cfg_dir: Path, tmp_path: Path) -> None:
    paths = fx.run(cfg_dir, tmp_path / "results", fetch=_synthetic_fetch())
    names = sorted(p.name for p in paths)
    assert names == ["aud_nzd", "eurchf", "eurchf_post_snb", "usdcad_crude"]
    import json
    for p in paths:
        assert validate_pair_results(p) == [], p
        s = json.loads((p / "summary.json").read_text())
        notes = s["data"]["flags"]["notes"]
        assert any(n.startswith("alignment:") for n in notes)
        sp = pd.read_parquet(p / "spread.parquet")
        assert (pd.DatetimeIndex(sp["date"]).day_of_week < 5).all()
    by = {p.name: json.loads((p / "summary.json").read_text()) for p in paths}
    assert by["eurchf_post_snb"]["split"]["in_sample"][0] == "2015-02-01"
    assert by["eurchf_post_snb"]["data"]["start"] >= "2015-02-01"
    for n in ("eurchf", "eurchf_post_snb"):
        assert by[n]["data"]["flags"]["structural_breaks"][0]["date"] == "2015-01-15"
        assert by[n]["tests"]["in_sample"]["engle_granger"] is None
    assert by["aud_nzd"]["data"]["flags"]["structural_breaks"] == []
    npp = by["usdcad_crude"]["data"]["flags"]["non_positive_prices"]
    assert npp and npp[0]["date"] == "2020-04-20" and npp[0]["leg"] == "CL=F"
    assert by["usdcad_crude"]["data"]["flags"]["alignment"]["ffilled"]["CL=F"] > 0
    assert by["aud_nzd"]["verdict"]["cointegrated_in_sample"] is True

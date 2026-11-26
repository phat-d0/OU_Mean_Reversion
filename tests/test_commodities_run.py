"""End-to-end ``commodities.run`` on synthetic prices (no network): every pair writes valid results."""

import json

import numpy as np
import pandas as pd

import ou_model.assets.commodities.runner as cm_run
from ou_model.core.results import validate_pair_results


def _synthetic(tickers, data_cfg, refresh=False):
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2014-01-01", "2023-06-30")
    n = len(idx)
    cl = 60 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    gc = 1500 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    df = pd.DataFrame({
        "BZ=F": cl + 4 + rng.normal(0, 0.5, n),
        "CL=F": cl,
        "RB=F": cl / 42 * 1.25 * np.exp(rng.normal(0, 0.01, n)),
        "HO=F": cl / 42 * 1.35 * np.exp(rng.normal(0, 0.01, n)),
        "GC=F": gc,
        "SI=F": gc / 80 * np.exp(rng.normal(0, 0.02, n)),
    }, index=idx)
    df.loc["2020-04-20", "CL=F"] = -37.63
    df.loc["2018-03-01":, "RB=F"] *= 1.15  # injected roll jump
    df.loc["2014-09-02":, "RB=F"] *= 1.15  # warm-up jump: detected but must not be reported
    df.loc["2014-11-03", "CL=F"] = -1.0  # warm-up non-positive price: not reported
    return df[list(dict.fromkeys(tickers))]


def test_run_all_pairs_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(cm_run, "load_prices", _synthetic)
    paths = cm_run.run(config_dir="config", results_root=tmp_path)
    assert [p.name for p in paths] == ["brent_wti", "brent_wti_hedged", "crack_321",
                                       "gold_silver_ratio", "gold_silver_hedged"]
    for p in paths:
        assert validate_pair_results(p) == []
    crack = json.loads((tmp_path / "commodities" / "crack_321" / "summary.json").read_text())
    assert crack["tests"]["in_sample"]["johansen"]["rank"] is not None
    assert crack["data"]["flags"]["non_positive_prices"][0]["date"] == "2020-04-20"
    assert any(g["date"] == "2018-03-01" and g["leg"] == "RB=F"
               for g in crack["data"]["flags"]["roll_gaps"])
    start = pd.Timestamp(crack["data"]["start"])
    for c in ("crack_321", "brent_wti", "gold_silver_hedged"):
        flags = json.loads((tmp_path / "commodities" / c / "summary.json").read_text())["data"]["flags"]
        for key in ("roll_gaps", "non_positive_prices"):
            assert all(pd.Timestamp(f["date"]) >= start for f in flags[key])
    sp = pd.read_parquet(tmp_path / "commodities" / "brent_wti" / "spread.parquet")
    assert np.isnan(sp.set_index("date").loc["2020-04-20", "spread"])

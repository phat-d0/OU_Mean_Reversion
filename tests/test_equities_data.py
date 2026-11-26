"""Equities loader: cache and retry behaviour (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from ou_model.assets.equities.data import cache_path, download_with_retry, load_ticker


def _series(start: str = "2014-01-02", n: int = 10) -> pd.Series:
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.Series(range(1, n + 1), index=idx, dtype=float, name="KO")


def _data_cfg(tmp_path) -> dict:
    return {"cache_dir": str(tmp_path), "download_start": "2014-01-01", "download_end": None,
            "price_field": "Adj Close", "max_retries": 3, "retry_backoff_s": 0.0,
            "cache_start_slack_days": 7}


def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    def flaky() -> pd.Series:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 429")
        return _series()

    sleeps: list[float] = []
    out = download_with_retry(flaky, max_retries=3, backoff_s=2.0, sleep=sleeps.append)
    assert len(out) == 10 and calls["n"] == 3
    assert sleeps == [2.0, 4.0]


def test_retry_gives_up():
    def broken() -> pd.Series:
        raise RuntimeError("HTTP 429")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        download_with_retry(broken, max_retries=2, backoff_s=0.0, sleep=lambda s: None)


def test_load_ticker_writes_then_reads_cache(tmp_path):
    cfg = _data_cfg(tmp_path)
    first = load_ticker("KO", cfg, fetch=_series)
    assert cache_path(tmp_path, "KO").exists()

    def must_not_fetch() -> pd.Series:
        raise AssertionError("cache should have been used")

    second = load_ticker("KO", cfg, fetch=must_not_fetch)
    pd.testing.assert_series_equal(first, second, check_freq=False)


def test_stale_cache_is_refreshed(tmp_path):
    cfg = _data_cfg(tmp_path)
    load_ticker("KO", cfg, fetch=lambda: _series(start="2020-01-01"))
    fresh = load_ticker("KO", cfg, fetch=_series)
    assert fresh.index.min() == pd.Timestamp("2014-01-02")

"""Equities data loading: yfinance adjusted closes with a per-ticker parquet cache."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

Downloader = Callable[[str, str, str | None], pd.Series]


def yfinance_download(ticker: str, start: str, end: str | None, price_field: str) -> pd.Series:
    """Download one ticker's ``price_field`` (e.g. ``"Adj Close"``) from yfinance, ``auto_adjust=False``."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False,
                      threads=False, multi_level_index=False)
    if raw is None or raw.empty or price_field not in raw:
        raise RuntimeError(f"yfinance returned no '{price_field}' data for {ticker}")
    col = raw[price_field]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    series = col.astype(float).rename(ticker)
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    series.index.name = "date"
    return series


def download_with_retry(fetch: Callable[[], pd.Series], max_retries: int, backoff_s: float,
                        sleep: Callable[[float], None] = time.sleep) -> pd.Series:
    """Call ``fetch`` up to ``max_retries`` times, sleeping ``backoff_s * 2**attempt`` between tries."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 - network/rate-limit errors vary by yfinance version
            last = exc
            wait = backoff_s * 2 ** attempt
            log.warning("download failed (%s), retry %d/%d in %.0fs", exc, attempt + 1, max_retries, wait)
            if attempt < max_retries - 1:
                sleep(wait)
    raise RuntimeError(f"download failed after {max_retries} attempts") from last


def cache_path(cache_dir: str | Path, ticker: str) -> Path:
    """Parquet cache file for one ticker."""
    return Path(cache_dir) / f"{ticker.replace('/', '_')}.parquet"


def read_cache(path: Path) -> pd.Series:
    """Read a cached price series (single column, DatetimeIndex)."""
    frame = pd.read_parquet(path)
    series = frame.iloc[:, 0].astype(float)
    series.index = pd.DatetimeIndex(series.index)
    return series


def load_ticker(ticker: str, data_cfg: dict, fetch: Callable[[], pd.Series] | None = None) -> pd.Series:
    """Load one ticker from cache, downloading (with retries) and caching on a miss.

    ``fetch`` overrides the yfinance call (used by tests). A cache whose first date is later
    than ``download_start`` plus a few days is treated as stale and refreshed.
    """
    path = cache_path(data_cfg["cache_dir"], ticker)
    start = data_cfg["download_start"]
    if path.exists():
        cached = read_cache(path)
        slack = pd.Timedelta(days=int(data_cfg["cache_start_slack_days"]))
        if not cached.empty and cached.index.min() <= pd.Timestamp(start) + slack:
            return cached.rename(ticker)
    if fetch is None:
        def fetch() -> pd.Series:
            return yfinance_download(ticker, start, data_cfg.get("download_end"), data_cfg["price_field"])
    series = download_with_retry(fetch, int(data_cfg["max_retries"]), float(data_cfg["retry_backoff_s"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename(ticker).to_frame().to_parquet(path)
    return series.rename(ticker)


def load_prices(tickers: list[str], data_cfg: dict) -> dict[str, pd.Series]:
    """Load every ticker (cached), returning raw series keyed by ticker."""
    return {t: load_ticker(t, data_cfg) for t in dict.fromkeys(tickers)}

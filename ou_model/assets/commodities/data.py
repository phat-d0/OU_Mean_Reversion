"""Download and cache yfinance continuous front-month futures (``=F``) closes."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    """Parquet file for one ticker (``BZ=F`` -> ``BZ_F.parquet``)."""
    return cache_dir / f"{ticker.replace('=', '_').replace('^', '')}.parquet"


def _download_one(ticker: str, start: str, end: str | None, field: str) -> pd.Series:
    """One yfinance request for ``ticker``; returns the ``field`` column as a float Series."""
    import yfinance as yf  # imported lazily so tests never need the network stack

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False,
                      threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    col = raw[field]
    if isinstance(col, pd.DataFrame):  # MultiIndex columns (field, ticker)
        col = col.iloc[:, 0]
    s = pd.Series(col, dtype=float, name=ticker)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s.index.name = "date"
    return s


def download_with_retry(ticker: str, start: str, end: str | None, field: str,
                        retries: int = 5, backoff: float = 10.0) -> pd.Series:
    """Download with exponential backoff (Yahoo answers HTTP 429 under load)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _download_one(ticker, start, end, field)
        except Exception as err:  # network / rate limit / empty response
            last = err
            wait = backoff * 2 ** attempt
            log.warning("download %s failed (%s); retry in %.0fs", ticker, err, wait)
            time.sleep(wait)
    raise RuntimeError(f"could not download {ticker} after {retries} attempts") from last


def load_prices(tickers: list[str], data_cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Close prices for ``tickers`` (one column each, outer-joined on date), cached as parquet.

    ``data_cfg`` is the ``data`` section of ``config/commodities.yaml`` (``download_start``,
    ``download_end``, ``price_field``, ``cache_dir``, optional ``retries``/``retry_backoff_s``).
    """
    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cols = {}
    for t in dict.fromkeys(tickers):
        path = _cache_path(cache_dir, t)
        if path.exists() and not refresh:
            s = pd.read_parquet(path).iloc[:, 0].rename(t)
        else:
            s = download_with_retry(t, data_cfg["download_start"], data_cfg.get("download_end"),
                                    data_cfg["price_field"], int(data_cfg.get("retries", 5)),
                                    float(data_cfg.get("retry_backoff_s", 10.0)))
            s.to_frame().to_parquet(path)
        cols[t] = s
    return pd.DataFrame(cols).sort_index()

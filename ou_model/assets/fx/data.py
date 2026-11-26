"""FX data loading: yfinance download with retries/backoff and a per-ticker parquet cache."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)

Fetcher = Callable[[str, str, str | None], pd.Series]


def cache_path(cache_dir: str | Path, ticker: str) -> Path:
    """Parquet cache file for ``ticker`` (non-alphanumerics replaced by ``_``)."""
    return Path(cache_dir) / f"{re.sub(r'[^A-Za-z0-9]+', '_', ticker)}.parquet"


def yfinance_fetch(ticker: str, start: str, end: str | None, price_field: str = "Close") -> pd.Series:
    """Download one ticker's daily ``price_field`` from Yahoo Finance (unadjusted bars)."""
    import yfinance as yf  # imported lazily so tests never need the network

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False,
                     threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    col = df[price_field]
    if isinstance(col, pd.DataFrame):  # yfinance >= 0.2 returns a (field, ticker) MultiIndex
        col = col.iloc[:, 0]
    s = pd.Series(col, dtype=float, name=ticker)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.sort_index()


def fetch_with_retry(fetch: Callable[[], pd.Series], max_retries: int, backoff_seconds: float,
                     sleep: Callable[[float], None] = time.sleep) -> pd.Series:
    """Call ``fetch`` up to ``max_retries`` times, waiting ``backoff_seconds * 2**k`` between tries."""
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch()
        except Exception as err:  # HTTP 429, empty frames, transient network errors
            last = err
            wait = backoff_seconds * 2 ** attempt
            log.warning("fetch failed (%s), retry %d/%d in %.0fs", err, attempt + 1, max_retries, wait)
            if attempt + 1 < max_retries:
                sleep(wait)
    raise RuntimeError(f"fetch failed after {max_retries} attempts: {last}")


def load_prices(tickers: list[str], data_cfg: dict, fetch: Fetcher | None = None) -> dict[str, pd.Series]:
    """Return raw daily prices per ticker, read from ``data_cfg['cache_dir']`` or downloaded once.

    ``data_cfg`` is the ``data`` section of ``config/fx.yaml``. ``fetch(ticker, start, end)``
    defaults to :func:`yfinance_fetch`; tests pass a synthetic fetcher.
    """
    field = data_cfg.get("price_field", "Close")
    fetch = fetch or (lambda t, s, e: yfinance_fetch(t, s, e, field))
    out: dict[str, pd.Series] = {}
    for ticker in dict.fromkeys(tickers):
        path = cache_path(data_cfg["cache_dir"], ticker)
        if path.is_file():
            s = pd.read_parquet(path).iloc[:, 0]
        else:
            s = fetch_with_retry(
                lambda t=ticker: fetch(t, data_cfg["download_start"], data_cfg.get("download_end")),
                int(data_cfg["max_retries"]), float(data_cfg["backoff_seconds"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            s.rename(ticker).to_frame().to_parquet(path)
        s = pd.Series(s, dtype=float, name=ticker)
        s.index = pd.DatetimeIndex(s.index)
        out[ticker] = s.sort_index()
    return out

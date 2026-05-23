"""
Data layer for the algorithmic trading system.

Pulls historical daily bars from Alpaca's Market Data API and caches them
locally as parquet files. Cache strategy: one parquet per symbol. On
subsequent calls, we read what's cached and only fetch the gap from the
last cached date to `end`. This keeps GitHub Actions runs cheap and fast.

Splits/dividends: Alpaca returns adjusted prices when `adjustment='all'`
is set on the request. We always use that, so the cached prices are
split- and dividend-adjusted at fetch time. NOTE: this means if a new
split happens after data is cached, the older cached values will be
stale relative to new fetches. The `force_refresh` flag handles that.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Columns we standardize on across the whole system. Alpaca's SDK returns
# these names already, but we enforce order + dtypes here so downstream
# modules can rely on the schema.
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]

DEFAULT_CACHE_DIR = Path("data/cache")


def _get_client() -> StockHistoricalDataClient:
    """
    Build an Alpaca historical data client from environment variables.

    Reads ALPACA_API_KEY and ALPACA_SECRET_KEY. Raises if either is missing
    — failing fast in CI is better than silently hitting an auth wall later.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment. "
            "Locally: put them in .env. In GitHub Actions: configure as repo secrets."
        )
    return StockHistoricalDataClient(api_key, secret_key)


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    """Path to the parquet file for a given symbol."""
    return cache_dir / f"{symbol.upper()}.parquet"


def _read_cache(symbol: str, cache_dir: Path) -> pd.DataFrame | None:
    """Read cached bars for a symbol, or None if no cache exists."""
    path = _cache_path(symbol, cache_dir)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # Index should be DatetimeIndex (tz-naive, date-only conceptually).
        # If somebody hand-edited the cache or the format changed, fail loudly.
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Cache for %s has non-datetime index, ignoring", symbol)
            return None
        return df
    except Exception as exc:  # noqa: BLE001 — cache corruption is recoverable
        logger.warning("Failed to read cache for %s: %s. Will refetch.", symbol, exc)
        return None


def _write_cache(df: pd.DataFrame, symbol: str, cache_dir: Path) -> None:
    """Atomically write the dataframe to the symbol's parquet cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, cache_dir)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, engine="pyarrow", compression="snappy")
    tmp_path.replace(path)  # atomic on POSIX, good enough on Windows


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    # Retry on Alpaca API errors (network/5xx) and OSError (DNS/socket). Do NOT
    # retry ValueError/TypeError/KeyError — those are programmer errors and
    # retrying 4x would just slow CI failures.
    retry=retry_if_exception_type((APIError, OSError, ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_from_alpaca(
    symbol: str,
    start: datetime,
    end: datetime,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:
    """
    Fetch daily bars from Alpaca for the given symbol and date range.

    Returns an empty DataFrame (with the right columns) if Alpaca has no
    data in the range — this happens at the edges, e.g. asking for today
    before the daily bar is published.

    Retries with exponential backoff: 4 attempts, 2s → 30s. Covers transient
    network blips and Alpaca's occasional 5xx.
    """
    client = _get_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,  # split + dividend adjusted
        feed=feed,
    )
    bars = client.get_stock_bars(request)

    # The SDK returns a BarSet; .df gives us a multi-indexed DataFrame
    # (symbol, timestamp). We single-symbol drop the symbol level.
    df = bars.df
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)

    # Alpaca returns tz-aware UTC timestamps. For daily bars we don't need
    # the time component or timezone — collapse to date-only midnight UTC,
    # but keep as Timestamp for clean pandas operations.
    df.index = pd.to_datetime(df.index).tz_convert("UTC").tz_localize(None).normalize()
    df.index.name = "timestamp"

    # Enforce column set and order. Alpaca may add columns over time; we
    # ignore extras to keep downstream code stable.
    for col in BAR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[BAR_COLUMNS].astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "trade_count": "float64",
            "vwap": "float64",
        }
    )

    return df.sort_index()


def get_bars(
    symbol: str,
    start: datetime | str,
    end: datetime | str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
    feed: DataFeed = DataFeed.IEX,
) -> pd.DataFrame:
    """
    Get daily OHLCV bars for `symbol` from `start` to `end` inclusive.

    Uses a local parquet cache. On a warm cache, only the gap between the
    last cached date and `end` is fetched from Alpaca. Returns a DataFrame
    indexed by tz-naive UTC midnight Timestamps with columns:
    open, high, low, close, volume, trade_count, vwap.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "SPY". Case-insensitive — stored uppercase.
    start, end : datetime | str
        Inclusive date range. Strings parsed as ISO ("2005-01-01").
        Naive datetimes treated as UTC.
    cache_dir : Path | str
        Directory for parquet cache. Created if missing.
    force_refresh : bool
        If True, bypass the cache entirely and refetch the full range.
        Use this when you suspect a corporate action invalidated old
        adjusted prices.
    feed : DataFeed
        IEX (free) or SIP (paid). Defaults to IEX.

    Returns
    -------
    pd.DataFrame
        Sliced to [start, end]. May be empty if no data exists for the range.

    Raises
    ------
    RuntimeError
        If Alpaca credentials are missing from the environment.
    ValueError
        If start > end.
    """
    symbol = symbol.upper()
    cache_dir = Path(cache_dir)
    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()

    if start_ts > end_ts:
        raise ValueError(f"start ({start_ts.date()}) is after end ({end_ts.date()})")

    cached = None if force_refresh else _read_cache(symbol, cache_dir)

    if cached is None or cached.empty:
        # Cold cache: fetch the whole range.
        logger.info("Fetching %s from %s to %s (cold cache)", symbol, start_ts.date(), end_ts.date())
        df = _fetch_from_alpaca(symbol, start_ts.to_pydatetime(), end_ts.to_pydatetime(), feed)
        if not df.empty:
            _write_cache(df, symbol, cache_dir)
        return df.loc[start_ts:end_ts].copy()

    # Warm cache. Three cases:
    #   1. Cache covers the whole request — slice and return.
    #   2. Request extends past cache — fetch the tail and merge.
    #   3. Request starts before cache — fetch the head and merge.
    # We handle (2) since it's the common case (daily updates). For (3),
    # we just refetch from start to be safe — old history is rarely
    # requested twice with different start dates.
    cache_min, cache_max = cached.index.min(), cached.index.max()

    if start_ts < cache_min:
        logger.info(
            "%s: requested start %s precedes cache start %s — refetching from start",
            symbol, start_ts.date(), cache_min.date(),
        )
        fetch_start = start_ts
    else:
        # Fetch from one day after the cache max to avoid duplicating the
        # last cached bar. If end <= cache_max, we fetch nothing.
        fetch_start = cache_max + pd.Timedelta(days=1)

    if fetch_start <= end_ts:
        logger.info("Fetching %s gap from %s to %s", symbol, fetch_start.date(), end_ts.date())
        new_bars = _fetch_from_alpaca(
            symbol, fetch_start.to_pydatetime(), end_ts.to_pydatetime(), feed
        )
        if not new_bars.empty:
            # Concat, drop any overlap (shouldn't happen, but be defensive),
            # and rewrite the cache.
            combined = pd.concat([cached, new_bars])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            _write_cache(combined, symbol, cache_dir)
            cached = combined

    result = cached.loc[start_ts:end_ts].copy()

    # Sanity check: flag big gaps (likely halts or our cache is broken).
    # Daily bars should have ~1 row per trading day; >5 calendar days
    # between consecutive rows mid-range is suspicious.
    if len(result) > 1:
        gaps = result.index.to_series().diff().dt.days
        big_gaps = gaps[gaps > 5]
        if not big_gaps.empty:
            logger.warning(
                "%s has %d gaps >5 days in [%s, %s]. First few: %s",
                symbol, len(big_gaps), start_ts.date(), end_ts.date(),
                big_gaps.head(3).to_dict(),
            )

    return result


if __name__ == "__main__":
    # Quick smoke test. Run with:
    #   ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python -m rsi_revert.data
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = get_bars("SPY", "2024-01-01", "2024-03-31")
    print(f"Fetched {len(df)} bars for SPY")
    print(df.head())
    print(df.tail())

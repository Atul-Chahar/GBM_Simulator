"""
data_fetcher.py — Binance API Data Pipeline
=============================================
Fetches BTCUSDT 1-hour kline (candlestick) data from Binance's
public API. Uses data-api.binance.vision to avoid geo-blocking
in India (as specified in the challenge brief).

No API key or account needed — all data is fully public.

Kline Response Format (per bar):
    [0] Open time (ms timestamp)
    [1] Open price
    [2] High price
    [3] Low price
    [4] Close price
    [5] Volume
    [6] Close time (ms timestamp)
    [7] Quote asset volume
    [8] Number of trades
    [9] Taker buy base asset volume
    [10] Taker buy quote asset volume
    [11] Ignore
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Primary endpoint (no geo-block). Fallback to api.binance.com if needed.
PRIMARY_URL = "https://data-api.binance.vision/api/v3/klines"
FALLBACK_URL = "https://api.binance.com/api/v3/klines"

# One hour in milliseconds
ONE_HOUR_MS = 3_600_000


def fetch_btc_klines(
    num_bars: int = 500,
    end_time: int = None,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """
    Fetch BTCUSDT hourly klines from Binance.

    Parameters
    ----------
    num_bars : int
        Total number of bars to fetch. Handles pagination automatically
        for requests > 1000 bars.
    end_time : int, optional
        End timestamp in milliseconds. Defaults to current time.
    interval : str
        Kline interval. Default "1h".
    symbol : str
        Trading pair. Default "BTCUSDT".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open_time, open, high, low, close, volume
        Indexed by open_time (datetime).
    """
    all_klines = []
    remaining = num_bars

    if end_time is None:
        end_time = int(time.time() * 1000)

    while remaining > 0:
        batch_size = min(remaining, 1000)

        # Calculate start_time for this batch
        start_time = end_time - (batch_size * ONE_HOUR_MS)

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": batch_size,
        }

        data = _request_with_fallback(params)

        if not data:
            break

        all_klines = data + all_klines  # Prepend (older data first)
        remaining -= len(data)

        # Move end_time back for next batch
        end_time = data[0][0] - 1  # 1ms before oldest bar in this batch

        # Respect rate limits
        time.sleep(0.1)

        if len(data) < batch_size:
            break

    return _klines_to_dataframe(all_klines)


def fetch_latest_bars(num_bars: int = 500) -> pd.DataFrame:
    """
    Fetch the most recent N closed 1-hour bars.
    Excludes the currently-forming (unclosed) bar.

    Parameters
    ----------
    num_bars : int
        Number of closed bars to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame of closed bars.
    """
    # Fetch one extra bar, then drop the last one (currently forming)
    df = fetch_btc_klines(num_bars=num_bars + 1)

    if len(df) <= 1:
        return df

    # The last row may be the currently-forming candle.
    # Check if its close_time is in the future.
    now_ms = int(time.time() * 1000)

    # Each bar's close_time marks when the bar closes.
    # If the last bar hasn't closed yet, drop it.
    if len(df) > 0:
        # Drop the last bar to ensure we only have closed bars
        df = df.iloc[:-1]

    return df.tail(num_bars)


def get_close_prices(df: pd.DataFrame) -> pd.Series:
    """
    Extract close prices as a pandas Series indexed by datetime.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from fetch_btc_klines or fetch_latest_bars.

    Returns
    -------
    pd.Series
        Close prices indexed by open_time.
    """
    return df["close"].copy()


def _request_with_fallback(params: dict) -> list:
    """
    Make API request with fallback URL on failure.
    """
    for url in [PRIMARY_URL, FALLBACK_URL]:
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            continue

    raise RuntimeError(
        "Failed to fetch data from both Binance endpoints. "
        "Check your internet connection."
    )


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    """
    Convert raw kline data to a clean DataFrame.
    """
    if not klines:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close", "volume"]
        )

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Keep only essential columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    # Remove duplicates (can happen with overlapping pagination)
    df = df[~df.index.duplicated(keep="last")]

    return df


# ─── Quick self-test ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching latest 10 BTCUSDT 1h bars...")
    df = fetch_latest_bars(num_bars=10)
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Close prices:\n{get_close_prices(df)}")

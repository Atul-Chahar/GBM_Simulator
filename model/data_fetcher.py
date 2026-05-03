"""
data_fetcher.py — Binance API Data Pipeline
============================================
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
from datetime import datetime, timedelta, timezone
import streamlit as st

PRIMARY_URL = "https://data-api.binance.vision/api/v3/klines"
FALLBACK_URL = "https://api.binance.com/api/v3/klines"
TIME_URL_PRIMARY = "https://api.binance.com/api/v3/time"
TIME_URL_FALLBACK = "https://data-api.binance.vision/api/v3/time"

ONE_HOUR_MS = 3_600_000

_binance_time_offset_ms = 0.0
_binance_time_last_check = 0.0
TIME_CACHE_TTL = 300


@st.cache_data(ttl=TIME_CACHE_TTL)
def _check_binance_time() -> float:
    global _binance_time_offset_ms, _binance_time_last_check
    now = time.time()
    if now - _binance_time_last_check < TIME_CACHE_TTL:
        return _binance_time_offset_ms

    time_urls = [TIME_URL_PRIMARY, TIME_URL_FALLBACK]
    for url in time_urls:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                server_time_ms = resp.json()["serverTime"]
                local_time_ms = int(time.time() * 1000)
                offset = server_time_ms - local_time_ms
                _binance_time_offset_ms = offset
                _binance_time_last_check = now
                if abs(offset) > 30_000:
                    print(f"⚠️ Binance clock drift: {offset/1000:.0f}s offset from local UTC")
                return offset
        except Exception:
            continue

    _binance_time_last_check = now
    return _binance_time_offset_ms


def fetch_btc_klines(
    num_bars: int = 500,
    end_time: int = None,
    interval: str = "1h",
    symbol: str = "BTCUSDT",
    closed_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch BTCUSDT hourly klines from Binance.
    """
    all_klines = []
    remaining = num_bars

    if end_time is None:
        end_time = int(time.time() * 1000)

    while remaining > 0:
        batch_size = min(remaining, 1000)

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

        all_klines = data + all_klines
        remaining -= len(data)

        end_time = data[0][0] - 1

        if remaining > 0:
            time.sleep(0.1)

        if len(data) < batch_size:
            break

    df = _klines_to_dataframe(all_klines)

    if closed_only and len(df) > 0:
        offset_ms = _check_binance_time()
        now = datetime.now(timezone.utc)
        last_close = df["close_time"].iloc[-1]
        adjusted_now = now + timedelta(milliseconds=int(offset_ms))
        if last_close > adjusted_now:
            df = df.iloc[:-1]

    return df


def fetch_latest_bars(num_bars: int = 500) -> pd.DataFrame:
    """
    Fetch the most recent N closed 1-hour bars.
    """
    df = fetch_btc_klines(num_bars=num_bars + 1, closed_only=True)
    return df.tail(num_bars)


def get_close_prices(df: pd.DataFrame) -> pd.Series:
    """
    Extract close prices as a pandas Series indexed by datetime.
    """
    return df["close"].copy()


def _request_with_fallback(params: dict) -> list:
    """
    Make API request with fallback URL on failure.
    """
    for url in [PRIMARY_URL, FALLBACK_URL]:
        try:
            response = requests.get(url, params=params, timeout=10)
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
            columns=["open_time", "close_time", "open", "high", "low", "close", "volume"]
        ).set_index("open_time")

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open_time", "close_time", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    df = df[~df.index.duplicated(keep="last")]

    return df


if __name__ == "__main__":
    print("Fetching latest 10 BTCUSDT 1h bars...")
    df = fetch_latest_bars(num_bars=10)
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Close prices:\n{get_close_prices(df)}")
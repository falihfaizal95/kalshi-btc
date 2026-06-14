"""
data/binance.py — Fetch OHLCV data from the Binance public REST API.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# binance.com returns HTTP 451 from US IPs; binance.us serves the same API.
BINANCE_BASES = [
    "https://api.binance.com/api/v3",
    "https://api.binance.us/api/v3",
]


def get_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT".
    interval : str
        Kline interval: "1m", "5m", "15m", "1h", "4h", "1d", etc.
    limit : int
        Number of candles to retrieve (max 1000).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (UTC datetime index), open, high, low, close, volume.
    """
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}

    raw = None
    last_exc: Exception | None = None
    for base in BINANCE_BASES:
        try:
            resp = requests.get(f"{base}/klines", params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Binance klines failed at %s: %s", base, exc)
    if raw is None:
        raise RuntimeError(f"Binance klines request failed ({interval}): {last_exc}") from last_exc

    if not raw:
        raise RuntimeError(f"Binance returned empty data for {symbol} {interval}.")

    # Binance kline format:
    # [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume, ...
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    logger.debug("Fetched %d %s candles for %s.", len(df), interval, symbol)
    return df


def get_current_price(symbol: str = "BTCUSDT") -> float:
    """
    Fetch the latest BTC/USD spot price from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT".

    Returns
    -------
    float
        Current price.
    """
    params = {"symbol": symbol}
    last_exc: Exception | None = None
    for base in BINANCE_BASES:
        try:
            resp = requests.get(f"{base}/ticker/price", params=params, timeout=10)
            resp.raise_for_status()
            price = float(resp.json()["price"])
            logger.debug("Binance spot price for %s: %.2f", symbol, price)
            return price
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_exc = exc
            logger.warning("Binance ticker/price failed at %s: %s", base, exc)
    raise RuntimeError(f"Binance ticker/price request failed for {symbol}: {last_exc}") from last_exc


def get_close_at(timestamp_ms: int, symbol: str = "BTCUSDT") -> Optional[float]:
    """
    Fetch the BTC close price for the 1h candle covering ``timestamp_ms``.

    Used to settle past predictions against the actual outcome. Returns None
    if no candle is available (e.g. timestamp is in the future).
    """
    params = {
        "symbol": symbol,
        "interval": "1h",
        "startTime": int(timestamp_ms),
        "limit": 1,
    }
    for base in BINANCE_BASES:
        try:
            resp = requests.get(f"{base}/klines", params=params, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            if raw:
                return float(raw[0][4])  # close
        except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
            logger.warning("Binance get_close_at failed at %s: %s", base, exc)
    return None


def get_ohlcv_multi(
    symbol: str = "BTCUSDT",
    intervals: Optional[List[str]] = None,
    limit: int = 500,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple timeframes.

    Parameters
    ----------
    symbol : str
        Trading pair.
    intervals : list of str
        Timeframes to fetch. Defaults to ["1m","5m","15m","1h","4h","1d"].
    limit : int
        Candles per timeframe.

    Returns
    -------
    dict mapping interval -> DataFrame
    """
    if intervals is None:
        intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]

    result: Dict[str, pd.DataFrame] = {}
    for interval in intervals:
        try:
            result[interval] = get_ohlcv(symbol=symbol, interval=interval, limit=limit)
        except Exception as exc:
            logger.warning("Failed to fetch %s %s: %s", symbol, interval, exc)
    return result

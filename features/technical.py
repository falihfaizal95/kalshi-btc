"""
features/technical.py — Compute technical indicators on OHLCV DataFrames using the `ta` library.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a comprehensive set of technical indicators and add them as columns.

    Input DataFrame must have columns: open, high, low, close, volume
    (case-insensitive). The index should be a sorted DatetimeIndex.

    Returns the same DataFrame with additional feature columns:
        rsi_14, macd_line, macd_signal, macd_hist,
        atr_14, bb_upper, bb_lower, bb_pct,
        adx_14, ema_9, ema_21, ema_50,
        volume_sma_20, volume_ratio,
        price_change_1h, price_change_4h, price_change_24h,
        hour_of_day, day_of_week
    """
    df = df.copy()

    # Normalise column names
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_features: missing columns {missing}")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ---- Momentum ----
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    df["rsi_14"] = rsi.rsi()

    # ---- Trend: MACD ----
    macd_ind = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_line"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    # ---- Volatility: ATR ----
    atr_ind = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr_14"] = atr_ind.average_true_range()

    # ---- Volatility: Bollinger Bands ----
    bb_ind = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb_ind.bollinger_hband()
    df["bb_lower"] = bb_ind.bollinger_lband()
    df["bb_pct"] = bb_ind.bollinger_pband()

    # ---- Trend: ADX ----
    adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["adx_14"] = adx_ind.adx()

    # ---- Trend: EMAs ----
    df["ema_9"] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

    # ---- Volume ----
    df["volume_sma_20"] = volume.rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = volume / df["volume_sma_20"].replace(0, np.nan)

    # ---- Price changes (percentage) ----
    df["price_change_1h"] = close.pct_change(periods=1)
    df["price_change_4h"] = close.pct_change(periods=4)
    df["price_change_24h"] = close.pct_change(periods=24)

    # ---- Temporal features ----
    if hasattr(df.index, "hour"):
        df["hour_of_day"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
    else:
        df["hour_of_day"] = 12
        df["day_of_week"] = 0

    logger.debug("compute_features: added %d feature columns to %d rows.", 19, len(df))
    return df

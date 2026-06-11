"""
features/engineer.py — Build a flat feature vector for a single market observation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from features.technical import compute_features

logger = logging.getLogger(__name__)

# The canonical ordered list of feature names the ML model trains/predicts on.
FEATURE_NAMES = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bb_upper",
    "bb_lower",
    "bb_pct",
    "adx_14",
    "ema_9",
    "ema_21",
    "ema_50",
    "volume_sma_20",
    "volume_ratio",
    "price_change_1h",
    "price_change_4h",
    "price_change_24h",
    "hour_of_day",
    "day_of_week",
    "distance_to_strike",
    "time_to_expiry_hours",
    "iv_annualized",
    "fear_greed_score",
    "volume_last_1h",
]


def get_feature_columns() -> list:
    """Return the canonical ordered list of feature column names."""
    return list(FEATURE_NAMES)


def build_feature_vector(
    btc_price: float,
    market: Dict[str, Any],
    ohlcv_1h: pd.DataFrame,
    iv: float,
    fear_greed_score: int,
) -> Dict[str, float]:
    """
    Build a flat feature dictionary for a single (market, timestamp) observation.

    Parameters
    ----------
    btc_price : float
        Current BTC/USD price.
    market : dict
        Market dict as returned by ``get_btc_markets()``.
        Expected keys: strike_price, expiry_time, market_id.
    ohlcv_1h : pd.DataFrame
        1-hour OHLCV DataFrame (columns: open, high, low, close, volume).
        Should have at least 50 rows for indicators to warm up.
    iv : float
        Annualized implied volatility (e.g. 0.65).
    fear_greed_score : int
        Fear & Greed index score 0–100.

    Returns
    -------
    dict mapping feature name -> float value.
    Missing / NaN features default to 0.0.
    """
    features: Dict[str, float] = {}

    # ---- Technical indicators from latest candle ----
    try:
        df_with_features = compute_features(ohlcv_1h)
        if len(df_with_features) == 0:
            raise ValueError("ohlcv_1h is empty after feature computation.")
        last = df_with_features.iloc[-1]

        tech_cols = [
            "rsi_14", "macd_line", "macd_signal", "macd_hist",
            "atr_14", "bb_upper", "bb_lower", "bb_pct",
            "adx_14", "ema_9", "ema_21", "ema_50",
            "volume_sma_20", "volume_ratio",
            "price_change_1h", "price_change_4h", "price_change_24h",
            "hour_of_day", "day_of_week",
        ]
        for col in tech_cols:
            val = last.get(col, 0.0)
            features[col] = float(val) if pd.notna(val) else 0.0

        # Volume of the latest 1h candle
        vol = last.get("volume", 0.0)
        features["volume_last_1h"] = float(vol) if pd.notna(vol) else 0.0

    except Exception as exc:
        logger.warning("Failed to compute technical features: %s. Using zeros.", exc)
        for col in FEATURE_NAMES:
            features.setdefault(col, 0.0)

    # ---- Market-specific features ----
    strike: Optional[float] = market.get("strike_price")
    if strike is not None and strike > 0 and btc_price > 0:
        features["distance_to_strike"] = (strike - btc_price) / btc_price
    else:
        features["distance_to_strike"] = 0.0

    expiry: Optional[datetime] = market.get("expiry_time")
    if expiry is not None:
        now = datetime.now(tz=timezone.utc)
        delta = expiry - now
        features["time_to_expiry_hours"] = max(0.0, delta.total_seconds() / 3600.0)
    else:
        features["time_to_expiry_hours"] = 1.0  # default 1h

    # ---- External / macro features ----
    features["iv_annualized"] = float(iv) if iv and iv > 0 else 0.65
    features["fear_greed_score"] = float(fear_greed_score)

    # Ensure all expected features are present (fill gaps with 0)
    for name in FEATURE_NAMES:
        features.setdefault(name, 0.0)

    return features

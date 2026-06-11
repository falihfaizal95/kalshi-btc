"""
backtest/simulate.py — Simulate historical Kalshi-style binary BTC markets.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from data.binance import get_ohlcv
from data.deribit import get_iv
from data.sentiment import get_fear_greed
from features.engineer import build_feature_vector
from models.lognormal import prob_above_strike

logger = logging.getLogger(__name__)

# Strike offsets relative to current price to simulate market variety
STRIKE_OFFSETS = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
EXPIRY_HOURS = [1, 2, 4]


def generate_historical_markets(
    ohlcv: pd.DataFrame,
    iv: float = 0.65,
    fear_greed: int = 50,
    min_rows: int = 60,
) -> pd.DataFrame:
    """
    For each row in the OHLCV history (representing a 1h candle), simulate
    multiple binary Kalshi-style markets and record features + actual outcomes.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        1h OHLCV with columns: open, high, low, close, volume, timestamp.
    iv : float
        Annualized implied volatility to use (static — live backtests can vary this).
    fear_greed : int
        Fear & Greed score to use (static).
    min_rows : int
        Minimum rows needed before generating a market (indicator warmup).

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, strike, expiry_hours, current_price,
        actual_outcome (1 = BTC above strike at expiry),
        kalshi_implied_prob, features..., lognormal_prob
    """
    records = []

    for i in range(min_rows, len(ohlcv) - max(EXPIRY_HOURS)):
        window = ohlcv.iloc[: i + 1].copy()
        current_price = float(window["close"].iloc[-1])
        ts = window.index[i] if hasattr(window.index, "__iter__") else i

        for expiry_h in EXPIRY_HOURS:
            # Actual future close price
            future_idx = i + expiry_h
            if future_idx >= len(ohlcv):
                continue
            future_price = float(ohlcv["close"].iloc[future_idx])

            for offset in STRIKE_OFFSETS:
                strike = round(current_price * (1 + offset), 0)

                # Actual outcome
                actual = 1 if future_price > strike else 0

                # Log-normal implied probability
                ln_prob = prob_above_strike(current_price, strike, expiry_h, iv)

                # Simulated Kalshi market dict
                from datetime import datetime, timezone, timedelta
                fake_market = {
                    "strike_price": strike,
                    "expiry_time": datetime.now(tz=timezone.utc) + timedelta(hours=expiry_h),
                    "market_id": f"SIM_{i}_{offset}_{expiry_h}",
                }

                try:
                    feats = build_feature_vector(
                        btc_price=current_price,
                        market=fake_market,
                        ohlcv_1h=window,
                        iv=iv,
                        fear_greed_score=fear_greed,
                    )
                except Exception as exc:
                    logger.debug("Feature build failed at i=%d: %s", i, exc)
                    continue

                # Simulate Kalshi pricing: log-normal + small noise
                noise = np.random.uniform(-0.03, 0.03)
                kalshi_implied = float(np.clip(ln_prob + noise, 0.02, 0.98))

                record = {
                    "timestamp": ts,
                    "strike": strike,
                    "expiry_hours": expiry_h,
                    "current_price": current_price,
                    "actual_outcome": actual,
                    "kalshi_implied_prob": kalshi_implied,
                    "lognormal_prob": ln_prob,
                }
                record.update(feats)
                records.append(record)

    if not records:
        logger.warning("No simulation records generated.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info("Generated %d simulated historical markets.", len(df))
    return df


def run_full_backtest(limit: int = 1000) -> pd.DataFrame:
    """
    Download historical BTC data and generate the full simulation DataFrame.
    Used by backtest/evaluate.py and main.py for model training.
    """
    logger.info("Downloading historical BTC OHLCV data...")
    ohlcv = get_ohlcv(interval="1h", limit=limit)

    logger.info("Fetching IV and sentiment for backtest...")
    try:
        iv = get_iv()
    except Exception:
        iv = 0.65
        logger.warning("Deribit IV fetch failed; using default 0.65")

    try:
        fg = get_fear_greed()
        fear_greed = fg["score"]
    except Exception:
        fear_greed = 50
        logger.warning("Fear & Greed fetch failed; using default 50")

    return generate_historical_markets(ohlcv, iv=iv, fear_greed=fear_greed)

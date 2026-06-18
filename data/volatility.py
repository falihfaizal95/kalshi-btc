"""
data/volatility.py — Short-horizon realized volatility for sub-hour pricing.

The lognormal model needs the volatility over the *market's actual horizon*
(minutes to ~1 hour), not Deribit's 30-day implied vol. Using 30-day DVOL for
sub-hour markets systematically overstates volatility and pushes near-the-money
probabilities toward 50% — the main calibration error seen in real outcomes.
This computes annualized realized vol from recent 1-minute returns.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_MINUTES_PER_YEAR = 525_600
# Sanity bounds for annualized BTC vol so a noisy/empty sample can't poison pricing.
_MIN_VOL = 0.15
_MAX_VOL = 2.5


def realized_vol(window_minutes: int = 120, fallback: float = 0.45) -> float:
    """
    Annualized realized volatility from the last ``window_minutes`` of 1-min BTC
    returns. Falls back to ``fallback`` if data is unavailable.
    """
    try:
        from data.binance import get_ohlcv
        df = get_ohlcv(interval="1m", limit=min(window_minutes + 5, 1000))
        rets = np.log(df["close"] / df["close"].shift(1)).dropna().tail(window_minutes)
        if len(rets) < 20:
            logger.warning("realized_vol: too few returns (%d); using fallback.", len(rets))
            return fallback
        rv = float(rets.std(ddof=1) * np.sqrt(_MINUTES_PER_YEAR))
        rv = max(_MIN_VOL, min(_MAX_VOL, rv))
        logger.debug("realized_vol (%dm window): %.1f%%", window_minutes, rv * 100)
        return rv
    except Exception as exc:
        logger.warning("realized_vol failed (%s); using fallback %.2f.", exc, fallback)
        return fallback


def pricing_vol(dvol: float, window_minutes: int = 120) -> float:
    """
    Volatility to price sub-hour markets: short-horizon realized vol, with the
    30-day DVOL as the fallback when realized vol can't be computed.
    """
    return realized_vol(window_minutes=window_minutes, fallback=dvol if dvol and dvol > 0 else 0.45)

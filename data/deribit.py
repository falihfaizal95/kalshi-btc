"""
data/deribit.py — Fetch BTC implied volatility and spot price from Deribit public API.
"""

from __future__ import annotations

import logging
from typing import Tuple

import requests

logger = logging.getLogger(__name__)

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
DEFAULT_IV = 0.65  # 65% annualized fallback


def get_spot_price() -> float:
    """
    Fetch current BTC spot price from Deribit index.

    Returns
    -------
    float
        BTC/USD spot price.
    """
    url = f"{DERIBIT_BASE}/get_index_price"
    params = {"index_name": "btc_usd"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price = data["result"]["index_price"]
        logger.debug("Deribit BTC spot price: %.2f", price)
        return float(price)
    except Exception as exc:
        logger.warning("Failed to fetch Deribit spot price: %s", exc)
        raise


def get_iv(expiry_approx_hours: int = 24) -> float:
    """
    Fetch BTC implied volatility from Deribit volatility index.

    Uses the DVOL index (Deribit Volatility Index for BTC) which represents
    30-day annualized implied volatility.

    Parameters
    ----------
    expiry_approx_hours : int
        Approximate hours to expiry (used for logging context only; DVOL
        is always 30-day). For very short expiries the term-structure
        premium is not modelled here.

    Returns
    -------
    float
        Annualized implied volatility as a decimal (e.g. 0.65 for 65%).
        Falls back to DEFAULT_IV (0.65) if the API is unavailable.
    """
    # Try volatility index endpoint first
    url = f"{DERIBIT_BASE}/get_volatility_index_data"
    # Request last 2 data points at 1-hour resolution
    import time as _time
    end_ts = int(_time.time() * 1000)
    start_ts = end_ts - 2 * 3600 * 1000  # 2 hours back

    params = {
        "currency": "BTC",
        "resolution": "3600",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("result", {})
        ticks = result.get("data", [])

        if ticks:
            # Each tick: [timestamp_ms, open, high, low, close]
            last_close = float(ticks[-1][4])
            iv = last_close / 100.0  # DVOL is in percentage points
            logger.debug(
                "Deribit DVOL: %.1f%% annualized (expiry ~%dh)", last_close, expiry_approx_hours
            )
            return iv

    except Exception as exc:
        logger.warning("Deribit volatility index request failed: %s. Trying fallback.", exc)

    # Fallback: try to derive IV from ATM option
    try:
        iv = _derive_iv_from_options(expiry_approx_hours)
        return iv
    except Exception as exc2:
        logger.warning(
            "Deribit option IV fallback failed: %s. Using default IV=%.2f.", exc2, DEFAULT_IV
        )
        return DEFAULT_IV


def _derive_iv_from_options(expiry_approx_hours: int) -> float:
    """
    Fallback: fetch nearest ATM option's mark IV from Deribit instruments.
    Returns annualized IV as decimal.
    """
    # Get current spot price
    spot = get_spot_price()

    # Get available BTC options
    url = f"{DERIBIT_BASE}/get_instruments"
    params = {"currency": "BTC", "kind": "option", "expired": "false"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    instruments = resp.json()["result"]

    import time as _time
    now = _time.time()
    target_expiry = now + expiry_approx_hours * 3600

    # Find the instrument with closest expiry and ATM strike
    best_instrument = None
    best_score = float("inf")

    for inst in instruments:
        if inst.get("option_type") != "call":
            continue
        exp_ts = inst.get("expiration_timestamp", 0) / 1000.0
        strike = inst.get("strike", 0)
        if exp_ts < now:
            continue
        time_diff = abs(exp_ts - target_expiry)
        strike_diff = abs(strike - spot) / spot
        score = time_diff / 3600 + strike_diff * 10  # weight strike diff
        if score < best_score:
            best_score = score
            best_instrument = inst

    if not best_instrument:
        raise RuntimeError("No suitable BTC option found on Deribit.")

    # Fetch ticker for mark IV
    ticker_url = f"{DERIBIT_BASE}/get_ticker"
    ticker_params = {"instrument_name": best_instrument["instrument_name"]}
    ticker_resp = requests.get(ticker_url, params=ticker_params, timeout=10)
    ticker_resp.raise_for_status()
    ticker_data = ticker_resp.json()["result"]

    mark_iv = ticker_data.get("mark_iv", DEFAULT_IV * 100) / 100.0
    logger.debug(
        "Derived IV from %s: %.1f%%", best_instrument["instrument_name"], mark_iv * 100
    )
    return mark_iv

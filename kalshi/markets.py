"""
kalshi/markets.py — Fetch and parse BTC binary markets from Kalshi.

Uses the structured market fields (strike_type, floor_strike, cap_strike)
rather than regex-parsing tickers, and queries known BTC series directly
instead of paging through every open market on the exchange.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Known Kalshi BTC price series. KXBTCD = daily settlement, KXBTC = hourly.
BTC_SERIES = ["KXBTCD", "KXBTC"]

# Strike types the lognormal/ML pipeline can price.
SUPPORTED_STRIKE_TYPES = {"greater", "greater_or_equal", "less", "less_or_equal", "between"}


def _parse_expiry(market: Dict[str, Any]) -> Optional[datetime]:
    """Parse expiry time from market dict (ISO 8601 strings)."""
    for key in ("close_time", "expiration_time", "latest_expiration_time"):
        raw = market.get(key)
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, AttributeError):
            continue
    return None


def _price_cents(market: Dict[str, Any], cent_key: str, dollar_key: str, default: float) -> float:
    """Read a price in cents, falling back to the *_dollars field."""
    val = market.get(cent_key)
    if val is not None and val != 0:
        return float(val)
    dollars = market.get(dollar_key)
    if dollars is not None:
        try:
            return float(dollars) * 100.0
        except (TypeError, ValueError):
            pass
    return default


def _parse_market(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw API market dict into our normalized format, or None to skip."""
    strike_type = (m.get("strike_type") or "").lower()
    if strike_type not in SUPPORTED_STRIKE_TYPES:
        return None

    floor_strike = m.get("floor_strike")
    cap_strike = m.get("cap_strike")

    # Normalize: "greater*" needs floor, "less*" needs cap, "between" needs both.
    if strike_type.startswith("greater"):
        strike_type = "greater"
        if floor_strike is None:
            return None
    elif strike_type.startswith("less"):
        strike_type = "less"
        if cap_strike is None:
            return None
    elif strike_type == "between":
        if floor_strike is None or cap_strike is None:
            return None

    ticker: str = m.get("ticker", "")
    return {
        "market_id": ticker,
        "ticker": ticker,
        "title": m.get("title", ""),
        "yes_bid": _price_cents(m, "yes_bid", "yes_bid_dollars", 0.0),
        "yes_ask": _price_cents(m, "yes_ask", "yes_ask_dollars", 100.0),
        "no_bid": _price_cents(m, "no_bid", "no_bid_dollars", 0.0),
        "no_ask": _price_cents(m, "no_ask", "no_ask_dollars", 100.0),
        "expiry_time": _parse_expiry(m),
        "strike_type": strike_type,
        "floor_strike": float(floor_strike) if floor_strike is not None else None,
        "cap_strike": float(cap_strike) if cap_strike is not None else None,
        # Representative strike used for display / distance features.
        "strike_price": float(floor_strike if floor_strike is not None else cap_strike),
        "volume": m.get("volume", 0),
        "liquidity": m.get("liquidity", 0),
    }


def _fetch_series(client, series_ticker: str) -> List[Dict[str, Any]]:
    """Fetch all open markets for one series, following pagination."""
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page_size = 200

    while True:
        params: Dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": page_size,
        }
        if cursor:
            params["cursor"] = cursor

        data = client.get("/markets", params=params)
        raw_markets = data.get("markets", [])

        for m in raw_markets:
            parsed = _parse_market(m)
            if parsed is not None:
                out.append(parsed)

        cursor = data.get("cursor")
        if not cursor or not raw_markets:
            break

    return out


def get_btc_markets(client) -> List[Dict[str, Any]]:
    """
    Fetch open Kalshi BTC price markets across known BTC series.

    Returns a list of dicts with keys:
        market_id, ticker, title, yes_bid, yes_ask, no_bid, no_ask,
        expiry_time, strike_type ("greater"|"less"|"between"),
        floor_strike, cap_strike, strike_price, volume, liquidity
    """
    markets: List[Dict[str, Any]] = []
    for series in BTC_SERIES:
        try:
            found = _fetch_series(client, series)
            logger.info("Series %s: %d open markets.", series, len(found))
            markets.extend(found)
        except Exception as exc:
            logger.error("Failed to fetch series %s: %s", series, exc)

    # De-dupe by ticker in case series overlap
    seen = set()
    unique = []
    for m in markets:
        if m["ticker"] not in seen:
            seen.add(m["ticker"])
            unique.append(m)

    logger.info("Found %d open BTC markets on Kalshi.", len(unique))
    return unique

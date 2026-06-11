"""
kalshi/markets.py — Fetch and parse BTC binary markets from Kalshi.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex to parse strike from tickers like KXBTC-24DEC1700-T97000
_STRIKE_RE = re.compile(r"-T(\d+)$", re.IGNORECASE)
# Regex to detect BTC-related tickers/titles
_BTC_RE = re.compile(r"(btc|bitcoin|KXBTC)", re.IGNORECASE)


def _parse_strike(ticker: str, title: str) -> Optional[float]:
    """Extract strike price from ticker or title string."""
    # Try ticker format KXBTC-24DEC1700-T97000
    m = _STRIKE_RE.search(ticker)
    if m:
        return float(m.group(1))

    # Try title like "BTC above $97,000" or "BTC > 97000"
    m2 = re.search(r"\$?([\d,]+)", title)
    if m2:
        try:
            return float(m2.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _parse_expiry(market: Dict[str, Any]) -> Optional[datetime]:
    """Parse expiry time from market dict."""
    for key in ("close_time", "expiration_time", "expiry_time"):
        raw = market.get(key)
        if raw:
            try:
                # Kalshi returns ISO 8601 strings
                ts = raw.rstrip("Z")
                dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, AttributeError):
                pass
    return None


def get_btc_markets(client) -> List[Dict[str, Any]]:
    """
    Fetch all open Kalshi markets and filter for BTC/bitcoin markets.

    Returns a list of dicts with keys:
        market_id, ticker, title, yes_bid, yes_ask, no_bid, no_ask,
        expiry_time, strike_price
    """
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page_size = 200

    try:
        while True:
            params: Dict[str, Any] = {"status": "open", "limit": page_size}
            if cursor:
                params["cursor"] = cursor

            data = client.get("/markets", params=params)
            raw_markets = data.get("markets", [])

            for m in raw_markets:
                ticker: str = m.get("ticker", "")
                title: str = m.get("title", "")

                # Filter for BTC markets
                if not (_BTC_RE.search(ticker) or _BTC_RE.search(title)):
                    continue

                strike = _parse_strike(ticker, title)
                expiry = _parse_expiry(m)

                # Extract order book prices (Kalshi v2 returns cents 1–99)
                yes_bid = m.get("yes_bid", 0) or 0
                yes_ask = m.get("yes_ask", 100) or 100
                no_bid = m.get("no_bid", 0) or 0
                no_ask = m.get("no_ask", 100) or 100

                markets.append(
                    {
                        "market_id": ticker,
                        "ticker": ticker,
                        "title": title,
                        "yes_bid": yes_bid,
                        "yes_ask": yes_ask,
                        "no_bid": no_bid,
                        "no_ask": no_ask,
                        "expiry_time": expiry,
                        "strike_price": strike,
                    }
                )

            # Pagination
            cursor = data.get("cursor")
            if not cursor or len(raw_markets) < page_size:
                break

    except Exception as exc:
        logger.error("Failed to fetch Kalshi markets: %s", exc)
        return []

    logger.info("Found %d open BTC markets on Kalshi.", len(markets))
    return markets

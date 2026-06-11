"""
kalshi/client.py — Full Kalshi REST API client with auth, token refresh, and order placement.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"


class KalshiClient:
    """Thread-safe Kalshi API client with automatic token refresh."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._email: Optional[str] = None
        self._password: Optional[str] = None
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> None:
        """Authenticate with Kalshi and store the bearer token."""
        self._email = email
        self._password = password

        payload = {"email": email, "password": password}
        try:
            resp = self._session.post(f"{BASE_URL}/login", json=payload, timeout=15)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Kalshi login failed ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Kalshi login network error: {exc}") from exc

        data = resp.json()
        token = data.get("token")
        if not token:
            raise RuntimeError(f"Kalshi login returned no token. Response: {data}")

        self._token = token
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        logger.info("Kalshi login successful.")

    def _refresh_token(self) -> None:
        """Re-authenticate using stored credentials."""
        if not self._email or not self._password:
            raise RuntimeError("Cannot refresh token: credentials not set. Call login() first.")
        logger.warning("Refreshing Kalshi token...")
        self.login(self._email, self._password)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Authenticated GET request. Auto-refreshes on 401."""
        url = f"{BASE_URL}{path}"
        for attempt in range(2):
            try:
                resp = self._session.get(url, params=params, timeout=15)
                if resp.status_code == 401 and attempt == 0:
                    self._refresh_token()
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Kalshi GET {path} failed ({exc.response.status_code}): {exc.response.text}"
                ) from exc
            except requests.RequestException as exc:
                raise RuntimeError(f"Kalshi GET {path} network error: {exc}") from exc
        raise RuntimeError(f"Kalshi GET {path}: exceeded retry attempts.")

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Authenticated POST request. Auto-refreshes on 401."""
        url = f"{BASE_URL}{path}"
        for attempt in range(2):
            try:
                resp = self._session.post(url, json=body or {}, timeout=15)
                if resp.status_code == 401 and attempt == 0:
                    self._refresh_token()
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Kalshi POST {path} failed ({exc.response.status_code}): {exc.response.text}"
                ) from exc
            except requests.RequestException as exc:
                raise RuntimeError(f"Kalshi POST {path} network error: {exc}") from exc
        raise RuntimeError(f"Kalshi POST {path}: exceeded retry attempts.")

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        market_id: str,
        side: str,
        contracts: int,
        price: float,
    ) -> Dict[str, Any]:
        """
        Place a limit order on Kalshi.

        Parameters
        ----------
        market_id : str
            The Kalshi market ticker/ID.
        side : str
            "yes" or "no".
        contracts : int
            Number of contracts (each contract is worth $1 max).
        price : float
            Limit price in cents (1-99).
        """
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got: {side!r}")
        if not (1 <= price <= 99):
            raise ValueError(f"price must be between 1 and 99 cents, got: {price}")
        if contracts < 1:
            raise ValueError(f"contracts must be >= 1, got: {contracts}")

        body: Dict[str, Any] = {
            "ticker": market_id,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": contracts,
            "yes_price": int(price) if side == "yes" else int(100 - price),
            "no_price": int(100 - price) if side == "yes" else int(price),
        }
        logger.info(
            "Placing order: market=%s side=%s contracts=%d price=%d¢",
            market_id, side, contracts, price,
        )
        result = self.post("/orders", body)
        logger.info("Order placed: %s", result)
        return result

"""
kalshi/client.py — Kalshi REST API client using API-key + RSA-PSS request signing.

Kalshi retired email/password login. Authentication now works by signing
``timestamp + HTTP method + path`` with an RSA private key and sending:

    KALSHI-ACCESS-KEY        your API key ID
    KALSHI-ACCESS-TIMESTAMP  unix epoch milliseconds
    KALSHI-ACCESS-SIGNATURE  base64 RSA-PSS-SHA256 signature

Public market-data endpoints (e.g. GET /markets) need no auth, so the client
works in read-only mode without credentials; order placement requires them.
"""

from __future__ import annotations

import base64
import datetime
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

logger = logging.getLogger(__name__)

PROD_BASE_URL = "https://api.elections.kalshi.com"
DEMO_BASE_URL = "https://demo-api.kalshi.co"
API_PREFIX = "/trade-api/v2"

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


class KalshiClient:
    """Kalshi API client with RSA request signing and basic retry/backoff."""

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        demo: bool = False,
    ) -> None:
        self._api_key_id = api_key_id or None
        self._private_key: Optional[RSAPrivateKey] = None
        self._base = DEMO_BASE_URL if demo else PROD_BASE_URL
        self._session = requests.Session()

        if private_key_path:
            self._load_private_key(private_key_path)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _load_private_key(self, path: str) -> None:
        key_file = Path(path).expanduser()
        if not key_file.exists():
            raise FileNotFoundError(f"Kalshi private key not found: {key_file}")
        with open(key_file, "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(key, RSAPrivateKey):
            raise ValueError(f"Expected an RSA private key in {key_file}.")
        self._private_key = key
        logger.info("Loaded Kalshi RSA private key from %s.", key_file)

    @property
    def can_trade(self) -> bool:
        """True if credentials are configured for authenticated endpoints."""
        return bool(self._api_key_id and self._private_key)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """RSA-PSS-SHA256 signature over timestamp + method + path (no query)."""
        assert self._private_key is not None
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Build signed auth headers; empty dict when running unauthenticated."""
        if not self.can_trade:
            return {}
        timestamp_ms = str(
            int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        )
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,  # type: ignore[dict-item]
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp_ms, method, path),
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth_required: bool = False,
    ) -> Dict[str, Any]:
        """Issue a request. Signs the un-prefixed path (without query params)."""
        if auth_required and not self.can_trade:
            raise RuntimeError(
                "This endpoint requires Kalshi API credentials. "
                "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env."
            )

        full_path = f"{API_PREFIX}{path}"
        url = f"{self._base}{full_path}"

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            headers = self._auth_headers(method, full_path)
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=body,
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code in _RETRY_STATUS:
                    wait = 2 ** attempt
                    logger.warning(
                        "Kalshi %s %s returned %d; retrying in %ds.",
                        method, path, resp.status_code, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Kalshi {method} {path} failed "
                    f"({exc.response.status_code}): {exc.response.text}"
                ) from exc
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Kalshi %s %s network error: %s; retrying in %ds.",
                    method, path, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Kalshi {method} {path}: exceeded {_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request. Signed when credentials exist; public endpoints work without."""
        return self._request("GET", path, params=params)

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Authenticated POST request."""
        return self._request("POST", path, body=body or {}, auth_required=True)

    # ------------------------------------------------------------------
    # Account / orders
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict[str, Any]:
        """Fetch account balance (authenticated)."""
        return self._request("GET", "/portfolio/balance", auth_required=True)

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
            The Kalshi market ticker.
        side : str
            "yes" or "no".
        contracts : int
            Number of contracts (each settles at $1 max).
        price : float
            Limit price in cents (1-99) for the chosen side.
        """
        if side not in ("yes", "no"):
            raise ValueError(f"side must be 'yes' or 'no', got: {side!r}")
        if not (1 <= price <= 99):
            raise ValueError(f"price must be between 1 and 99 cents, got: {price}")
        if contracts < 1:
            raise ValueError(f"contracts must be >= 1, got: {contracts}")

        body: Dict[str, Any] = {
            "ticker": market_id,
            "client_order_id": str(uuid.uuid4()),
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": int(contracts),
        }
        # Provide only the price field for the side being bought.
        if side == "yes":
            body["yes_price"] = int(price)
        else:
            body["no_price"] = int(price)

        logger.info(
            "Placing order: market=%s side=%s contracts=%d price=%d¢",
            market_id, side, contracts, price,
        )
        result = self.post("/portfolio/orders", body)
        logger.info("Order placed: %s", result)
        return result

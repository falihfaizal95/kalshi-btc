"""
data/sentiment.py — Fetch Crypto Fear & Greed Index from Alternative.me API.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import requests

logger = logging.getLogger(__name__)

FNG_URL = "https://api.alternative.me/fng/?limit=1"
_FALLBACK = {"score": 50, "classification": "Neutral"}


def get_fear_greed() -> Dict[str, Union[int, str]]:
    """
    Fetch the latest Crypto Fear & Greed Index.

    Returns
    -------
    dict with keys:
        'score'          : int, 0 (Extreme Fear) to 100 (Extreme Greed)
        'classification' : str, e.g. "Fear", "Greed", "Neutral", etc.

    Falls back to {'score': 50, 'classification': 'Neutral'} on any error.
    """
    try:
        resp = requests.get(FNG_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("data", [])
        if not items:
            logger.warning("Fear & Greed API returned no data items.")
            return dict(_FALLBACK)

        entry = items[0]
        score = int(entry.get("value", 50))
        classification = str(entry.get("value_classification", "Neutral"))

        logger.debug("Fear & Greed: %d (%s)", score, classification)
        return {"score": score, "classification": classification}

    except requests.RequestException as exc:
        logger.warning("Fear & Greed API request failed: %s. Using fallback.", exc)
        return dict(_FALLBACK)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Fear & Greed API parse error: %s. Using fallback.", exc)
        return dict(_FALLBACK)

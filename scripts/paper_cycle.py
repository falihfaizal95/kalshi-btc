#!/usr/bin/env python3
"""
scripts/paper_cycle.py — One paper-trading cycle (settle matured + open new).

Designed to be invoked on a timer (e.g. a launchd agent every 15 minutes) so
the paper account trades and learns continuously in the background. Always
settles first, so matured positions resolve even when no new opportunities or
the market scan fails.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paper_cycle")


def main() -> None:
    if not cfg.PAPER_TRADE:
        logger.info("PAPER_TRADE disabled; nothing to do.")
        return

    from kalshi.client import KalshiClient
    from alerts.engine import scan_markets
    from paper.account import paper_trade_cycle

    client = KalshiClient(
        api_key_id=cfg.KALSHI_API_KEY_ID or None,
        private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
        demo=cfg.KALSHI_DEMO,
    )

    try:
        qualifying = scan_markets(client, cfg)
    except Exception as exc:
        logger.exception("Scan failed (%s); still settling matured positions.", exc)
        qualifying = []

    s = paper_trade_cycle(qualifying, cfg)
    logger.info(
        "Paper cycle done: equity=$%.2f realized=$%+.2f open=%d closed=%d win=%.0f%%",
        s["equity"], s["realized_pnl"], s["open_positions"], s["closed_trades"],
        s["win_rate"] * 100,
    )


if __name__ == "__main__":
    main()

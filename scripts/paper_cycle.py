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
    import prediction_log

    client = KalshiClient(
        api_key_id=cfg.KALSHI_API_KEY_ID or None,
        private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
        demo=cfg.KALSHI_DEMO,
    )

    # Always settle matured predictions first (real outcomes), even if the scan fails.
    prediction_log.settle_predictions(cfg)

    try:
        all_alerts = scan_markets(client, cfg, full=True)
    except Exception as exc:
        logger.exception("Scan failed (%s); skipping new predictions/trades.", exc)
        all_alerts = []

    # Record the full unbiased dataset (every liquid market), then trade the edges.
    prediction_log.record_predictions(all_alerts, cfg)
    qualifying = [a for a in all_alerts if a.get("abs_edge", 0) >= cfg.EDGE_THRESHOLD]

    s = paper_trade_cycle(qualifying, cfg)
    cal = prediction_log.calibration_summary(cfg)
    logger.info(
        "Paper cycle done: equity=$%.2f realized=$%+.2f open=%d closed=%d win=%.0f%% | "
        "dataset: %d settled / %d open, brier=%.3f, calib_gap=%+.1f%%",
        s["equity"], s["realized_pnl"], s["open_positions"], s["closed_trades"],
        s["win_rate"] * 100, cal["settled"], cal["open"], cal["brier"],
        cal["calibration_gap"] * 100,
    )


if __name__ == "__main__":
    main()

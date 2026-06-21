#!/usr/bin/env python3
"""
scripts/recommend.py — "What should I enter right now?" phone-alert recommender.

Prints a concise, phone-readable list of the highest-conviction manual trades on
live Kalshi BTC markets. Prioritizes the CONFIDENT WINDOW — markets expiring
within ~30 minutes — because near expiry the outcome is more locked in, so the
model's edge is more reliable. For each play it tells you the side, the exact
limit price to enter at, and the model's read vs. the market.

Usage:
    python scripts/recommend.py            # confident window (<=30 min)
    python scripts/recommend.py --hour     # also show the rest of the hour
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def _fmt(a: dict, now: datetime) -> str:
    mins = (a["expiry_dt"] - now).total_seconds() / 60.0
    side = a["direction"]
    entry = int(a["yes_ask"]) if side == "YES" else int(a["no_ask"])
    return (
        f"  {side:3} @ {entry:>2}c  | exp.return {a.get('ev', 0):+5.0%}  model {a['model_prob']:>4.0%}  "
        f"| {mins:>4.0f}m left | {a['market_id']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hour", action="store_true", help="also show plays in the rest of the hour")
    args = ap.parse_args()

    from kalshi.client import KalshiClient
    from alerts.engine import scan_markets

    client = KalshiClient(
        api_key_id=cfg.KALSHI_API_KEY_ID or None,
        private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
        demo=cfg.KALSHI_DEMO,
    )

    # Score every liquid market in the next hour, then split by time-to-expiry.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # suppress the scanner's big table
        alerts = scan_markets(client, cfg, full=True)

    now = datetime.now(timezone.utc)
    window = cfg.CONFIDENT_WINDOW_MINUTES

    ev_thr = getattr(cfg, "EV_THRESHOLD", 0.15)
    qual = [a for a in alerts if a.get("ev", 0) >= ev_thr]
    confident = sorted(
        [a for a in qual if (a["expiry_dt"] - now).total_seconds() / 60.0 <= window],
        key=lambda a: a.get("ev", 0), reverse=True,
    )
    rest = sorted(
        [a for a in qual if (a["expiry_dt"] - now).total_seconds() / 60.0 > window],
        key=lambda a: a.get("ev", 0), reverse=True,
    )

    print(f"\nBTC ${alerts[0]['current_price']:,.0f}  |  {now:%H:%M UTC}  |  "
          f"confident window: <={window:.0f} min to expiry\n" if alerts else "No markets.\n")

    print(f"=== CONFIDENT PLAYS (<= {window:.0f} min, ranked by edge) ===")
    if confident:
        for a in confident[:8]:
            print(_fmt(a, now))
        print("\n  -> 'BUY YES @ 18c' means place a limit buy on the YES side at 18 cents.")
    else:
        print("  none right now — wait until a batch is within the window.")

    if args.hour:
        print(f"\n=== REST OF THE HOUR (>{window:.0f} min, lower confidence) ===")
        for a in rest[:8]:
            print(_fmt(a, now))

    print()


if __name__ == "__main__":
    main()

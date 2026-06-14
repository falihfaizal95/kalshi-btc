"""
paper/account.py — Virtual paper-trading account.

Simulates buying Kalshi contracts at the ask price against a virtual bankroll,
then settles each position against the actual BTC outcome once the market
expires. No real money or orders are involved. The resulting trade history is a
real-outcome dataset the strategy can learn from.

State lives in a single CSV (config.PAPER_TRADES_CSV), committed to git so the
account persists and compounds across runs. Equity is recomputed from the CSV
itself — there is no separate balance file to corrupt.
"""

from __future__ import annotations

import csv
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FIELDS = [
    "trade_id", "opened_ts", "market_id", "strike_type",
    "floor_strike", "cap_strike", "expiry_iso", "side", "contracts",
    "entry_price_cents", "stake_usd", "model_prob", "win_prob",
    "status", "settled_ts", "actual_close", "pnl",
]

# Don't open more than this many new paper positions per cycle (risk guard).
MAX_NEW_POSITIONS_PER_CYCLE = 10


def _kelly_fraction_of_bankroll(win_prob: float, cost: float, kelly_fraction: float, max_bet_pct: float) -> float:
    """Fractional-Kelly bankroll fraction for a binary contract bought at `cost`."""
    if cost <= 0 or cost >= 1:
        return 0.0
    full = (win_prob - cost) / (1.0 - cost)
    if full <= 0:
        return 0.0
    return min(kelly_fraction * full, max_bet_pct)


class PaperAccount:
    def __init__(self, csv_path: Path, starting_bankroll: float) -> None:
        self.csv_path = Path(csv_path)
        self.starting_bankroll = float(starting_bankroll)
        self.trades: List[Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def _save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.trades)

    # ------------------------------------------------------------------
    @property
    def realized_pnl(self) -> float:
        return sum(float(t["pnl"]) for t in self.trades if t["status"] in ("won", "lost"))

    @property
    def open_stake(self) -> float:
        return sum(float(t["stake_usd"]) for t in self.trades if t["status"] == "open")

    @property
    def equity(self) -> float:
        """Total account value: starting bankroll plus realized P&L."""
        return self.starting_bankroll + self.realized_pnl

    @property
    def available_cash(self) -> float:
        """Equity not currently tied up in open positions."""
        return self.equity - self.open_stake

    # ------------------------------------------------------------------
    def settle(self) -> int:
        """Settle open positions whose markets have expired. Returns count settled."""
        from data.binance import get_close_at

        now = datetime.now(timezone.utc)
        settled = 0
        for t in self.trades:
            if t["status"] != "open":
                continue
            try:
                expiry = datetime.fromisoformat(t["expiry_iso"])
            except (ValueError, KeyError):
                continue
            if expiry > now:
                continue

            close = get_close_at(int(expiry.timestamp() * 1000))
            if close is None:
                continue  # price not available yet; retry next cycle

            floor = float(t["floor_strike"]) if t.get("floor_strike") else None
            cap = float(t["cap_strike"]) if t.get("cap_strike") else None
            st = t.get("strike_type", "greater")
            if st == "greater":
                actual_yes = close > (floor if floor is not None else 0)
            elif st == "less":
                actual_yes = close < (cap if cap is not None else float("inf"))
            elif st == "between" and floor is not None and cap is not None:
                actual_yes = floor <= close <= cap
            else:
                continue

            side = t["side"]
            won = (side == "YES" and actual_yes) or (side == "NO" and not actual_yes)
            contracts = int(t["contracts"])
            stake = float(t["stake_usd"])
            pnl = (contracts * 1.0 - stake) if won else -stake

            t["status"] = "won" if won else "lost"
            t["settled_ts"] = now.isoformat()
            t["actual_close"] = f"{close:.2f}"
            t["pnl"] = f"{pnl:.2f}"
            settled += 1

        if settled:
            self._save()
        logger.info("Paper account: settled %d positions.", settled)
        return settled

    # ------------------------------------------------------------------
    def open_positions(self, qualifying: List[Dict[str, Any]], cfg) -> int:
        """Open new paper positions for qualifying opportunities. Returns count opened."""
        now = datetime.now(timezone.utc)
        held = {t["market_id"] for t in self.trades if t["status"] == "open"}
        available = self.available_cash
        opened = 0

        ranked = sorted(qualifying, key=lambda a: a.get("abs_edge", 0), reverse=True)
        for a in ranked:
            if opened >= MAX_NEW_POSITIONS_PER_CYCLE or available < 0.01:
                break
            market_id = a["market_id"]
            if market_id in held:
                continue
            expiry_dt = a.get("expiry_dt")
            if expiry_dt is None:
                continue

            side = a["direction"]
            if side == "YES":
                entry_cents = int(a.get("yes_ask", 100) or 100)
                win_prob = a["model_prob"]
            else:
                entry_cents = int(a.get("no_ask", 100) or 100)
                win_prob = 1.0 - a["model_prob"]
            if entry_cents < 1 or entry_cents > 99:
                continue

            cost = entry_cents / 100.0
            frac = _kelly_fraction_of_bankroll(
                win_prob, cost, cfg.KELLY_FRACTION, cfg.MAX_BET_PCT
            )
            if frac <= 0:
                continue

            target_stake = min(self.equity * frac, available)
            contracts = int(math.floor(target_stake / cost))
            if contracts < 1:
                continue
            actual_stake = contracts * cost
            if actual_stake > available:
                continue

            self.trades.append({
                "trade_id": str(uuid.uuid4()),
                "opened_ts": now.isoformat(),
                "market_id": market_id,
                "strike_type": a.get("strike_type", "greater"),
                "floor_strike": a.get("floor_strike") if a.get("floor_strike") is not None else "",
                "cap_strike": a.get("cap_strike") if a.get("cap_strike") is not None else "",
                "expiry_iso": expiry_dt.isoformat(),
                "side": side,
                "contracts": contracts,
                "entry_price_cents": entry_cents,
                "stake_usd": f"{actual_stake:.2f}",
                "model_prob": f"{a['model_prob']:.4f}",
                "win_prob": f"{win_prob:.4f}",
                "status": "open",
                "settled_ts": "",
                "actual_close": "",
                "pnl": "",
            })
            held.add(market_id)
            available -= actual_stake
            opened += 1

        if opened:
            self._save()
        logger.info("Paper account: opened %d new positions.", opened)
        return opened

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        closed = [t for t in self.trades if t["status"] in ("won", "lost")]
        n = len(closed)
        wins = sum(1 for t in closed if t["status"] == "won")
        staked = sum(float(t["stake_usd"]) for t in closed)
        return {
            "starting_bankroll": self.starting_bankroll,
            "equity": self.equity,
            "available_cash": self.available_cash,
            "open_positions": sum(1 for t in self.trades if t["status"] == "open"),
            "open_stake": self.open_stake,
            "closed_trades": n,
            "win_rate": (wins / n) if n else 0.0,
            "realized_pnl": self.realized_pnl,
            "roi": (self.realized_pnl / staked) if staked else 0.0,
        }


def paper_trade_cycle(qualifying: List[Dict[str, Any]], cfg) -> Dict[str, Any]:
    """
    Run one full paper-trading cycle: settle matured positions, open new ones.

    Guarded by an exclusive file lock so the always-on daemon and the daily
    agent can't corrupt the account CSV if they run at the same time.
    """
    import fcntl

    lock_path = Path(cfg.PAPER_TRADES_CSV).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning("Another paper cycle holds the lock; skipping this run.")
            return PaperAccount(cfg.PAPER_TRADES_CSV, cfg.PAPER_STARTING_BANKROLL).summary()

        acct = PaperAccount(cfg.PAPER_TRADES_CSV, cfg.PAPER_STARTING_BANKROLL)
        acct.settle()
        acct.open_positions(qualifying, cfg)
        return acct.summary()

"""
prediction_log.py — Unbiased labeled dataset for model calibration/learning.

Records the model's predicted probability for EVERY liquid next-hour market
(not just the ones we trade), then settles each against the actual BTC outcome.
This complete, unbiased record of (model_prob -> realized outcome) is what lets
the strategy become better-calibrated over time. Deduped by market ticker so a
market is logged once no matter how often the cycle runs.

Shared by scripts/paper_cycle.py (every cycle) and scripts/daily_backtest.py.
"""

from __future__ import annotations

import csv
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

PRED_FIELDS = [
    "pred_id", "snapshot_ts", "market_id", "strike_type",
    "floor_strike", "cap_strike", "expiry_iso", "direction",
    "model_prob", "win_prob", "cost", "edge", "kelly_bet_usd",
]
SETTLE_FIELDS = [
    "pred_id", "settled_ts", "actual_close", "actual_yes", "won",
    "win_prob", "cost", "staked", "pnl",
]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def record_predictions(alerts: List[Dict[str, Any]], cfg) -> int:
    """Log predictions for all scored markets not already recorded. Returns count added."""
    existing = {r["market_id"] for r in _read_csv(cfg.PREDICTIONS_CSV)}
    now = datetime.now(timezone.utc)

    rows: List[Dict[str, Any]] = []
    for a in alerts:
        market_id = a["market_id"]
        if market_id in existing:
            continue
        expiry_dt = a.get("expiry_dt")
        if expiry_dt is None:
            continue
        direction = a["direction"]
        if direction == "YES":
            cost = (a.get("yes_ask", 100) or 100) / 100.0
            win_prob = a["model_prob"]
        else:
            cost = (a.get("no_ask", 100) or 100) / 100.0
            win_prob = 1.0 - a["model_prob"]
        if cost <= 0 or cost >= 1:
            continue

        existing.add(market_id)
        rows.append({
            "pred_id": str(uuid.uuid4()),
            "snapshot_ts": now.isoformat(),
            "market_id": market_id,
            "strike_type": a.get("strike_type", "greater"),
            "floor_strike": a.get("floor_strike") if a.get("floor_strike") is not None else "",
            "cap_strike": a.get("cap_strike") if a.get("cap_strike") is not None else "",
            "expiry_iso": expiry_dt.isoformat(),
            "direction": direction,
            "model_prob": f"{a['model_prob']:.4f}",
            "win_prob": f"{win_prob:.4f}",
            "cost": f"{cost:.4f}",
            "edge": f"{a['edge']:.4f}",
            "kelly_bet_usd": f"{a['kelly_bet_usd']:.2f}",
        })

    _append_csv(cfg.PREDICTIONS_CSV, PRED_FIELDS, rows)
    logger.info("Recorded %d new predictions.", len(rows))
    return len(rows)


def settle_predictions(cfg) -> int:
    """Settle matured, unsettled predictions against the real BTC close. Returns count."""
    from data.binance import get_close_at

    preds = _read_csv(cfg.PREDICTIONS_CSV)
    settled_ids = {r["pred_id"] for r in _read_csv(cfg.SETTLEMENTS_CSV)}
    now = datetime.now(timezone.utc)

    new: List[Dict[str, Any]] = []
    for p in preds:
        if p["pred_id"] in settled_ids:
            continue
        try:
            expiry = datetime.fromisoformat(p["expiry_iso"])
        except (ValueError, KeyError):
            continue
        if expiry > now:
            continue

        close = get_close_at(int(expiry.timestamp() * 1000))
        if close is None:
            continue

        floor = float(p["floor_strike"]) if p.get("floor_strike") else None
        cap = float(p["cap_strike"]) if p.get("cap_strike") else None
        st = p.get("strike_type", "greater")
        if st == "greater":
            actual_yes = close > (floor if floor is not None else 0)
        elif st == "less":
            actual_yes = close < (cap if cap is not None else float("inf"))
        elif st == "between" and floor is not None and cap is not None:
            actual_yes = floor <= close <= cap
        else:
            continue

        direction = p["direction"]
        won = (direction == "YES" and actual_yes) or (direction == "NO" and not actual_yes)
        cost = float(p["cost"])
        staked = float(p["kelly_bet_usd"])
        pnl = staked * ((1.0 - cost) / cost) if won else -staked

        new.append({
            "pred_id": p["pred_id"],
            "settled_ts": now.isoformat(),
            "actual_close": f"{close:.2f}",
            "actual_yes": int(actual_yes),
            "won": int(won),
            "win_prob": p["win_prob"],
            "cost": f"{cost:.4f}",
            "staked": f"{staked:.2f}",
            "pnl": f"{pnl:.2f}",
        })

    _append_csv(cfg.SETTLEMENTS_CSV, SETTLE_FIELDS, new)
    logger.info("Settled %d matured predictions.", len(new))
    return len(new)


def calibration_summary(cfg) -> Dict[str, Any]:
    """Model-quality metrics over all settled predictions."""
    settlements = _read_csv(cfg.SETTLEMENTS_CSV)
    preds = _read_csv(cfg.PREDICTIONS_CSV)
    n = len(settlements)
    if not n:
        return {
            "settled": 0, "open": len(preds), "hit_rate": 0.0,
            "mean_pred": 0.0, "calibration_gap": 0.0, "brier": 0.0,
        }
    wins = sum(int(s["won"]) for s in settlements)
    mean_pred = sum(float(s["win_prob"]) for s in settlements) / n
    hit_rate = wins / n
    brier = sum((float(s["win_prob"]) - int(s["won"])) ** 2 for s in settlements) / n
    return {
        "settled": n,
        "open": len(preds) - n,
        "hit_rate": hit_rate,
        "mean_pred": mean_pred,
        "calibration_gap": mean_pred - hit_rate,
        "brier": brier,
    }

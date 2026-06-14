#!/usr/bin/env python3
"""
scripts/daily_backtest.py — Daily strategy maintenance + real-outcome tracking.

Designed to run once a day (locally via cron or via the scheduled cloud agent).
Each run does four things:

  1. SETTLE   — for every past prediction whose market has expired, fetch the
                actual BTC outcome and record whether our call won and its P&L.
                This is the only source of *real* edge measurement.
  2. RETRAIN  — retrain the XGBoost model on the latest BTC history.
  3. SNAPSHOT — run a live scan and log today's qualifying predictions (with the
                real Kalshi prices we'd pay) so they can be settled later.
  4. REPORT   — write a dated markdown report summarizing accumulated real
                performance and a pipeline-sanity backtest.

The CSVs in tracking/ are committed to git so data compounds across runs even
when the agent starts from a fresh clone each day.
"""

from __future__ import annotations

import csv
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_backtest")

TRACKING_DIR = PROJECT_ROOT / "tracking"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_CSV = TRACKING_DIR / "predictions.csv"
SETTLEMENTS_CSV = TRACKING_DIR / "settlements.csv"

PRED_FIELDS = [
    "pred_id", "snapshot_ts", "market_id", "strike_type",
    "floor_strike", "cap_strike", "expiry_iso", "direction",
    "model_prob", "win_prob", "cost", "edge", "kelly_bet_usd",
]
SETTLE_FIELDS = [
    "pred_id", "settled_ts", "actual_close", "actual_yes", "won",
    "win_prob", "cost", "staked", "pnl",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 1. Settle matured predictions
# ---------------------------------------------------------------------------

def settle_predictions() -> int:
    """Settle any unsettled, expired predictions against the real BTC outcome."""
    from data.binance import get_close_at

    preds = _read_csv(PREDICTIONS_CSV)
    settled_ids = {r["pred_id"] for r in _read_csv(SETTLEMENTS_CSV)}
    now = datetime.now(timezone.utc)

    new_settlements: list[dict] = []
    for p in preds:
        if p["pred_id"] in settled_ids:
            continue
        try:
            expiry = datetime.fromisoformat(p["expiry_iso"])
        except (ValueError, KeyError):
            continue
        if expiry > now:
            continue  # not matured yet

        close = get_close_at(int(expiry.timestamp() * 1000))
        if close is None:
            continue  # price not available yet; retry next run

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

        new_settlements.append({
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

    _append_csv(SETTLEMENTS_CSV, SETTLE_FIELDS, new_settlements)
    logger.info("Settled %d matured predictions.", len(new_settlements))
    return len(new_settlements)


# ---------------------------------------------------------------------------
# 2. Retrain model
# ---------------------------------------------------------------------------

def retrain_model() -> dict:
    from backtest.simulate import run_full_backtest
    from backtest.evaluate import run_backtest, train_model_from_history
    from models.lognormal import prob_above_strike

    df = run_full_backtest(limit=500 * 24)
    if df.empty:
        logger.warning("No history for retraining.")
        return {}

    df["model_prob"] = df.apply(
        lambda r: prob_above_strike(
            r["current_price"], r["strike"], r["expiry_hours"],
            r.get("iv_annualized", 0.65),
        ),
        axis=1,
    )
    df["edge"] = df["model_prob"] - df["kalshi_implied_prob"]

    results = run_backtest(
        df, edge_threshold=cfg.EDGE_THRESHOLD, bankroll=cfg.BANKROLL,
        kelly_fraction=cfg.KELLY_FRACTION, max_bet_pct=cfg.MAX_BET_PCT,
    )
    try:
        train_model_from_history(df)
        logger.info("Model retrained and saved.")
    except Exception as exc:
        logger.warning("Retrain failed: %s", exc)
    return results


# ---------------------------------------------------------------------------
# 3. Snapshot today's predictions
# ---------------------------------------------------------------------------

def snapshot_predictions() -> int:
    from kalshi.client import KalshiClient
    from alerts.engine import scan_markets

    client = KalshiClient(
        api_key_id=cfg.KALSHI_API_KEY_ID or None,
        private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
        demo=cfg.KALSHI_DEMO,
    )
    qualifying = scan_markets(client, cfg)
    now = datetime.now(timezone.utc)

    rows: list[dict] = []
    for a in qualifying:
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

        rows.append({
            "pred_id": str(uuid.uuid4()),
            "snapshot_ts": now.isoformat(),
            "market_id": a["market_id"],
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

    _append_csv(PREDICTIONS_CSV, PRED_FIELDS, rows)
    logger.info("Snapshotted %d new predictions.", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

def write_report(backtest_results: dict, n_settled: int, n_new: int) -> Path:
    settlements = _read_csv(SETTLEMENTS_CSV)
    preds = _read_csv(PREDICTIONS_CSV)
    now = datetime.now(timezone.utc)

    n = len(settlements)
    if n:
        wins = sum(int(s["won"]) for s in settlements)
        total_pnl = sum(float(s["pnl"]) for s in settlements)
        total_staked = sum(float(s["staked"]) for s in settlements)
        hit_rate = wins / n
        roi = total_pnl / total_staked if total_staked else 0.0
        mean_pred = sum(float(s["win_prob"]) for s in settlements) / n
        calibration_gap = mean_pred - hit_rate
    else:
        wins = total_pnl = total_staked = hit_rate = roi = mean_pred = calibration_gap = 0.0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{now:%Y-%m-%d}.md"
    bt = backtest_results or {}
    lines = [
        f"# Daily report — {now:%Y-%m-%d %H:%M UTC}",
        "",
        "## Real performance (settled predictions)",
        f"- Total settled bets: **{n}**",
        f"- Open predictions awaiting settlement: **{len(preds) - n}**",
        f"- Newly settled this run: {n_settled}",
        f"- New predictions logged this run: {n_new}",
        f"- Hit rate: **{hit_rate:.1%}**" if n else "- Hit rate: n/a (no settled bets yet)",
        f"- Total P&L: **${total_pnl:+,.2f}** on ${total_staked:,.2f} staked (ROI {roi:+.1%})" if n else "",
        f"- Mean predicted win prob: {mean_pred:.1%} | Calibration gap: {calibration_gap:+.1%}" if n else "",
        "",
        "## Pipeline backtest (synthetic prices — sanity only, not real edge)",
        f"- Total bets: {bt.get('total_bets', 'n/a')}",
        f"- Win rate: {bt.get('win_rate', 0):.1%}" if bt else "- n/a",
        f"- ROI: {bt.get('roi', 0):.1%}" if bt else "",
        f"- Sharpe: {bt.get('sharpe_ratio', 0):.2f}" if bt else "",
        "",
        "## Notes for next iteration",
        "- If calibration gap is consistently positive, the model is "
        "over-confident; consider lowering ml_weight or widening EDGE_THRESHOLD.",
        "- If hit rate < cost-implied breakeven across many bets, the edge is "
        "not real; revisit features or the volatility input.",
        "",
    ]
    path.write_text("\n".join(l for l in lines if l is not None))
    logger.info("Wrote report to %s", path)
    return path


# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Daily backtest run starting ===")
    n_settled = settle_predictions()
    bt = retrain_model()
    n_new = snapshot_predictions()
    report = write_report(bt, n_settled, n_new)
    print(f"\nDaily run complete. Report: {report}")
    print(report.read_text())


if __name__ == "__main__":
    main()

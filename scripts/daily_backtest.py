#!/usr/bin/env python3
"""
scripts/daily_backtest.py — Daily strategy maintenance + real-outcome tracking.

Runs once a day (locally or via the scheduled cloud agent). Each run:

  1. SETTLE   — settle matured predictions and paper positions against the
                actual BTC outcome (the only real edge measurement).
  2. RETRAIN  — retrain the XGBoost model on the latest BTC history.
  3. RECORD   — log predictions for ALL liquid next-hour markets (unbiased
                calibration dataset) and run a paper-trading cycle on the edges.
  4. REPORT   — write a dated markdown report of accumulated real performance.

The continuous launchd cycle (scripts/paper_cycle.py) already does the
settle/record/trade steps every 15 minutes; this daily job adds retraining,
the report, and the git commit (via the scheduled agent). All tracking CSVs in
tracking/ are committed so data compounds across runs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
import prediction_log  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("daily_backtest")

REPORTS_DIR = PROJECT_ROOT / "reports"


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

    # Refit the model+market calibrator on the growing real-outcome dataset.
    try:
        from models.calibrator import train_from_history
        if train_from_history(cfg):
            logger.info("Calibrator refit on real outcomes.")
    except Exception as exc:
        logger.warning("Calibrator refit failed: %s", exc)

    return results


def record_and_trade() -> tuple[int, dict]:
    """Scan all liquid markets, log the full dataset, run a paper cycle."""
    from kalshi.client import KalshiClient
    from alerts.engine import scan_markets

    client = KalshiClient(
        api_key_id=cfg.KALSHI_API_KEY_ID or None,
        private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
        demo=cfg.KALSHI_DEMO,
    )
    try:
        all_alerts = scan_markets(client, cfg, full=True)
    except Exception as exc:
        logger.exception("Scan failed: %s", exc)
        all_alerts = []

    n_new = prediction_log.record_predictions(all_alerts, cfg)

    paper_summary: dict = {}
    if getattr(cfg, "PAPER_TRADE", False):
        from paper.account import paper_trade_cycle
        qualifying = [a for a in all_alerts if a.get("abs_edge", 0) >= cfg.EDGE_THRESHOLD]
        paper_summary = paper_trade_cycle(qualifying, cfg)
        logger.info("Paper account: %s", paper_summary)

    return n_new, paper_summary


def write_report(bt: dict, n_settled: int, n_new: int, paper: dict) -> Path:
    now = datetime.now(timezone.utc)
    cal = prediction_log.calibration_summary(cfg)
    p = paper or {}
    bt = bt or {}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{now:%Y-%m-%d}.md"
    lines = [
        f"# Daily report — {now:%Y-%m-%d %H:%M UTC}",
        "",
        "## Paper trading account",
        f"- Equity: **${p.get('equity', 0):,.2f}** "
        f"(started ${p.get('starting_bankroll', 0):,.2f}, "
        f"realized P&L ${p.get('realized_pnl', 0):+,.2f})" if p else "- (paper trading disabled)",
        f"- Open positions: {p.get('open_positions', 0)} "
        f"(${p.get('open_stake', 0):,.2f} staked) | "
        f"Available cash: ${p.get('available_cash', 0):,.2f}" if p else "",
        f"- Closed trades: {p.get('closed_trades', 0)} @ "
        f"{p.get('win_rate', 0):.1%} win rate | ROI {p.get('roi', 0):+.1%}" if p else "",
        "",
        "## Model calibration dataset (all liquid markets)",
        f"- Settled outcomes: **{cal['settled']}** | Awaiting settlement: {cal['open']}",
        f"- New predictions logged this run: {n_new} | Newly settled: {n_settled}",
        f"- Hit rate: **{cal['hit_rate']:.1%}**" if cal["settled"] else "- Hit rate: n/a yet",
        f"- Mean predicted: {cal['mean_pred']:.1%} | Calibration gap: "
        f"{cal['calibration_gap']:+.1%} | Brier: {cal['brier']:.3f}" if cal["settled"] else "",
        "",
        "## Pipeline backtest (synthetic prices — sanity only, not real edge)",
        f"- Total bets: {bt.get('total_bets', 'n/a')}",
        f"- Win rate: {bt.get('win_rate', 0):.1%}" if bt else "- n/a",
        f"- ROI: {bt.get('roi', 0):.1%}" if bt else "",
        "",
        "## Notes for next iteration",
        "- Brier score is the headline model-quality metric (lower = better; "
        "0.25 = no skill). Track it across days.",
        "- Positive calibration gap = model over-confident; consider the vol "
        "input or lowering ML_WEIGHT.",
        "- Only change strategy once there are ~30+ settled outcomes.",
        "",
    ]
    path.write_text("\n".join(l for l in lines if l is not None))
    logger.info("Wrote report to %s", path)
    return path


def main() -> None:
    logger.info("=== Daily backtest run starting ===")
    n_settled = prediction_log.settle_predictions(cfg)
    bt = retrain_model()
    n_new, paper = record_and_trade()
    report = write_report(bt, n_settled, n_new, paper)
    print(f"\nDaily run complete. Report: {report}")
    print(report.read_text())


if __name__ == "__main__":
    main()

"""
main.py — Entry point for the Kalshi BTC scanner bot.

Usage:
    python main.py                  # Run immediately, then hourly
    python main.py --backtest-only  # Run backtest + train model, then exit
    python main.py --once           # Run one scan then exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import schedule

import config
from alerts.engine import scan_markets
from backtest.evaluate import (
    print_backtest_report,
    run_backtest,
    train_model_from_history,
)
from backtest.simulate import run_full_backtest
from features.engineer import FEATURE_NAMES
from kalshi.client import KalshiClient
from models.ml_model import MLModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_model() -> MLModel:
    """Load existing model or train a new one from historical data."""
    model = MLModel(pkl_path=config.MODEL_PKL_PATH)

    if model.load():
        logger.info("Loaded existing ML model.")
        return model

    logger.info("No saved model found — running backtest + training now...")
    try:
        history = run_full_backtest(limit=1000)

        if history.empty:
            logger.warning("Backtest returned no data; will use lognormal-only mode.")
            return model

        # Compute lognormal probs for backtest evaluation
        from models.lognormal import prob_above_strike

        ln_probs = history.apply(
            lambda r: prob_above_strike(
                r["current_price"], r["strike"], r["expiry_hours"], 0.65
            ),
            axis=1,
        )

        metrics = run_backtest(history, ln_probs)
        logger.info("Pre-training backtest metrics (lognormal-only):")
        print_backtest_report(metrics)

        # Train XGBoost
        model = train_model_from_history(history)

        # Re-evaluate with trained model
        if model.is_trained:
            ml_probs = history.apply(
                lambda r: model.predict_proba(r.to_dict()), axis=1
            )
            from models.ensemble import predict as ensemble_predict

            ens_probs = [
                ensemble_predict(ln, ml, ml_trained=True)
                for ln, ml in zip(ln_probs, ml_probs)
            ]
            import pandas as pd

            metrics_after = run_backtest(history, pd.Series(ens_probs))
            logger.info("Post-training backtest metrics (ensemble):")
            print_backtest_report(metrics_after)

    except Exception as exc:
        logger.error("Backtest/training failed: %s", exc)
        logger.warning("Falling back to lognormal-only mode.")

    return model


def run_scan(client: KalshiClient, model: MLModel) -> None:
    logger.info("Running market scan...")
    try:
        alerts = scan_markets(client, ml_model=model)
        if alerts:
            logger.info("Found %d opportunities above edge threshold.", len(alerts))
        else:
            logger.info("No opportunities above threshold this scan.")
    except Exception as exc:
        logger.error("Scan failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi BTC Scanner Bot")
    parser.add_argument("--backtest-only", action="store_true", help="Run backtest and exit")
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    args = parser.parse_args()

    # Validate config
    if not config.KALSHI_EMAIL or not config.KALSHI_PASSWORD:
        print(
            "\nERROR: KALSHI_EMAIL and KALSHI_PASSWORD must be set in your .env file.\n"
            "Copy .env.example to .env and fill in your credentials.\n"
        )
        sys.exit(1)

    # Login to Kalshi
    client = KalshiClient()
    try:
        client.login(config.KALSHI_EMAIL, config.KALSHI_PASSWORD)
    except Exception as exc:
        print(f"\nERROR: Kalshi login failed: {exc}\n")
        sys.exit(1)

    # Backtest-only mode
    if args.backtest_only:
        logger.info("Running backtest-only mode...")
        history = run_full_backtest(limit=1000)
        if history.empty:
            print("No backtest data generated.")
            sys.exit(1)
        from models.lognormal import prob_above_strike
        import pandas as pd

        ln_probs = history.apply(
            lambda r: prob_above_strike(r["current_price"], r["strike"], r["expiry_hours"], 0.65),
            axis=1,
        )
        metrics = run_backtest(history, ln_probs)
        print_backtest_report(metrics)
        sys.exit(0)

    # Setup model (load or train)
    model = setup_model()

    # Single scan mode
    if args.once:
        run_scan(client, model)
        sys.exit(0)

    # Continuous hourly mode
    logger.info("Starting hourly scan loop. Press Ctrl+C to stop.")
    run_scan(client, model)  # Run immediately on start

    schedule.every(1).hours.do(run_scan, client=client, model=model)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
main.py — Entry point for the Kalshi BTC trading bot.

Usage:
    python main.py                 # Full mode: backtest + train + live scan loop
    python main.py --backtest-only # Backtest/train only, then exit
    python main.py --no-backtest   # Skip backtest, go straight to live scanning
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import schedule
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# ---------------------------------------------------------------------------
# Bootstrap logging before any local imports so all modules inherit config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
logger = logging.getLogger("main")
console = Console()

# Ensure logs dir exists before FileHandler tries to write
Path("logs").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Local imports (after logging setup)
# ---------------------------------------------------------------------------
import config
from kalshi.client import KalshiClient
from alerts.engine import scan_markets
from backtest.simulate import run_full_backtest
from backtest.evaluate import run_backtest, train_model_from_history, print_backtest_report
from models.ml_model import MLModel
from models.ensemble import EnsembleModel
from features.engineer import FEATURE_NAMES


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalshi BTC Binary Market Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run backtest and model training, then exit without live scanning.",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip backtest entirely, jump straight to live scanning.",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        metavar="MINUTES",
        help="How often to run a live market scan (default: 60 minutes).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=1000,
        metavar="CANDLES",
        help="Number of 1h historical candles to download for backtest (default: 1000).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backtest + model training
# ---------------------------------------------------------------------------

def run_backtest_pipeline(history_limit: int) -> MLModel:
    """
    Download historical data, simulate markets, train model, print results.
    Returns the trained MLModel instance (or an untrained one on failure).
    """
    console.rule("[bold cyan]Backtest Pipeline[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading historical BTC data...", total=None)
        try:
            history_df = run_full_backtest(limit=history_limit)
            progress.update(task, description=f"Downloaded {len(history_df):,} simulation rows.")
        except Exception as exc:
            logger.error("Backtest data download failed: %s", exc)
            console.print(f"[red]Backtest failed:[/red] {exc}")
            return MLModel(pkl_path=config.MODEL_PKL_PATH)

    if history_df.empty:
        console.print("[yellow]Warning:[/yellow] No historical data generated. Skipping backtest.")
        return MLModel(pkl_path=config.MODEL_PKL_PATH)

    console.print(
        Panel(
            f"Simulated [bold]{len(history_df):,}[/bold] historical market observations\n"
            f"Columns: {', '.join(history_df.columns[:8])}...",
            title="Simulation Complete",
            border_style="blue",
        )
    )

    # Train model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Training XGBoost model...", total=None)
        try:
            model = train_model_from_history(history_df)
            progress.update(task, description="Model training complete.")
        except Exception as exc:
            logger.error("Model training failed: %s", exc)
            console.print(f"[red]Training failed:[/red] {exc}")
            model = MLModel(pkl_path=config.MODEL_PKL_PATH)

    # Compute backtest metrics using model probabilities
    if model.is_trained:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Evaluating backtest performance...", total=None)
            try:
                import pandas as pd
                feature_cols = [c for c in FEATURE_NAMES if c in history_df.columns]
                model_probs = pd.Series(
                    [model.predict_proba(row.to_dict()) for _, row in history_df[feature_cols].iterrows()],
                    index=history_df.index,
                )
                metrics = run_backtest(
                    history_df=history_df,
                    model_probs=model_probs,
                    edge_threshold=config.EDGE_THRESHOLD,
                )
                progress.update(task, description="Evaluation complete.")
                print_backtest_report(metrics)
            except Exception as exc:
                logger.error("Backtest evaluation failed: %s", exc)
                console.print(f"[yellow]Backtest evaluation error:[/yellow] {exc}")
    else:
        console.print("[yellow]Model not trained (not enough data). Skipping performance evaluation.[/yellow]")

    return model


# ---------------------------------------------------------------------------
# Live scanning
# ---------------------------------------------------------------------------

def run_live_scan(client: KalshiClient, ml_model: MLModel) -> None:
    """Execute a single market scan and handle any errors."""
    console.rule("[bold green]Market Scan[/bold green]")
    try:
        alerts = scan_markets(client=client, ml_model=ml_model)
        if alerts:
            console.print(f"[green]Found {len(alerts)} opportunity/ies above edge threshold.[/green]")

            # Auto-trade if enabled
            if config.AUTO_TRADE:
                _execute_trades(client, alerts)
        else:
            console.print("[dim]No opportunities found above threshold this scan.[/dim]")
    except Exception as exc:
        logger.error("Live scan failed: %s", exc, exc_info=True)
        console.print(f"[red]Scan error:[/red] {exc}")


def _execute_trades(client: KalshiClient, alerts: list) -> None:
    """Place orders for all alerts when AUTO_TRADE=true."""
    import csv
    from pathlib import Path
    from datetime import datetime, timezone

    trades_log = config.TRADES_CSV
    trades_log.parent.mkdir(parents=True, exist_ok=True)

    for alert in alerts:
        market_id = alert.get("market_id", "")
        side = alert.get("side", "YES").lower()
        kelly_usd = alert.get("kelly_bet_usd", 0.0)
        ensemble_prob = alert.get("ensemble_prob", 0.5)

        if kelly_usd < 1.0:
            logger.debug("Skipping %s: Kelly bet $%.2f < $1 minimum.", market_id, kelly_usd)
            continue

        # Convert probability to Kalshi cent price
        price_cents = int(round(ensemble_prob * 100))
        price_cents = max(1, min(99, price_cents))

        # Number of contracts = dollar amount (each contract = $1 max payout)
        contracts = max(1, int(kelly_usd))

        try:
            result = client.place_order(
                market_id=market_id,
                side=side,
                contracts=contracts,
                price=price_cents,
            )
            logger.info(
                "Placed order: %s %s x%d @ %d¢ — result: %s",
                side.upper(), market_id, contracts, price_cents, result,
            )

            # Log to trades CSV
            trade_record = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "market_id": market_id,
                "side": side.upper(),
                "contracts": contracts,
                "price_cents": price_cents,
                "kelly_usd": round(kelly_usd, 2),
                "ensemble_prob": round(ensemble_prob, 4),
                "edge": round(alert.get("edge", 0), 4),
                "order_result": str(result.get("order", {}).get("status", "unknown")),
            }
            file_exists = trades_log.exists()
            with open(trades_log, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(trade_record.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade_record)

        except Exception as exc:
            logger.error("Failed to place order for %s: %s", market_id, exc)
            console.print(f"[red]Order failed for {market_id}:[/red] {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print startup banner
    console.print(
        Panel.fit(
            "[bold cyan]Kalshi BTC Binary Market Trading Bot[/bold cyan]\n"
            f"Bankroll: [green]${config.BANKROLL:,.0f}[/green]  "
            f"Kelly: [yellow]{config.KELLY_FRACTION:.0%}[/yellow]  "
            f"Max bet: [yellow]{config.MAX_BET_PCT:.0%}[/yellow]  "
            f"Edge threshold: [yellow]{config.EDGE_THRESHOLD:.0%}[/yellow]  "
            f"Auto-trade: [{'green' if config.AUTO_TRADE else 'red'}]{config.AUTO_TRADE}[/]",
            box=box.DOUBLE_EDGE,
            border_style="cyan",
        )
    )

    # ---- Step 1: Model training / loading ----
    ml_model: MLModel

    if args.no_backtest:
        # Try loading existing model
        ml_model = MLModel(pkl_path=config.MODEL_PKL_PATH)
        if ml_model.load():
            console.print("[green]Loaded existing model from disk.[/green]")
        else:
            console.print(
                "[yellow]No saved model found. Running in lognormal-only mode.[/yellow]"
            )
    else:
        need_train = not config.MODEL_PKL_PATH.exists()
        if need_train:
            console.print("[bold]No saved model found — running initial backtest + training...[/bold]")
            ml_model = run_backtest_pipeline(args.history_limit)
        else:
            if args.backtest_only:
                # Force retrain even if model exists
                console.print("[bold]--backtest-only: retraining model...[/bold]")
                ml_model = run_backtest_pipeline(args.history_limit)
            else:
                # Load existing model, skip backtest
                ml_model = MLModel(pkl_path=config.MODEL_PKL_PATH)
                if ml_model.load():
                    console.print("[green]Loaded existing model from disk. Skipping backtest.[/green]")
                else:
                    console.print("[yellow]Failed to load model, retraining...[/yellow]")
                    ml_model = run_backtest_pipeline(args.history_limit)

    # Exit here if backtest-only mode
    if args.backtest_only:
        console.print("[bold green]Backtest complete. Exiting.[/bold green]")
        return

    # ---- Step 2: Kalshi authentication ----
    if not config.KALSHI_EMAIL or not config.KALSHI_PASSWORD:
        console.print(
            "[red]ERROR:[/red] KALSHI_EMAIL and KALSHI_PASSWORD must be set in .env\n"
            "Copy [bold].env.example[/bold] → [bold].env[/bold] and fill in your credentials."
        )
        sys.exit(1)

    client = KalshiClient()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Logging into Kalshi...", total=None)
        try:
            client.login(config.KALSHI_EMAIL, config.KALSHI_PASSWORD)
            progress.update(task, description="Kalshi login successful.")
            console.print("[green]Logged into Kalshi.[/green]")
        except Exception as exc:
            logger.error("Kalshi login failed: %s", exc)
            console.print(f"[red]Kalshi login failed:[/red] {exc}")
            sys.exit(1)

    # ---- Step 3: Immediate first scan ----
    run_live_scan(client, ml_model)

    # ---- Step 4: Scheduled hourly scans ----
    interval = args.scan_interval
    console.print(
        f"\n[bold]Scheduling market scans every [cyan]{interval}[/cyan] minute(s).[/bold]\n"
        "Press [bold]Ctrl+C[/bold] to stop.\n"
    )

    schedule.every(interval).minutes.do(run_live_scan, client=client, ml_model=ml_model)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Shutting down.[/yellow]")


if __name__ == "__main__":
    main()

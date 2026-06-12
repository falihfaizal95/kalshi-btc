#!/usr/bin/env python3
"""
main.py — Entry point for the Kalshi BTC trading bot.

Usage:
    python main.py                  # Run live scanning (trains model if needed)
    python main.py --backtest-only  # Run backtest + train model, then exit
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

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path so relative imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402  (import after path fixup)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOGS_DIR / "bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_and_backtest() -> None:
    """Download history, run backtest, and train/save the XGBoost model."""
    from backtest.simulate import run_full_backtest
    from backtest.evaluate import run_backtest, train_model_from_history
    from models.lognormal import prob_above_strike

    console.print(
        Panel(
            "[bold cyan]Running backtest on 500 days of historical data...[/bold cyan]",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading historical BTC data...", total=None)
        history_df = run_full_backtest(limit=500 * 24)  # ~500 days at 1h granularity

    if history_df.empty:
        console.print("[red]Backtest data is empty. Skipping model training.[/red]")
        return

    # Compute model_prob column using lognormal (cold-start — no ML model yet for training data)
    console.print("[cyan]Computing log-normal probabilities for backtest rows...[/cyan]")
    history_df["model_prob"] = history_df.apply(
        lambda r: prob_above_strike(
            current_price=r["current_price"],
            strike=r["strike"],
            time_hours=r["expiry_hours"],
            annual_vol=r.get("iv_annualized", 0.65),
        ),
        axis=1,
    )
    history_df["edge"] = history_df["model_prob"] - history_df["kalshi_implied_prob"]

    # Run backtest evaluation
    console.print("[cyan]Evaluating backtest performance...[/cyan]")
    results = run_backtest(
        history_df=history_df,
        edge_threshold=cfg.EDGE_THRESHOLD,
        bankroll=cfg.BANKROLL,
        kelly_fraction=cfg.KELLY_FRACTION,
        max_bet_pct=cfg.MAX_BET_PCT,
    )
    logger.info("Backtest results: %s", results)
    console.print(
        "[dim]Note: backtest prices are simulated (lognormal + noise), so P&L "
        "validates the pipeline, not real market edge.[/dim]"
    )

    # Train XGBoost model
    console.print("[cyan]Training XGBoost model on historical features...[/cyan]")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Training XGBoost...", total=None)
            model = train_model_from_history(history_df)
        console.print(
            f"[green]Model trained and saved to {cfg.MODEL_PKL_PATH}[/green]"
        )
    except Exception as exc:
        console.print(f"[yellow]Model training failed: {exc}. Will use lognormal only.[/yellow]")
        logger.warning("Model training failed: %s", exc)


def _run_scan(client) -> None:
    """Run one market scan cycle."""
    try:
        from alerts.engine import scan_markets
        qualifying = scan_markets(client, cfg)
        logger.info("Scan complete — %d qualifying opportunities.", len(qualifying))
    except Exception as exc:
        console.print(f"[red]Scan error: {exc}[/red]")
        logger.exception("Scan error: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalshi BTC trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Run backtest and model training, then exit without live scanning.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training / backtest on startup.",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold green]Kalshi BTC Trading Bot[/bold green]\n"
            f"[dim]Bankroll: ${cfg.BANKROLL:,.2f} | "
            f"Kelly: {cfg.KELLY_FRACTION:.0%} | "
            f"Max bet: {cfg.MAX_BET_PCT:.0%} | "
            f"Edge threshold: {cfg.EDGE_THRESHOLD:.0%} | "
            f"Auto-trade: {'ON' if cfg.AUTO_TRADE else 'OFF'}[/dim]",
            border_style="green",
        )
    )

    # ------------------------------------------------------------------
    # Set up Kalshi client (market data is public; keys only needed to trade)
    # ------------------------------------------------------------------
    from kalshi.client import KalshiClient

    try:
        client = KalshiClient(
            api_key_id=cfg.KALSHI_API_KEY_ID or None,
            private_key_path=cfg.KALSHI_PRIVATE_KEY_PATH or None,
            demo=cfg.KALSHI_DEMO,
        )
    except Exception as exc:
        console.print(f"[red]Failed to initialize Kalshi client: {exc}[/red]")
        logger.error("Kalshi client init failed: %s", exc)
        sys.exit(1)

    if client.can_trade:
        try:
            balance = client.get_balance()
            cents = balance.get("balance", 0)
            console.print(
                f"[green]Kalshi API key verified. Balance: ${cents / 100:,.2f}[/green]"
            )
        except Exception as exc:
            console.print(f"[red]Kalshi API key check failed: {exc}[/red]")
            logger.error("Kalshi auth check failed: %s", exc)
            sys.exit(1)
    else:
        if cfg.AUTO_TRADE:
            console.print(
                "[red]ERROR: AUTO_TRADE=true requires KALSHI_API_KEY_ID and "
                "KALSHI_PRIVATE_KEY_PATH in .env[/red]"
            )
            sys.exit(1)
        console.print(
            "[yellow]No Kalshi API credentials — running in read-only scan mode.[/yellow]"
        )

    # ------------------------------------------------------------------
    # Backtest / model training
    # ------------------------------------------------------------------
    needs_training = not cfg.MODEL_PKL_PATH.exists()

    if args.backtest_only or (needs_training and not args.no_train):
        _train_and_backtest()
    elif not args.no_train and not needs_training:
        console.print(
            f"[dim]Model already exists at {cfg.MODEL_PKL_PATH} — skipping training.[/dim]"
        )

    if args.backtest_only:
        console.print("[bold green]Backtest-only mode complete. Exiting.[/bold green]")
        return

    # ------------------------------------------------------------------
    # Initial scan
    # ------------------------------------------------------------------
    console.print("\n[bold cyan]Running initial market scan...[/bold cyan]")
    _run_scan(client)

    # ------------------------------------------------------------------
    # Hourly scheduled scans
    # ------------------------------------------------------------------
    console.print("[dim]Scheduling hourly scans. Press Ctrl+C to stop.[/dim]")
    schedule.every(1).hours.do(_run_scan, client=client)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped by user.[/yellow]")
        logger.info("Bot stopped by user.")


if __name__ == "__main__":
    main()

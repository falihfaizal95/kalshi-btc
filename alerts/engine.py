"""
alerts/engine.py — Scan Kalshi BTC markets, compute edge, print ranked alert table.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import config
from data.binance import get_ohlcv
from data.deribit import get_iv
from data.sentiment import get_fear_greed
from features.engineer import build_feature_vector
from kalshi.client import KalshiClient
from kalshi.markets import get_btc_markets
from models.ensemble import predict as ensemble_predict
from models.lognormal import prob_above_strike
from models.ml_model import MLModel

logger = logging.getLogger(__name__)


def compute_edge(market: Dict, ensemble_prob: float) -> float:
    """
    Edge = model_prob - kalshi_implied_prob.
    Positive edge → bet YES. Negative edge → bet NO.
    """
    kalshi_yes_price = market.get("yes_ask", 50.0)
    kalshi_implied = kalshi_yes_price / 100.0
    return ensemble_prob - kalshi_implied


def kelly_bet_size(edge: float, win_prob: float, bankroll: float) -> float:
    if win_prob <= 0 or win_prob >= 1 or edge <= 0:
        return 0.0
    kelly = (edge / (1.0 - win_prob)) * config.KELLY_FRACTION
    return min(kelly * bankroll, bankroll * config.MAX_BET_PCT)


def _log_alert(alert: Dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(alert.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(alert)


def scan_markets(client: KalshiClient, ml_model: Optional[MLModel] = None) -> List[Dict]:
    """
    Full market scan:
    1. Fetch all active Kalshi BTC markets
    2. Fetch current BTC price + 1h OHLCV
    3. Fetch IV and Fear & Greed
    4. Score each market with ensemble model
    5. Return list of alert dicts sorted by |edge|

    Also prints a rich table to terminal and logs to CSV.
    """
    # ---- Fetch market data ----
    try:
        markets = get_btc_markets(client)
        logger.info("Fetched %d BTC markets from Kalshi.", len(markets))
    except Exception as exc:
        logger.error("Failed to fetch Kalshi markets: %s", exc)
        return []

    if not markets:
        logger.warning("No active BTC markets found on Kalshi.")
        return []

    # ---- Fetch price / chart data ----
    try:
        ohlcv = get_ohlcv(interval="1h", limit=200)
        btc_price = float(ohlcv["close"].iloc[-1])
        logger.info("BTC price: $%.2f", btc_price)
    except Exception as exc:
        logger.error("Failed to fetch BTC OHLCV: %s", exc)
        return []

    # ---- External signals ----
    try:
        iv = get_iv()
    except Exception:
        iv = 0.65
        logger.warning("IV fetch failed; using 0.65")

    try:
        fg = get_fear_greed()
        fear_greed_score = fg["score"]
    except Exception:
        fear_greed_score = 50
        logger.warning("Fear & Greed fetch failed; using 50")

    # ---- Score each market ----
    alerts = []
    for market in markets:
        strike = market.get("strike_price")
        expiry = market.get("expiry_time")
        if not strike or not expiry:
            continue

        now = datetime.now(tz=timezone.utc)
        time_to_expiry_h = max(0.0, (expiry - now).total_seconds() / 3600.0)
        if time_to_expiry_h <= 0:
            continue

        try:
            features = build_feature_vector(
                btc_price=btc_price,
                market=market,
                ohlcv_1h=ohlcv,
                iv=iv,
                fear_greed_score=fear_greed_score,
            )
        except Exception as exc:
            logger.debug("Feature build failed for market %s: %s", market.get("market_id"), exc)
            continue

        # Probabilities
        ln_prob = prob_above_strike(btc_price, strike, time_to_expiry_h, iv)
        ml_prob = ml_model.predict_proba(features) if (ml_model and ml_model.is_trained) else 0.5
        ml_trained = ml_model.is_trained if ml_model else False
        ens_prob = ensemble_predict(ln_prob, ml_prob, ml_trained=ml_trained)

        edge = compute_edge(market, ens_prob)

        if abs(edge) < config.EDGE_THRESHOLD:
            continue

        side = "YES" if edge > 0 else "NO"
        win_prob = ens_prob if edge > 0 else 1.0 - ens_prob
        bet_size = kelly_bet_size(abs(edge), win_prob, config.BANKROLL)

        alert = {
            "scanned_at": now.isoformat(),
            "market_id": market.get("market_id", ""),
            "title": market.get("title", ""),
            "expiry": expiry.isoformat(),
            "time_to_expiry_h": round(time_to_expiry_h, 2),
            "strike": strike,
            "btc_price": round(btc_price, 2),
            "kalshi_yes_ask": market.get("yes_ask", 0),
            "kalshi_yes_bid": market.get("yes_bid", 0),
            "lognormal_prob": round(ln_prob, 4),
            "ml_prob": round(ml_prob, 4),
            "ensemble_prob": round(ens_prob, 4),
            "edge": round(edge, 4),
            "side": side,
            "kelly_bet_usd": round(bet_size, 2),
        }
        alerts.append(alert)
        _log_alert(alert, config.ALERTS_CSV)

    # Sort by absolute edge descending
    alerts.sort(key=lambda x: abs(x["edge"]), reverse=True)

    _print_alert_table(alerts, btc_price, fear_greed_score)

    # ---- Auto-trade if enabled ----
    if config.AUTO_TRADE and client is not None:
        for a in alerts:
            if a["kelly_bet_usd"] < 1.0:
                continue
            _place_and_log_trade(client, a)

    return alerts


def _place_and_log_trade(client: KalshiClient, alert: Dict) -> None:
    """Place a Kalshi order for an alert and log to trades.csv."""
    import math

    side = alert["side"].lower()         # "yes" or "no"
    market_id = alert["market_id"]
    bet_usd = alert["kelly_bet_usd"]

    # Each Kalshi contract costs the yes_price in cents (max payout = $1)
    if side == "yes":
        price_cents = alert["kalshi_yes_ask"]
    else:
        price_cents = 100 - alert["kalshi_yes_bid"]  # cost of a NO contract

    if price_cents <= 0:
        logger.warning("Skipping trade for %s: zero price.", market_id)
        return

    # Number of contracts = budget / cost-per-contract
    contracts = max(1, math.floor(bet_usd / (price_cents / 100.0)))

    try:
        result = client.place_order(
            market_id=market_id,
            side=side,
            contracts=contracts,
            price=price_cents,
        )
        trade = {
            "placed_at": alert["scanned_at"],
            "market_id": market_id,
            "side": side,
            "contracts": contracts,
            "price_cents": price_cents,
            "bet_usd": round(contracts * price_cents / 100.0, 2),
            "order_id": result.get("order", {}).get("order_id", ""),
            "edge": alert["edge"],
            "ensemble_prob": alert["ensemble_prob"],
        }
        _log_alert(trade, config.TRADES_CSV)
        logger.info(
            "Placed order: %s %s x%d @ %d¢  order_id=%s",
            side.upper(), market_id, contracts, price_cents,
            trade["order_id"],
        )
    except Exception as exc:
        logger.error("Failed to place order for %s: %s", market_id, exc)


def _print_alert_table(alerts: List[Dict], btc_price: float, fear_greed: int) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text

        console = Console()
        console.rule(f"[bold cyan]Kalshi BTC Scanner[/] — BTC: [yellow]${btc_price:,.2f}[/]  Fear/Greed: [magenta]{fear_greed}[/]")

        if not alerts:
            console.print("[dim]No opportunities above edge threshold.[/dim]")
            return

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold white")
        table.add_column("#", style="dim", width=3)
        table.add_column("Market / Title", style="white", max_width=35)
        table.add_column("Strike", justify="right", style="yellow")
        table.add_column("Expiry (h)", justify="right")
        table.add_column("Kalshi%", justify="right")
        table.add_column("Model%", justify="right")
        table.add_column("Edge%", justify="right")
        table.add_column("Side", justify="center")
        table.add_column("Kelly $", justify="right", style="green")

        for i, a in enumerate(alerts[:20], 1):
            edge_pct = a["edge"] * 100
            edge_color = "green" if edge_pct > 0 else "red"
            side_color = "green" if a["side"] == "YES" else "red"

            table.add_row(
                str(i),
                a["title"][:35] or a["market_id"],
                f"${a['strike']:,.0f}",
                f"{a['time_to_expiry_h']:.1f}h",
                f"{a['kalshi_yes_ask']}¢",
                f"{a['ensemble_prob']*100:.1f}%",
                Text(f"{edge_pct:+.1f}%", style=edge_color),
                Text(a["side"], style=side_color),
                f"${a['kelly_bet_usd']:.2f}",
            )

        console.print(table)
        console.print(f"[dim]Logged to {config.ALERTS_CSV}[/dim]")

    except ImportError:
        print(f"\n--- BTC: ${btc_price:,.2f} | Fear/Greed: {fear_greed} ---")
        if not alerts:
            print("No opportunities above threshold.")
            return
        header = f"{'#':>3}  {'Market':35}  {'Strike':>10}  {'Exp':>5}  {'Kalshi':>7}  {'Model':>7}  {'Edge':>7}  {'Side':>4}  {'Kelly$':>7}"
        print(header)
        print("-" * len(header))
        for i, a in enumerate(alerts[:20], 1):
            print(
                f"{i:>3}  {str(a['title'] or a['market_id'])[:35]:35}  "
                f"${a['strike']:>9,.0f}  {a['time_to_expiry_h']:>4.1f}h  "
                f"{a['kalshi_yes_ask']:>6}¢  {a['ensemble_prob']*100:>6.1f}%  "
                f"{a['edge']*100:>+6.1f}%  {a['side']:>4}  ${a['kelly_bet_usd']:>6.2f}"
            )

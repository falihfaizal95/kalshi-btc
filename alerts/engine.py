"""
alerts/engine.py — Market scanning engine: fetch, score, display, and optionally trade.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_edge(market: Dict[str, Any], ensemble_prob: float) -> float:
    """
    Compute the edge between model probability and Kalshi market implied probability.

    Edge = ensemble_prob - kalshi_yes_mid_price / 100

    The Kalshi yes_ask / yes_bid are in cents (1–99). We use the mid-price.
    Positive edge → bet YES; negative edge → bet NO.

    Parameters
    ----------
    market : dict
        Market dict from get_btc_markets() with keys yes_bid, yes_ask.
    ensemble_prob : float
        Model's probability that BTC is above the strike at expiry.

    Returns
    -------
    float
        Signed edge in probability units (e.g. 0.08 = 8% edge on YES side).
    """
    yes_bid = float(market.get("yes_bid", 0) or 0)
    yes_ask = float(market.get("yes_ask", 100) or 100)
    # Mid-price in [0,1]
    kalshi_yes_mid = (yes_bid + yes_ask) / 2.0 / 100.0
    return float(ensemble_prob - kalshi_yes_mid)


def _kelly_bet_size(
    edge: float,
    model_prob: float,
    bankroll: float,
    kelly_fraction: float,
    max_bet_pct: float,
) -> float:
    """Return Kelly-fraction bet in USD."""
    denom = 1.0 - model_prob
    if denom <= 0 or abs(edge) < 1e-6:
        return 0.0
    raw = kelly_fraction * abs(edge) / denom
    capped = min(raw, max_bet_pct)
    return float(bankroll * capped)


def _log_alert(
    csv_path: Path,
    row: Dict[str, Any],
) -> None:
    """Append one alert row to the CSV log (creates file + header if needed)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp", "market_id", "title", "expiry", "strike",
        "current_price", "kalshi_yes_mid_pct", "model_prob_pct",
        "edge_pct", "kelly_bet_usd", "direction",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _log_trade(
    csv_path: Path,
    market_id: str,
    side: str,
    contracts: int,
    price: float,
    result: Dict[str, Any],
) -> None:
    """Append one trade row to the trades CSV log."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp", "market_id", "side", "contracts", "price_cents",
        "order_id", "status",
    ]
    row = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "market_id": market_id,
        "side": side,
        "contracts": contracts,
        "price_cents": price,
        "order_id": result.get("order", {}).get("order_id", ""),
        "status": result.get("order", {}).get("status", ""),
    }
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def scan_markets(client, cfg) -> List[Dict[str, Any]]:
    """
    Main scanning loop: fetch markets, score them, display results, optionally trade.

    Parameters
    ----------
    client : KalshiClient
        Authenticated Kalshi API client.
    cfg : module
        Loaded config module (config.py) with BANKROLL, KELLY_FRACTION, etc.

    Returns
    -------
    list of alert dicts (ranked by |edge|).
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    from kalshi.markets import get_btc_markets
    from data.binance import get_ohlcv
    from data.deribit import get_iv
    from data.sentiment import get_fear_greed
    from features.engineer import build_feature_vector
    from models.lognormal import prob_above_strike
    from models.ensemble import EnsembleModel

    console = Console()
    now_utc = datetime.now(tz=timezone.utc)

    # ------------------------------------------------------------------
    # 1. Fetch data
    # ------------------------------------------------------------------
    console.print("[bold cyan]Fetching BTC Kalshi markets...[/bold cyan]")
    markets = get_btc_markets(client)
    if not markets:
        console.print("[yellow]No BTC markets found on Kalshi.[/yellow]")
        return []

    console.print(f"[green]Found {len(markets)} BTC markets.[/green]")

    console.print("[bold cyan]Fetching BTC price + OHLCV from Binance...[/bold cyan]")
    try:
        ohlcv_1h = get_ohlcv(symbol="BTCUSDT", interval="1h", limit=200)
        btc_price = float(ohlcv_1h["close"].iloc[-1])
    except Exception as exc:
        console.print(f"[red]Binance fetch failed: {exc}[/red]")
        return []

    console.print(f"[green]BTC Price: ${btc_price:,.2f}[/green]")

    console.print("[bold cyan]Fetching IV from Deribit...[/bold cyan]")
    try:
        iv = get_iv()
        console.print(f"[green]IV: {iv:.1%}[/green]")
    except Exception as exc:
        iv = 0.65
        console.print(f"[yellow]IV fetch failed ({exc}); using default {iv:.0%}[/yellow]")

    console.print("[bold cyan]Fetching Fear & Greed index...[/bold cyan]")
    fg_data = get_fear_greed()
    fear_greed_score = int(fg_data.get("score", 50))
    fg_class = fg_data.get("classification", "Neutral")
    console.print(f"[green]Fear & Greed: {fear_greed_score} ({fg_class})[/green]")

    # ------------------------------------------------------------------
    # 2. Score each market
    # ------------------------------------------------------------------
    ensemble = EnsembleModel(ml_weight=0.6)
    alerts: List[Dict[str, Any]] = []

    for market in markets:
        strike = market.get("strike_price")
        expiry = market.get("expiry_time")

        if strike is None or strike <= 0:
            continue

        # Hours to expiry
        if expiry is not None:
            time_to_expiry_h = max(0.0, (expiry - now_utc).total_seconds() / 3600.0)
        else:
            time_to_expiry_h = 1.0

        if time_to_expiry_h < 0.05:
            continue  # market essentially expired

        # Log-normal probability
        ln_prob = prob_above_strike(btc_price, strike, time_to_expiry_h, iv)

        # Build features
        try:
            features = build_feature_vector(
                btc_price=btc_price,
                market=market,
                ohlcv_1h=ohlcv_1h,
                iv=iv,
                fear_greed_score=fear_greed_score,
            )
        except Exception as exc:
            logger.warning("Feature build failed for %s: %s", market["market_id"], exc)
            features = None

        # Ensemble prediction
        ensemble_prob = ensemble.predict(ln_prob, features_dict=features)

        # Edge
        edge = compute_edge(market, ensemble_prob)

        # Kalshi mid price
        yes_mid = ((market.get("yes_bid", 0) or 0) + (market.get("yes_ask", 100) or 100)) / 2.0

        # Kelly bet
        kelly_usd = _kelly_bet_size(
            edge=edge,
            model_prob=ensemble_prob,
            bankroll=cfg.BANKROLL,
            kelly_fraction=cfg.KELLY_FRACTION,
            max_bet_pct=cfg.MAX_BET_PCT,
        )

        direction = "YES" if edge > 0 else "NO"
        expiry_str = expiry.strftime("%m/%d %H:%M") if expiry else "N/A"

        alerts.append(
            {
                "market_id": market["market_id"],
                "title": market.get("title", ""),
                "expiry": expiry_str,
                "expiry_dt": expiry,
                "strike": strike,
                "current_price": btc_price,
                "kalshi_yes_mid_pct": yes_mid,
                "model_prob": ensemble_prob,
                "edge": edge,
                "kelly_bet_usd": kelly_usd,
                "direction": direction,
                "abs_edge": abs(edge),
                "yes_bid": market.get("yes_bid", 0),
                "yes_ask": market.get("yes_ask", 100),
            }
        )

    # Sort by absolute edge descending
    alerts.sort(key=lambda a: a["abs_edge"], reverse=True)

    # Filter by threshold
    qualifying = [a for a in alerts if a["abs_edge"] >= cfg.EDGE_THRESHOLD]

    # ------------------------------------------------------------------
    # 3. Display rich table
    # ------------------------------------------------------------------
    table = Table(
        title=f"BTC Kalshi Opportunities — {now_utc.strftime('%Y-%m-%d %H:%M UTC')}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Market", style="bold")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("BTC $", justify="right")
    table.add_column("Kalshi %", justify="right")
    table.add_column("Model %", justify="right")
    table.add_column("Edge %", justify="right")
    table.add_column("Kelly $", justify="right")
    table.add_column("Dir", justify="center")

    for rank, a in enumerate(alerts[:20], start=1):  # Show top 20
        edge_pct = a["edge"] * 100
        edge_color = "green" if abs(edge_pct) >= cfg.EDGE_THRESHOLD * 100 else "white"
        edge_str = f"[{edge_color}]{edge_pct:+.1f}%[/{edge_color}]"

        dir_color = "green" if a["direction"] == "YES" else "red"
        dir_str = f"[{dir_color}]{a['direction']}[/{dir_color}]"

        table.add_row(
            str(rank),
            a["market_id"][:24],
            a["expiry"],
            f"${a['strike']:,.0f}",
            f"${a['current_price']:,.0f}",
            f"{a['kalshi_yes_mid_pct']:.1f}¢",
            f"{a['model_prob']:.1%}",
            edge_str,
            f"${a['kelly_bet_usd']:.2f}" if a["kelly_bet_usd"] > 0 else "-",
            dir_str,
        )

    console.print(Panel(table, border_style="blue"))

    if not qualifying:
        console.print(
            f"[yellow]No markets exceed edge threshold of {cfg.EDGE_THRESHOLD:.0%}.[/yellow]"
        )

    # ------------------------------------------------------------------
    # 4. Log to CSV
    # ------------------------------------------------------------------
    for a in qualifying:
        _log_alert(
            csv_path=cfg.ALERTS_CSV,
            row={
                "timestamp": now_utc.isoformat(),
                "market_id": a["market_id"],
                "title": a["title"],
                "expiry": a["expiry"],
                "strike": a["strike"],
                "current_price": a["current_price"],
                "kalshi_yes_mid_pct": f"{a['kalshi_yes_mid_pct']:.2f}",
                "model_prob_pct": f"{a['model_prob'] * 100:.2f}",
                "edge_pct": f"{a['edge'] * 100:+.2f}",
                "kelly_bet_usd": f"{a['kelly_bet_usd']:.2f}",
                "direction": a["direction"],
            },
        )

    # ------------------------------------------------------------------
    # 5. Auto-trade if enabled
    # ------------------------------------------------------------------
    if cfg.AUTO_TRADE and qualifying:
        console.print("[bold yellow]AUTO_TRADE enabled — placing orders...[/bold yellow]")
        for a in qualifying:
            try:
                side = a["direction"].lower()  # "yes" or "no"
                # Use yes_ask for YES bets (conservative), yes_bid for NO bets
                if side == "yes":
                    price_cents = int(min(a["yes_ask"], 99))
                else:
                    # Betting "no" — no_ask = 100 - yes_bid
                    no_ask = 100 - int(a["yes_bid"] or 1)
                    price_cents = int(min(max(no_ask, 1), 99))

                # Number of contracts = floor(kelly_usd / price_cents * 100)
                # Each contract costs price_cents / 100 dollars
                cost_per_contract = price_cents / 100.0
                contracts = max(1, int(a["kelly_bet_usd"] / cost_per_contract))

                result = client.place_order(
                    market_id=a["market_id"],
                    side=side,
                    contracts=contracts,
                    price=price_cents,
                )
                console.print(
                    f"[green]Placed {side.upper()} order on {a['market_id']}: "
                    f"{contracts} contracts @ {price_cents}¢[/green]"
                )
                _log_trade(
                    csv_path=cfg.TRADES_CSV,
                    market_id=a["market_id"],
                    side=side,
                    contracts=contracts,
                    price=price_cents,
                    result=result,
                )
            except Exception as exc:
                console.print(f"[red]Failed to place order on {a['market_id']}: {exc}[/red]")
                logger.error("Order placement failed for %s: %s", a["market_id"], exc)

    return qualifying

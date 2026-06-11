"""
backtest/evaluate.py — Run walk-forward backtest and evaluate strategy performance.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from models.ml_model import MLModel
from features.engineer import FEATURE_NAMES

logger = logging.getLogger(__name__)


def _kelly_bet(
    edge: float,
    model_prob: float,
    bankroll: float,
    kelly_fraction: float,
    max_bet_pct: float,
) -> float:
    """
    Compute Kelly-fractional bet size.

    For a binary yes/no market:
        full_kelly = edge / (1 - model_prob)   (approximation for binary bets)
    Fractional Kelly = kelly_fraction * full_kelly
    Capped at max_bet_pct of bankroll.
    """
    denom = 1.0 - model_prob
    if denom <= 0:
        return 0.0
    raw_kelly = kelly_fraction * abs(edge) / denom
    raw_kelly = min(raw_kelly, max_bet_pct)
    return float(bankroll * raw_kelly)


def print_backtest_report(results: Dict[str, Any], bankroll: float = 500.0) -> None:
    """Public wrapper to pretty-print backtest results (alias for _print_results)."""
    _print_results(results, bankroll)


def run_backtest(
    history_df: pd.DataFrame,
    model_probs: "Optional[pd.Series]" = None,
    edge_threshold: float = 0.05,
    bankroll: float = 500.0,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Simulate betting on the historical markets DataFrame and return performance metrics.

    Parameters
    ----------
    history_df : pd.DataFrame
        Output of ``generate_historical_markets()`` that also includes a
        ``model_prob`` column (lognormal or ensemble probability).
    edge_threshold : float
        Minimum |edge| required to place a bet.
    bankroll : float
        Starting bankroll in USD.
    kelly_fraction : float
        Fractional Kelly multiplier.
    max_bet_pct : float
        Maximum single bet as fraction of current bankroll.

    Returns
    -------
    dict with keys:
        total_bets, win_rate, total_pnl, roi, sharpe_ratio, max_drawdown, avg_edge
    """
    if history_df.empty:
        logger.warning("run_backtest: empty history DataFrame.")
        return {
            "total_bets": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
        }

    df = history_df.copy()

    # Inject externally-supplied model probabilities (e.g. from main.py)
    if model_probs is not None:
        df["model_prob"] = model_probs.values if hasattr(model_probs, "values") else list(model_probs)

    # If model_prob column missing, compute it from lognormal_prob
    if "model_prob" not in df.columns:
        if "lognormal_prob" in df.columns:
            df["model_prob"] = df["lognormal_prob"]
        else:
            raise ValueError("history_df must contain 'model_prob' or 'lognormal_prob' column.")

    # Compute edge: model_prob minus Kalshi implied
    if "edge" not in df.columns:
        df["edge"] = df["model_prob"] - df["kalshi_implied_prob"]

    # Filter by edge threshold
    bets_df = df[df["edge"].abs() >= edge_threshold].copy()
    if bets_df.empty:
        logger.warning("No bets pass the edge threshold %.2f.", edge_threshold)
        return {
            "total_bets": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
        }

    # Simulate sequential betting
    current_bankroll = bankroll
    pnl_series = []
    wins = 0

    for _, row in bets_df.iterrows():
        bet_usd = _kelly_bet(
            edge=row["edge"],
            model_prob=row["model_prob"],
            bankroll=current_bankroll,
            kelly_fraction=kelly_fraction,
            max_bet_pct=max_bet_pct,
        )
        if bet_usd < 0.01:
            continue

        # Determine direction: bet "yes" if edge > 0, "no" if edge < 0
        bet_yes = row["edge"] > 0
        outcome = int(row["actual_outcome"])

        # Payout: Kalshi contracts pay $1 each.
        # If bet_yes and outcome=1: win (1 - kalshi_yes_price) per $ risked
        # If bet_yes and outcome=0: lose bet_usd
        kalshi_yes = float(row["kalshi_implied_prob"])  # in range [0,1]
        kalshi_no = 1.0 - kalshi_yes

        if bet_yes:
            # cost per contract = kalshi_yes, payout = 1 → profit = (1-kalshi_yes)/kalshi_yes per $ bet
            if outcome == 1:
                pnl = bet_usd * (1.0 - kalshi_yes) / max(kalshi_yes, 1e-6)
                wins += 1
            else:
                pnl = -bet_usd
        else:
            # Bet "no" — cost per contract = kalshi_no, payout = 1 if outcome=0
            if outcome == 0:
                pnl = bet_usd * (1.0 - kalshi_no) / max(kalshi_no, 1e-6)
                wins += 1
            else:
                pnl = -bet_usd

        current_bankroll += pnl
        pnl_series.append(pnl)
        if current_bankroll <= 0:
            logger.warning("Bankroll exhausted during backtest.")
            break

    if not pnl_series:
        return {
            "total_bets": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "roi": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_edge": 0.0,
        }

    total_bets = len(pnl_series)
    total_pnl = float(sum(pnl_series))
    win_rate = wins / total_bets if total_bets > 0 else 0.0
    roi = total_pnl / bankroll

    # Sharpe ratio (annualized using hourly returns approximation)
    pnl_arr = np.array(pnl_series)
    daily_std = float(np.std(pnl_arr)) if len(pnl_arr) > 1 else 1.0
    sharpe = float(np.mean(pnl_arr) / daily_std * np.sqrt(24 * 365)) if daily_std > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    avg_edge = float(bets_df["edge"].abs().mean())

    results = {
        "total_bets": total_bets,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "roi": roi,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "avg_edge": avg_edge,
    }

    # Pretty-print via rich
    _print_results(results, bankroll)

    return results


def _print_results(results: Dict[str, Any], bankroll: float) -> None:
    """Print backtest results as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Starting Bankroll", f"${bankroll:,.2f}")
        table.add_row("Total Bets", str(results["total_bets"]))
        table.add_row("Win Rate", f"{results['win_rate']:.1%}")
        table.add_row("Total P&L", f"${results['total_pnl']:+,.2f}")
        table.add_row("ROI", f"{results['roi']:.1%}")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"${results['max_drawdown']:,.2f}")
        table.add_row("Avg |Edge|", f"{results['avg_edge']:.1%}")

        console.print(Panel(table, border_style="green"))
    except Exception as exc:
        logger.warning("Could not print rich table: %s", exc)
        for k, v in results.items():
            print(f"  {k}: {v}")


def print_backtest_report(results: Dict[str, Any]) -> None:
    """Convenience wrapper — pretty-print a results dict from run_backtest()."""
    _print_results(results, bankroll=results.get("starting_bankroll", 0.0))


def train_model_from_history(history_df: pd.DataFrame) -> MLModel:
    """
    Extract features and outcomes from the history DataFrame and train an XGBoost model.

    Parameters
    ----------
    history_df : pd.DataFrame
        Output of ``generate_historical_markets()``. Must contain feature columns
        and ``actual_outcome``.

    Returns
    -------
    MLModel
        Trained and saved ML model instance.
    """
    if history_df.empty:
        raise ValueError("train_model_from_history: history_df is empty.")

    # Extract only the feature columns that are present
    available_features = [f for f in FEATURE_NAMES if f in history_df.columns]
    if len(available_features) < 5:
        raise ValueError(
            f"Not enough feature columns in history_df. Found: {available_features}"
        )

    X = history_df[available_features].copy()
    y = history_df["actual_outcome"].astype(int)

    # Drop rows with all-NaN features
    valid_mask = X.notna().any(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 30:
        raise ValueError(
            f"Not enough valid samples after filtering: {len(X)} (need ≥30)."
        )

    model = MLModel()

    # Ensure all canonical features are present (fill missing with 0)
    for feat in FEATURE_NAMES:
        if feat not in X.columns:
            X[feat] = 0.0
    X = X[FEATURE_NAMES]

    model.train(X, y)
    logger.info(
        "train_model_from_history: trained on %d samples, %d features.",
        len(X),
        len(FEATURE_NAMES),
    )
    return model

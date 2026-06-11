"""
backtest/evaluate.py — Evaluate model performance and train XGBoost from history.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from features.engineer import FEATURE_NAMES
from models.ml_model import MLModel

logger = logging.getLogger(__name__)


def compute_kelly_bet(edge: float, win_prob: float, bankroll: float) -> float:
    """
    Half-Kelly bet sizing.

    kelly_full = edge / (1 - win_prob)   [approximate for binary bets]
    Returns dollar amount, capped at MAX_BET_PCT of bankroll.
    """
    if win_prob <= 0 or win_prob >= 1 or edge <= 0:
        return 0.0
    loss_prob = 1.0 - win_prob
    kelly_full = edge / loss_prob
    kelly_frac = kelly_full * config.KELLY_FRACTION
    max_bet = bankroll * config.MAX_BET_PCT
    return min(kelly_frac * bankroll, max_bet)


def run_backtest(
    history_df: pd.DataFrame,
    model_probs: pd.Series,
    edge_threshold: float = None,
) -> Dict:
    """
    Simulate betting on historical markets using model probabilities.

    Parameters
    ----------
    history_df : pd.DataFrame
        Output of backtest/simulate.py with columns: actual_outcome, kalshi_implied_prob.
    model_probs : pd.Series
        Ensemble model probability for each row (index-aligned to history_df).
    edge_threshold : float
        Minimum edge required to bet. Defaults to config.EDGE_THRESHOLD.

    Returns
    -------
    dict with performance metrics.
    """
    if edge_threshold is None:
        edge_threshold = config.EDGE_THRESHOLD

    df = history_df.copy()
    df["model_prob"] = model_probs.values
    df["edge"] = df["model_prob"] - df["kalshi_implied_prob"]

    # Only bet when edge exceeds threshold
    bet_mask = df["edge"].abs() >= edge_threshold
    bets = df[bet_mask].copy()

    if len(bets) == 0:
        logger.warning("No bets passed edge threshold %.2f in backtest.", edge_threshold)
        return {"total_bets": 0, "win_rate": 0, "total_pnl": 0, "roi": 0, "sharpe": 0, "max_drawdown": 0}

    bankroll = config.BANKROLL
    pnl_list = []
    cumulative = 0.0

    for _, row in bets.iterrows():
        edge = row["edge"]
        model_prob = row["model_prob"]
        actual = row["actual_outcome"]

        # Bet YES if edge > 0, NO if edge < 0
        if edge > 0:
            win_prob = model_prob
            side_correct = actual == 1
        else:
            win_prob = 1.0 - model_prob
            side_correct = actual == 0

        bet_size = compute_kelly_bet(abs(edge), win_prob, bankroll)
        if bet_size <= 0:
            continue

        # Kalshi: win gets (100 - price) per $price bet
        # Simplified: pnl = +bet_size if correct, -bet_size if wrong
        pnl = bet_size if side_correct else -bet_size
        pnl_list.append(pnl)
        cumulative += pnl
        bankroll = max(bankroll + pnl, 1.0)  # bankroll can't go below $1

    if not pnl_list:
        return {"total_bets": 0}

    pnl_arr = np.array(pnl_list)
    wins = int((pnl_arr > 0).sum())
    win_rate = wins / len(pnl_arr)
    total_pnl = float(pnl_arr.sum())
    roi = total_pnl / config.BANKROLL

    # Sharpe (annualized hourly)
    mean_pnl = pnl_arr.mean()
    std_pnl = pnl_arr.std() + 1e-9
    sharpe = (mean_pnl / std_pnl) * np.sqrt(8760)

    # Max drawdown
    cumulative_arr = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cumulative_arr)
    drawdown = peak - cumulative_arr
    max_drawdown = float(drawdown.max())

    avg_edge = float(bets["edge"].abs().mean())

    return {
        "total_bets": len(pnl_arr),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 2),
        "avg_edge": round(avg_edge, 4),
    }


def train_model_from_history(history_df: pd.DataFrame) -> MLModel:
    """
    Train XGBoost on simulated historical markets.
    Uses walk-forward: train on first 80%, evaluate on last 20%.
    """
    from sklearn.model_selection import TimeSeriesSplit

    model = MLModel(pkl_path=config.MODEL_PKL_PATH)

    feature_cols = [c for c in FEATURE_NAMES if c in history_df.columns]
    X = history_df[feature_cols].copy()
    y = history_df["actual_outcome"].copy()

    if len(X) < 100:
        logger.warning("Not enough history (%d rows) to train. Need 100+.", len(X))
        return model

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    logger.info("Training XGBoost on %d samples...", len(X_train))
    model.train(X_train, y_train)

    # Quick validation
    val_probs = pd.Series([model.predict_proba(row.to_dict()) for _, row in X_val.iterrows()])
    val_preds = (val_probs >= 0.5).astype(int)
    val_acc = (val_preds.values == y_val.values).mean()
    logger.info("Validation accuracy: %.2f%%", val_acc * 100)

    return model


def print_backtest_report(metrics: Dict) -> None:
    """Print a formatted backtest report to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(title="Backtest Results", box=box.ROUNDED, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Bets", str(metrics.get("total_bets", 0)))
        table.add_row("Win Rate", f"{metrics.get('win_rate', 0)*100:.1f}%")
        table.add_row("Total P&L", f"${metrics.get('total_pnl', 0):.2f}")
        table.add_row("ROI", f"{metrics.get('roi', 0)*100:.1f}%")
        table.add_row("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
        table.add_row("Max Drawdown", f"${metrics.get('max_drawdown', 0):.2f}")
        table.add_row("Avg Edge", f"{metrics.get('avg_edge', 0)*100:.2f}%")

        console.print(table)
    except ImportError:
        for k, v in metrics.items():
            print(f"  {k}: {v}")

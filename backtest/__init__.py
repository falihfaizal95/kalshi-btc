"""backtest package — historical simulation and model evaluation."""
from .simulate import generate_historical_markets
from .evaluate import run_backtest, train_model_from_history, print_backtest_report

__all__ = ["generate_historical_markets", "run_backtest", "train_model_from_history", "print_backtest_report"]

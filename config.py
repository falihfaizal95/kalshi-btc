"""
config.py — Load all environment variables from .env and expose as typed constants.
"""

import os
import pathlib

from dotenv import load_dotenv

load_dotenv()

# Kalshi API credentials (generate a key pair at kalshi.com -> Account -> API keys).
# Only required for order placement / balance; market scanning is public.
KALSHI_API_KEY_ID: str = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

_demo_raw: str = os.getenv("KALSHI_DEMO", "false").strip().lower()
KALSHI_DEMO: bool = _demo_raw in ("1", "true", "yes", "on")

# Trading parameters
BANKROLL: float = float(os.getenv("BANKROLL", "500"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.5"))
MAX_BET_PCT: float = float(os.getenv("MAX_BET_PCT", "0.05"))
EDGE_THRESHOLD: float = float(os.getenv("EDGE_THRESHOLD", "0.05"))
# Skip markets whose bid/ask spread is wider than this (illiquid/empty books)
MAX_SPREAD_CENTS: float = float(os.getenv("MAX_SPREAD_CENTS", "10"))
# Only trade markets expiring within this many hours (default: the next hour only)
MAX_EXPIRY_HOURS: float = float(os.getenv("MAX_EXPIRY_HOURS", "1"))
# Weight of the XGBoost model in the ensemble. Default 0 (lognormal only): the
# model is trained on synthetic backtest data and is untrustworthy live until
# the daily loop has retrained it on real settled outcomes.
ML_WEIGHT: float = float(os.getenv("ML_WEIGHT", "0"))
# "Confident" trading window: minutes-to-expiry under which plays are most
# reliable (outcome more locked in). Used by the phone-alert recommender.
CONFIDENT_WINDOW_MINUTES: float = float(os.getenv("CONFIDENT_WINDOW_MINUTES", "30"))
# Select trades by expected return, not raw edge. Out-of-sample, requiring a
# >=15% expected return per $1 staked ~tripled ROI vs. betting every +EV market.
EV_THRESHOLD: float = float(os.getenv("EV_THRESHOLD", "0.15"))

# Auto-trade flag
_auto_trade_raw: str = os.getenv("AUTO_TRADE", "false").strip().lower()
AUTO_TRADE: bool = _auto_trade_raw in ("1", "true", "yes", "on")

# Paper trading: simulate fills against a virtual bankroll instead of placing
# real orders. Default ON so the bot can build a real-outcome track record
# with no money at risk. AUTO_TRADE (real orders) takes precedence if both set.
_paper_raw: str = os.getenv("PAPER_TRADE", "true").strip().lower()
PAPER_TRADE: bool = _paper_raw in ("1", "true", "yes", "on")
PAPER_STARTING_BANKROLL: float = float(os.getenv("PAPER_STARTING_BANKROLL", str(BANKROLL)))

# Derived paths
ROOT_DIR = pathlib.Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
TRACKING_DIR = ROOT_DIR / "tracking"
MODEL_PKL_PATH = MODELS_DIR / "btc_model.pkl"
ALERTS_CSV = LOGS_DIR / "alerts.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"
PAPER_TRADES_CSV = TRACKING_DIR / "paper_trades.csv"
PREDICTIONS_CSV = TRACKING_DIR / "predictions.csv"
SETTLEMENTS_CSV = TRACKING_DIR / "settlements.csv"

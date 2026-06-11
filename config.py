"""
config.py — Load all environment variables from .env and expose as typed constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Kalshi credentials
KALSHI_EMAIL: str = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASSWORD: str = os.getenv("KALSHI_PASSWORD", "")

# Trading parameters
BANKROLL: float = float(os.getenv("BANKROLL", "500"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.5"))
MAX_BET_PCT: float = float(os.getenv("MAX_BET_PCT", "0.05"))
EDGE_THRESHOLD: float = float(os.getenv("EDGE_THRESHOLD", "0.05"))

# Auto-trade flag
_auto_trade_raw: str = os.getenv("AUTO_TRADE", "false").strip().lower()
AUTO_TRADE: bool = _auto_trade_raw in ("1", "true", "yes", "on")

# Derived paths
import pathlib
ROOT_DIR = pathlib.Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
MODEL_PKL_PATH = MODELS_DIR / "btc_model.pkl"
ALERTS_CSV = LOGS_DIR / "alerts.csv"
TRADES_CSV = LOGS_DIR / "trades.csv"

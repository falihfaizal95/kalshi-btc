"""
models/calibrator.py — Learns, from real settled outcomes, how to combine our
lognormal model probability with the market's own implied probability.

Out-of-sample testing showed the market price is ~4-5x more informative than our
standalone model, and blending the two (a 2-parameter logistic on the log-odds
of each) improves both ROI and calibration. This is the core "learn from more
trades" mechanism: it refits as the settled dataset grows.

Falls back to the raw model probability until enough real outcomes exist.
"""

from __future__ import annotations

import csv
import logging
import pathlib
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES = 300  # need a reasonable real-outcome sample before trusting the fit


def _logit(p: np.ndarray | float):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Calibrator:
    """Logistic blend of model and market probabilities, fit on real outcomes."""

    def __init__(self, pkl_path: Optional[pathlib.Path] = None) -> None:
        self.pkl_path = pathlib.Path(pkl_path) if pkl_path else (
            pathlib.Path(__file__).parent / "calibrator.pkl"
        )
        self._lr = None

    def load(self) -> bool:
        if not self.pkl_path.exists():
            return False
        try:
            import joblib
            self._lr = joblib.load(self.pkl_path)
            return True
        except Exception as exc:
            logger.warning("Calibrator load failed: %s", exc)
            self._lr = None
            return False

    def fit(self, model_yes: np.ndarray, market_yes: np.ndarray, actual_yes: np.ndarray) -> bool:
        """Fit on (model prob, market prob, outcome). Returns True if fitted+saved."""
        if len(actual_yes) < MIN_SAMPLES or len(set(actual_yes.tolist())) < 2:
            logger.info("Calibrator: not enough data (%d); keeping raw model.", len(actual_yes))
            return False
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([_logit(model_yes), _logit(market_yes)])
        lr = LogisticRegression().fit(X, actual_yes)
        self._lr = lr
        try:
            import joblib
            self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(lr, self.pkl_path)
        except Exception as exc:
            logger.warning("Calibrator save failed: %s", exc)
        logger.info(
            "Calibrator fit on %d outcomes (model coef=%.2f, market coef=%.2f).",
            len(actual_yes), lr.coef_[0][0], lr.coef_[0][1],
        )
        return True

    def predict(self, model_yes: float, market_yes: float) -> float:
        """Calibrated P(YES). Falls back to the raw model prob if not fitted."""
        if self._lr is None:
            return float(model_yes)
        X = np.array([[_logit(model_yes), _logit(market_yes)]])
        return float(np.clip(self._lr.predict_proba(X)[0][1], 1e-6, 1 - 1e-6))


def train_from_history(cfg) -> bool:
    """Join predictions + settlements into (model, market, outcome) and fit."""
    preds = {}
    try:
        for r in csv.DictReader(open(cfg.PREDICTIONS_CSV)):
            preds[r["pred_id"]] = r
    except FileNotFoundError:
        return False

    model_yes, market_yes, actual_yes = [], [], []
    try:
        settle_rows = list(csv.DictReader(open(cfg.SETTLEMENTS_CSV)))
    except FileNotFoundError:
        return False

    for s in settle_rows:
        p = preds.get(s["pred_id"])
        if not p or not p.get("model_prob") or "actual_yes" not in s:
            continue
        my = float(p["model_prob"])  # model's P(YES)
        cost = float(p["cost"])
        # market P(YES): cost is the price of the side we bet
        mkt = cost if p["direction"] == "YES" else 1.0 - cost
        model_yes.append(my)
        market_yes.append(mkt)
        actual_yes.append(int(s["actual_yes"]))

    if not actual_yes:
        return False
    cal = Calibrator(cfg.MODELS_DIR / "calibrator.pkl")
    return cal.fit(np.array(model_yes), np.array(market_yes), np.array(actual_yes))

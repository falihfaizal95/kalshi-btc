"""
models/ml_model.py — XGBoost binary classifier for BTC above/below strike probability.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Default path — can be overridden by passing pkl_path to MLModel()
_DEFAULT_PKL = pathlib.Path(__file__).parent / "btc_model.pkl"

# Canonical feature order (must match features/engineer.py FEATURE_NAMES)
FEATURE_NAMES: List[str] = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bb_upper",
    "bb_lower",
    "bb_pct",
    "adx_14",
    "ema_9",
    "ema_21",
    "ema_50",
    "volume_sma_20",
    "volume_ratio",
    "price_change_1h",
    "price_change_4h",
    "price_change_24h",
    "hour_of_day",
    "day_of_week",
    "distance_to_strike",
    "time_to_expiry_hours",
    "iv_annualized",
    "fear_greed_score",
    "volume_last_1h",
]


class MLModel:
    """
    XGBoost-based binary classifier that predicts P(BTC closes above strike).

    Usage
    -----
    model = MLModel()
    if not model.load():
        model.train(X_train, y_train)

    prob = model.predict_proba(feature_dict)
    """

    def __init__(self, pkl_path: Optional[pathlib.Path] = None) -> None:
        self.pkl_path = pathlib.Path(pkl_path) if pkl_path else _DEFAULT_PKL
        self._clf: Optional[XGBClassifier] = None
        self._trained = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Attempt to load a previously saved model.

        Returns True if successful, False if no saved model exists.
        """
        if not self.pkl_path.exists():
            logger.info("No saved model found at %s.", self.pkl_path)
            return False
        try:
            self._clf = joblib.load(self.pkl_path)
            self._trained = True
            logger.info("Loaded XGBoost model from %s.", self.pkl_path)
            return True
        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", self.pkl_path, exc)
            self._clf = None
            self._trained = False
            return False

    def _save(self) -> None:
        """Persist the fitted model to disk."""
        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, self.pkl_path)
        logger.info("Saved XGBoost model to %s.", self.pkl_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the XGBoost classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Columns must include (at minimum) FEATURE_NAMES;
            extra columns are silently dropped.
        y : pd.Series
            Binary labels — 1 if BTC closed above strike, 0 otherwise.
        """
        if len(X) < 30:
            raise ValueError(
                f"Training requires at least 30 samples, got {len(X)}."
            )

        X_aligned = self._align_features(X)

        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        self._clf = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        self._clf.fit(X_aligned, y)
        self._trained = True
        self._save()
        logger.info(
            "Trained XGBoost on %d samples (pos=%d, neg=%d).",
            len(X), pos_count, neg_count,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features_dict: Dict[str, float]) -> float:
        """
        Predict P(BTC > strike) for a single observation.

        Parameters
        ----------
        features_dict : dict
            Feature name -> value mapping. Missing keys default to 0.

        Returns
        -------
        float in [0, 1].  Returns 0.5 if model is not trained.
        """
        if not self._trained or self._clf is None:
            logger.debug("ML model not trained; returning 0.5.")
            return 0.5

        row = {name: features_dict.get(name, 0.0) for name in FEATURE_NAMES}
        df = pd.DataFrame([row], columns=FEATURE_NAMES)

        # Replace inf/nan with 0
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        proba = self._clf.predict_proba(df)[0][1]
        return float(np.clip(proba, 1e-6, 1 - 1e-6))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has exactly the canonical feature columns in order."""
        for col in FEATURE_NAMES:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0
        X = X[FEATURE_NAMES].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return the canonical ordered list of feature names."""
        return list(FEATURE_NAMES)

    @property
    def is_trained(self) -> bool:
        return self._trained

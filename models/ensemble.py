"""
models/ensemble.py — Weighted ensemble of log-normal and XGBoost model probabilities.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Blend log-normal and ML model probabilities.

    On the first call to ``predict()``, the ML model is loaded from disk
    (if available). If it is not available the ensemble degrades gracefully
    to log-normal only.

    Parameters
    ----------
    ml_weight : float
        Weight given to the ML model (0-1). The log-normal model receives
        ``1 - ml_weight``. Default 0.6.
    model_dir : str, optional
        Directory where the ML model pickle is stored. Defaults to the
        package's own models/ directory (absolute, independent of cwd).
    """

    def __init__(self, ml_weight: float = 0.6, model_dir: Optional[str] = None) -> None:
        import pathlib
        self._ml_weight = float(ml_weight)
        self._model_dir = pathlib.Path(model_dir) if model_dir else pathlib.Path(__file__).parent
        self._ml: Optional[object] = None   # lazy-loaded MLModel
        self._ml_loaded = False
        self._ml_trained = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_ml_loaded(self) -> None:
        if self._ml_loaded:
            return
        try:
            from models.ml_model import MLModel
            ml = MLModel(pkl_path=self._model_dir / "btc_model.pkl")
            self._ml_trained = ml.load()
            self._ml = ml
        except Exception as exc:
            logger.warning("Could not load ML model: %s. Falling back to lognormal only.", exc)
            self._ml_trained = False
        finally:
            self._ml_loaded = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        lognormal_prob: float,
        features_dict: Optional[Dict[str, float]] = None,
        ml_weight: Optional[float] = None,
    ) -> float:
        """
        Return blended probability P(BTC > strike).

        Parameters
        ----------
        lognormal_prob : float
            Probability from the log-normal model.
        features_dict : dict, optional
            Feature dict for ML prediction. If None or ML not trained,
            the ensemble falls back to lognormal_prob.
        ml_weight : float, optional
            Override instance ml_weight for this call.

        Returns
        -------
        float in [0, 1].
        """
        self._ensure_ml_loaded()

        effective_weight = ml_weight if ml_weight is not None else self._ml_weight

        if not self._ml_trained or self._ml is None or features_dict is None:
            # Cold start or no features: use lognormal only
            return float(max(0.0, min(1.0, lognormal_prob)))

        try:
            ml_prob = self._ml.predict_proba(features_dict)
        except Exception as exc:
            logger.warning("ML predict_proba failed: %s. Using lognormal.", exc)
            return float(max(0.0, min(1.0, lognormal_prob)))

        blended = (1.0 - effective_weight) * lognormal_prob + effective_weight * ml_prob
        return float(max(0.0, min(1.0, blended)))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def ml_is_trained(self) -> bool:
        self._ensure_ml_loaded()
        return self._ml_trained

    def retrain(self, X, y) -> None:
        """Re-train the underlying ML model and reload it."""
        from models.ml_model import MLModel
        ml = MLModel(pkl_path=self._model_dir / "btc_model.pkl")
        ml.train(X, y)
        self._ml = ml
        self._ml_trained = True
        self._ml_loaded = True
        logger.info("EnsembleModel: ML model retrained.")


def predict(
    lognormal_prob: float,
    ml_prob: float,
    ml_weight: float = 0.6,
    ml_trained: bool = True,
) -> float:
    """
    Module-level convenience function for blending probabilities.

    If ml_trained is False, returns lognormal_prob unchanged.
    """
    if not ml_trained:
        return float(max(0.0, min(1.0, lognormal_prob)))
    blended = (1.0 - ml_weight) * lognormal_prob + ml_weight * ml_prob
    return float(max(0.0, min(1.0, blended)))

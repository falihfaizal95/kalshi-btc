"""
models/lognormal.py — Log-normal probability model for BTC price targets.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def prob_above_strike(
    current_price: float,
    strike: float,
    time_hours: float,
    annual_vol: float,
) -> float:
    """
    P(BTC > strike at T) using a log-normal price model (risk-neutral).

    Under the log-normal assumption:
        ln(S_T / S_0) ~ N((mu - 0.5*sigma^2)*T, sigma^2*T)

    For a probability estimate we use mu=0 (martingale / risk-neutral drift),
    which is standard for short-horizon binary option pricing.

    Parameters
    ----------
    current_price : float
        Current BTC price S_0.
    strike : float
        Target strike K.
    time_hours : float
        Hours until expiry.
    annual_vol : float
        Annualized volatility (e.g. 0.65 for 65%).

    Returns
    -------
    float
        Probability in [0, 1] that BTC is above the strike at expiry.
    """
    T = time_hours / 8760.0  # fraction of a year

    if T <= 0:
        return 1.0 if current_price > strike else 0.0

    if current_price <= 0 or strike <= 0 or annual_vol <= 0:
        return 0.5  # degenerate inputs — return neutral probability

    # d1-equivalent for P(S_T > K) with zero drift
    d = (np.log(current_price / strike) + 0.5 * annual_vol**2 * T) / (
        annual_vol * np.sqrt(T)
    )
    return float(norm.cdf(d))


def prob_below_strike(
    current_price: float,
    strike: float,
    time_hours: float,
    annual_vol: float,
) -> float:
    """
    P(BTC < strike at T) = 1 - P(BTC > strike at T).
    """
    return 1.0 - prob_above_strike(current_price, strike, time_hours, annual_vol)

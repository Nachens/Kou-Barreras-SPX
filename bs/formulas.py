"""Black-Scholes con dividendo continuo delta (§5.2.1 de la tesis)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d_pm(S0, K, T, r, delta, sigma):
    s = sigma * np.sqrt(T)
    dp = (np.log(S0 / K) + (r - delta + 0.5 * sigma ** 2) * T) / s
    return dp, dp - s


def call(S0, K, T, r, delta, sigma):
    dp, dm = d_pm(S0, K, T, r, delta, sigma)
    return S0 * np.exp(-delta * T) * norm.cdf(dp) - K * np.exp(-r * T) * norm.cdf(dm)


def put(S0, K, T, r, delta, sigma):
    dp, dm = d_pm(S0, K, T, r, delta, sigma)
    return K * np.exp(-r * T) * norm.cdf(-dm) - S0 * np.exp(-delta * T) * norm.cdf(-dp)


def vega(S0, K, T, r, delta, sigma):
    dp, _ = d_pm(S0, K, T, r, delta, sigma)
    return S0 * np.exp(-delta * T) * norm.pdf(dp) * np.sqrt(T)

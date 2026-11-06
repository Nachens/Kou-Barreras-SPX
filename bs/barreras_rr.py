"""Barreras bajo Black-Scholes (límite sin saltos de Kou), sin rebate.

* rr_up_out_call / rr_down_out_put: combinación [1]-[2]+[3]-[4] de Rubinstein y Reiner, con
  (eta, phi) = (-1, 1) para la call up-and-out y (1, -1) para la put down-and-out (§5.2.4).
* reflexion_*: la misma barrera por el principio de reflexión del browniano con deriva
  (P(X_T >= a, M_T >= b), a <= b). Las dos tienen que coincidir.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

N = norm.cdf


def _rr(S0, K, H, T, r, delta, sigma, eta, phi):
    b = r - delta
    mu = (b - sigma ** 2 / 2) / sigma ** 2
    s = sigma * np.sqrt(T)
    x1 = np.log(S0 / K) / s + (1 + mu) * s
    x2 = np.log(S0 / H) / s + (1 + mu) * s
    y1 = np.log(H ** 2 / (S0 * K)) / s + (1 + mu) * s
    y2 = np.log(H / S0) / s + (1 + mu) * s
    f1 = phi * S0 * np.exp((b - r) * T)
    f2 = phi * K * np.exp(-r * T)
    A = f1 * N(phi * x1) - f2 * N(phi * x1 - phi * s)
    B = f1 * N(phi * x2) - f2 * N(phi * x2 - phi * s)
    C = f1 * (H / S0) ** (2 * (mu + 1)) * N(eta * y1) - f2 * (H / S0) ** (2 * mu) * N(eta * y1 - eta * s)
    D = f1 * (H / S0) ** (2 * (mu + 1)) * N(eta * y2) - f2 * (H / S0) ** (2 * mu) * N(eta * y2 - eta * s)
    return A - B + C - D


def rr_up_out_call(S0, K, L, T, r, delta, sigma):
    if K >= L:
        return 0.0
    return _rr(S0, K, L, T, r, delta, sigma, eta=-1, phi=1)


def rr_down_out_put(S0, K, L, T, r, delta, sigma):
    if K <= L:
        return 0.0
    return _rr(S0, K, L, T, r, delta, sigma, eta=1, phi=-1)


# ---------------------------------------------------------------- reflexión
def _joint(a, b, g, s, T):
    """P(X_T >= a, max_{t<=T} X_t >= b) para X = g t + s W, a <= b, b > 0."""
    sT = s * np.sqrt(T)
    return N((g * T - b) / sT) + np.exp(2 * g * b / s ** 2) * (N((-b - g * T) / sT) - N((a - 2 * b - g * T) / sT))


def reflexion_up_out_call(S0, K, L, T, r, delta, s):
    from . import formulas as f
    if K >= L:
        return 0.0
    k, b = np.log(K / S0), np.log(L / S0)
    g = r - delta - s * s / 2
    g1 = g + s * s
    vin = S0 * np.exp(-delta * T) * _joint(k, b, g1, s, T) - K * np.exp(-r * T) * _joint(k, b, g, s, T)
    return f.call(S0, K, T, r, delta, s) - vin


def reflexion_down_out_put(S0, K, L, T, r, delta, s):
    from . import formulas as f
    if K <= L:
        return 0.0
    kp, bp = np.log(S0 / K), np.log(S0 / L)
    g = -(r - delta - s * s / 2)
    g1 = g - s * s
    vin = K * np.exp(-r * T) * _joint(kp, bp, g, s, T) - S0 * np.exp(-delta * T) * _joint(kp, bp, g1, s, T)
    return f.put(S0, K, T, r, delta, s) - vin

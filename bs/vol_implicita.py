"""Volatilidad implícita por Brent en el intervalo de no arbitraje (§5.2.1).

Devuelve NaN fuera del intervalo ((S0 e^{-delta T} - K e^{-rT})^+, S0 e^{-delta T}); nunca un valor
inventado. Las puts se pasan a call por paridad.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from . import formulas as bsf


def vol_implicita_call(C, S0, K, T, r, delta, lo=1e-6, hi=10.0):
    base = S0 * np.exp(-delta * T)
    inf = max(base - K * np.exp(-r * T), 0.0)
    if not (np.isfinite(C) and inf < C < base):
        return np.nan
    f = lambda s: bsf.call(S0, K, T, r, delta, s) - C
    try:
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def vol_implicita(precio, tipo, S0, K, T, r, delta):
    """tipo 'C' o 'P'."""
    C = precio if tipo == "C" else precio + S0 * np.exp(-delta * T) - K * np.exp(-r * T)
    return vol_implicita_call(C, S0, K, T, r, delta)

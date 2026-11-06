"""Inversión de Laplace por Gaver-Stehfest con extrapolación de Richardson (ec. (5.3) de la tesis).

f*_n(t) = sum_{k=1}^n varrho_{k,n} f~_{k+B}(t),  varrho_{k,n} = (-1)^{n-k} k^n / (k!(n-k)!),
f~_m(t) = (ln2/t) (2m)!/(m!(m-1)!) sum_{i=0}^m (-1)^i C(m,i) fhat((m+i) ln2/t).

Todo es una combinación lineal de fhat en alpha_j = j ln2/t, j = 1..2(n+B): los pesos son
racionales exactos y las transformadas (y sus raíces) se evalúan una vez por j.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import mpmath as mp


def dps_defecto(n, B=2):
    return max(30, math.ceil(2.5 * (n + B)) + 5)


# Tramos de n para la inversión adaptativa: la precisión se fija para el extremo del tramo, así las
# raíces se recalculan solo al pasar de un tramo al siguiente.
TRAMOS_N = (20, 40, 60, 80, 100, 140)


def tramo(n):
    """Menor extremo de tramo >= n."""
    for t in TRAMOS_N:
        if n <= t:
            return t
    return n


def invertir_desde(fv, T, n, B=2):
    """f*_n(T) con los valores fv[j] = fhat(alpha_j) ya calculados (dict j -> mpf)."""
    ln2 = mp.log(2)
    tot = mp.mpf(0)
    for j, wj in pesos_gs(n, B).items():
        tot += mp.mpf(wj.numerator) / mp.mpf(wj.denominator) * fv[j]
    return ln2 / mp.mpf(T) * tot


@lru_cache(maxsize=None)
def pesos_gs(n, B=2):
    """Dict j -> peso racional w_j tal que f*_n(t) = (ln2/t) sum_j w_j fhat(j ln2/t)."""
    w: dict[int, Fraction] = {}
    for k in range(1, n + 1):
        rho = Fraction((-1) ** (n - k) * k ** n, math.factorial(k) * math.factorial(n - k))
        m = k + B
        base = rho * Fraction(math.factorial(2 * m), math.factorial(m) * math.factorial(m - 1))
        for i in range(m + 1):
            j = m + i
            w[j] = w.get(j, Fraction(0)) + base * (-1) ** i * math.comb(m, i)
    return w


def indices(n, B=2):
    return sorted(pesos_gs(n, B))


def invertir(fhat, T, n=14, B=2):
    """f*_n(T) con fhat(j) = transformada en alpha_j = j ln2/T (ya evaluada por quien llama)."""
    ln2 = mp.log(2)
    tot = mp.mpf(0)
    for j, wj in pesos_gs(n, B).items():
        tot += mp.mpf(wj.numerator) / mp.mpf(wj.denominator) * fhat(j)
    return ln2 / mp.mpf(T) * tot

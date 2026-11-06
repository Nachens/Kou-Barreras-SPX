"""Control: Teorema B.35 (forma cerrada de Kou y Wang, Teo. 4.1) en mpmath.

Transformada de P(X_t >= a, tau_b <= t) con a <= b, como serie de Poisson truncada en N términos con las
funciones H_i(kappa, upsilon, c; n) = (1/sqrt(2 pi)) int_0^inf e^{(c^2/2 - upsilon) t} t^{n+i/2} Hh_i(c sqrt(t) + kappa/sqrt(t)) dt
(Def. B.34). Las H_i con i >= 1 salen de H_{-1} y H_0 por la recursión
    H_i(n) = H_{i-2}(n+1)/i - (c/i) H_{i-1}(n+1) - (kappa/i) H_{i-1}(n),
con dígitos de guarda (la recursión hacia arriba pierde dígitos). H_{-1} tiene forma cerrada (Bessel K de
orden semientero) y H_0 se calcula por cuadratura. `H_cuadratura` calcula cualquier H_i por cuadratura directa
(con la función cilíndrica parabólica) para validar la recursión. Solo se usa en pocos puntos: la velocidad no importa.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import mpmath as mp

from .raices import raices
from .transformadas import pesos_AB

GUARDA_BASE = 25     # dígitos de guarda: GUARDA_BASE + GUARDA_POR_PASO * N (la recursión pierde ~7 dígitos por paso)
GUARDA_POR_PASO = 7
_CORTES = (0.05, 0.2, 0.5, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 55, 75, 100, 130, 170, 230)   # en unidades de 1/nu


def guarda(N):
    return GUARDA_BASE + GUARDA_POR_PASO * N


def _H_base_m1(kappa, nu, c, m):
    """H_{-1}(kappa, nu, c; m): forma cerrada con Bessel K (o Gamma si kappa = 0)."""
    kappa, nu, c = mp.mpf(kappa), mp.mpf(nu), mp.mpf(c)
    pref = mp.exp(-c * kappa) / mp.sqrt(2 * mp.pi)
    if kappa == 0:
        return pref * mp.gamma(m + mp.mpf(1) / 2) / nu ** (m + mp.mpf(1) / 2)
    z = abs(kappa) * mp.sqrt(2 * nu)
    I = 2 * (kappa ** 2 / (2 * nu)) ** ((m + mp.mpf(1) / 2) / 2) * mp.besselk(m + mp.mpf(1) / 2, z)
    return pref * I


def _H_base_0(kappa, nu, c, m):
    """H_0(kappa, nu, c; m) = int_0^inf e^{(c^2/2 - nu) t} t^m N(-x) dt, x = c sqrt(t) + kappa/sqrt(t)."""
    kappa, nu, c = mp.mpf(kappa), mp.mpf(nu), mp.mpf(c)

    def f(t):
        x = c * mp.sqrt(t) + kappa / mp.sqrt(t)
        return mp.exp((c * c / 2 - nu) * t) * t ** m * mp.erfc(x / mp.sqrt(2)) / 2
    pts = [0] + [mp.mpf(u) / nu for u in _CORTES]
    return mp.quad(f, pts, maxdegree=10)


def _tarea(args):
    dps, i, kappa, nu, c, m = args
    with mp.workdps(dps):
        v = _H_base_m1(kappa, nu, c, m) if i == -1 else _H_base_0(kappa, nu, c, m)
        return mp.nstr(v, dps)


def tabla_H(combos, nu, mmax, dps, pool=None):
    """Valores base H_{-1}, H_0 para m = 0..mmax en cada (kappa, c) de `combos`. Dict ((kappa, c), i, m) -> mpf."""
    tareas = [(dps, i, k, nu, c, m) for (k, c) in combos for i in (-1, 0) for m in range(mmax + 1)]
    if pool is not None:
        res = list(pool.map(_tarea, tareas))
    else:
        res = [_tarea(t) for t in tareas]
    out = {}
    for (d_, i, k, nu_, c, m), v in zip(tareas, res):
        out[((k, c), i, m)] = mp.mpf(v)
    return out


def H_recursion(base, key, i, n, cache):
    """H_i(kappa, nu, c; n) por la recursión desde la tabla base."""
    (kappa, c) = key
    if i in (-1, 0):
        return base[(key, i, n)]
    k = (key, i, n)
    if k not in cache:
        cache[k] = (H_recursion(base, key, i - 2, n + 1, cache) / i
                    - mp.mpf(c) / i * H_recursion(base, key, i - 1, n + 1, cache)
                    - mp.mpf(kappa) / i * H_recursion(base, key, i - 1, n, cache))
    return cache[k]


def H_cuadratura(i, kappa, nu, c, n):
    """H_i por cuadratura directa, con Hh_i(x) = e^{-x^2/4} D_{-i-1}(x) (validación de la recursión)."""
    kappa, nu, c = mp.mpf(kappa), mp.mpf(nu), mp.mpf(c)

    def Hh(x):
        if i == -1:
            return mp.exp(-x * x / 2)
        return mp.exp(-x * x / 4) * mp.pcfd(-i - 1, x)

    def f(t):
        x = c * mp.sqrt(t) + kappa / mp.sqrt(t)
        return mp.exp((c * c / 2 - nu) * t) * t ** (n + mp.mpf(i) / 2) * Hh(x) / mp.sqrt(2 * mp.pi)
    pts = [0] + [mp.mpf(u) / nu for u in _CORTES]
    return mp.quad(f, pts, maxdegree=10)


def _binom(n, k):
    return math.comb(n, k)


def thm_B35(alpha, a, b, kou, N=14, dps=45, pool=None):
    """Transformada de P(X_t >= a, tau_b <= t) por el Teo. B.35 (a <= b), con N términos de la serie de Poisson."""
    G_ = guarda(N)
    with mp.workdps(dps + G_):
        k = kou.to_mp()
        alpha, a, b = mp.mpf(alpha), mp.mpf(a), mp.mpf(b)
        if a > b:
            raise ValueError("el Teo. B.35 pide a <= b")
        gam, sig, lam, p, e1, e2 = k.gamma, k.sigma, k.lam, k.p, k.eta1, k.eta2
        q = 1 - p
        rs = raices(k, alpha)
        A, B = pesos_AB(b, rs, e1)
        ell = (b - a) / sig
        cp, cm = sig * e1 + gam / sig, sig * e2 - gam / sig
        ups = alpha + lam + gam ** 2 / (2 * sig ** 2)
        r1, r2 = e1 / (e1 + e2), e2 / (e1 + e2)

        def P(n, i):
            if i == n:
                return p ** n
            return sum(_binom(n, j) * p ** j * q ** (n - j) * _binom(n - i - 1, j - i) * r1 ** (j - i) * r2 ** (n - j)
                       for j in range(i, n))

        def Q(n, i):
            if i == n:
                return q ** n
            return sum(_binom(n, j) * q ** j * p ** (n - j) * _binom(n - i - 1, j - i) * r2 ** (j - i) * r1 ** (n - j)
                       for j in range(i, n))

        def Pbar(n, i):
            if i == 1:
                return sum(Q(n, kk) * r2 ** kk for kk in range(1, n + 1))
            return P(n, i - 1)

        def Qbar(n, i):
            return sum(_binom(n, j) * q ** j * p ** (n - j) * _binom(n - i, j - i) * r2 ** (j - i) * r1 ** (n - j + 1)
                       for j in range(i, n + 1))

        combos = [(-ell, -gam / sig), (ell, cp), (-ell, cm)]
        combos = [(mp.nstr(kk, dps + G_), mp.nstr(cc, dps + G_)) for kk, cc in combos]
        base = tabla_H(combos, ups, 2 * N + 1, dps + G_, pool)
        cache = {}
        H = lambda i, kk, n: H_recursion(base, combos[kk], i, n, cache)
        fact = math.factorial
        s1 = sum(lam ** n / fact(n) * H(0, 0, n) for n in range(0, N + 1))
        tot = (A + B) * s1
        s2 = mp.mpf(0)
        for n in range(1, N + 1):
            for j in range(1, n + 1):
                co = A * P(n, j) + B * Pbar(n, j)
                s2 += lam ** n / fact(n) * co * sum((sig * e1) ** i * H(i, 1, n) for i in range(0, j))
        tot += mp.exp(ell * sig * e1) * s2
        s3 = mp.mpf(0)
        for n in range(1, N + 1):
            for j in range(1, n + 1):
                co = A * Q(n, j) + B * Qbar(n, j)
                s3 += lam ** n / fact(n) * co * sum((sig * e2) ** i * H(i, 2, n) for i in range(0, j))
        tot -= mp.exp(-ell * sig * e2) * s3
        s4 = sum((lam * p) ** n / fact(n) * sum((sig * e1) ** i * H(i, 1, n) for i in range(0, n + 1)) for n in range(0, N + 1))
        tot += mp.exp(ell * sig * e1) * B * s4
        return +tot


# --------------------------------------------------------------------------------- control
def control_b35(n_aleatorios=20, N=14, dps=20, jobs=10, semilla=7):
    """Filas de la Tabla 5.4: B.35 contra la resolvente en α fijo (3 casos de referencia y n_aleatorios)."""
    import numpy as np
    from .modelo import Kou
    from .transformadas import lt_conjunta
    casos = [(1.5, 0.1, 0.3, Kou(0.05, 0.2, 0.6, 0.4, 10.0, 5.0)),
             (1.0, 0.3, 0.3, Kou(-0.1, 0.2, 0.6, 0.5, 50.0, 1 / 0.03)),
             (2.0, -0.2, 0.3, Kou(0.1, 0.3, 0.6, 0.3, 8.0, 6.0))]
    rng = np.random.default_rng(semilla)
    # Dominio de los casos aleatorios: alpha en [1,2; 3] y lambda en [0,2; 0,5]. El error del Teo. B.35 con N términos
    # es de truncación y crece con lambda/alpha (con alpha=0,8 y lambda=0,8, N=14 da 9e-6; N=10, 1e-4): fuera de este
    # dominio N=14 no alcanza para la tolerancia de 1e-8, y no es un error de la fórmula.
    for _ in range(n_aleatorios):
        k = Kou(rng.uniform(-0.1, 0.1), rng.uniform(0.15, 0.4), rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.8),
                rng.uniform(5, 30), rng.uniform(4, 20))
        al = rng.uniform(1.2, 3.0)
        b = rng.uniform(0.05, 0.4)
        casos.append((al, b - rng.uniform(0, 0.4), b, k))
    err = 0.0
    pool = ProcessPoolExecutor(jobs) if jobs and jobs > 1 else None
    try:
        for al, a, b, k in casos:
            with mp.workdps(dps):
                rs = raices(k, al)
                ref = lt_conjunta(a, b, k, rs)[0]
            v = thm_B35(al, a, b, k, N, dps, pool)
            err = max(err, float(abs(v - ref)))
    finally:
        if pool:
            pool.shutdown()
    return [dict(control="B.35 contra la resolvente", que=f"{len(casos)} casos en α fijo (3 de la referencia + {n_aleatorios} aleatorios), N={N}",
                 error_max=err, tolerancia=1e-8, ok=bool(err <= 1e-8))]

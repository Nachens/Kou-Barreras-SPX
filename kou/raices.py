"""Las cuatro raíces reales de G(beta) = alpha (Prop. 4.3 de la tesis, §5.2.2).

beta1 in (0, eta1), beta2 in (eta1, inf) y -beta3, -beta4 con 0 < beta3 < eta2 < beta4.
Todo en mpmath con la precisión del contexto: una raíz en doble arruina la inversión.
Método: encierro + Newton protegido (cae a bisección si el paso sale del encierro).
"""
from __future__ import annotations

import mpmath as mp


def _newton_encerrado(f, fp, lo, hi, x0, tol, maxit=4000):
    """Raíz de f en (lo, hi) con f(lo) < 0 < f(hi); Newton con bisección de respaldo."""
    x = x0 if lo < x0 < hi else (lo + hi) / 2
    for _ in range(maxit):
        fx = f(x)
        if fx == 0:
            return x
        if fx < 0:
            lo = x
        else:
            hi = x
        d = fp(x)
        xn = x - fx / d if d != 0 else None
        if xn is None or not (lo < xn < hi):
            xn = (lo + hi) / 2
        if abs(xn - x) <= tol * max(1, abs(xn)):
            return xn
        x = xn
    raise RuntimeError("raíz sin converger")


def _cuatro(kou, alpha, eps, tol):
    e1, e2 = kou.eta1, kou.eta2
    f = lambda x: kou.G(x) - alpha
    fp = kou.dG
    g = lambda x: kou.G(-x) - alpha
    gp = lambda x: -kou.dG(-x)
    zero = 0 * e1 + eps          # "eps" del tipo numérico de e1
    # encierros: f<0 en el extremo izquierdo, f>0 en el derecho
    b1 = _newton_encerrado(f, fp, zero, e1 * (1 - eps), e1 / 2, tol)
    hi = e1 * 2
    while f(hi) < 0:
        hi *= 2
    b2 = _newton_encerrado(f, fp, e1 * (1 + eps), hi, (e1 + hi) / 2, tol)
    b3 = _newton_encerrado(g, gp, zero, e2 * (1 - eps), e2 / 2, tol)
    hi = e2 * 2
    while g(hi) < 0:
        hi *= 2
    b4 = _newton_encerrado(g, gp, e2 * (1 + eps), hi, (e2 + hi) / 2, tol)
    return b1, b2, b3, b4


def raices(kou, alpha):
    """(beta1, beta2, beta3, beta4) para alpha > 0, en la precisión de mp.mp.dps.

    Para cada raíz se hace primero una pasada en doble precisión (semilla) y después se refina
    en mpmath; en la práctica el encierro garantiza el orden.
    """
    dps = mp.mp.dps
    eps = mp.mpf(10) ** (-(dps - 5))
    tol = mp.mpf(10) ** (-(dps - 3))
    k = kou.to_mp()
    return _cuatro(k, mp.mpf(alpha), eps, tol)


def residuo_max(kou, alpha, rs):
    """max |G(rho_j) - alpha| para rho = (b1, b2, -b3, -b4); para los tests."""
    k = kou.to_mp()
    b1, b2, b3, b4 = rs
    return max(abs(k.G(x) - alpha) for x in (b1, b2, -b3, -b4))

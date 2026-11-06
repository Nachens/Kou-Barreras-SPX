"""Transformadas de Laplace en t de las probabilidades de pasaje (Teo. 4.4 y Prop. 4.7).

L(alpha) = int_0^inf e^{-alpha t} P(X_t >= a, tau_b <= t) dt = A F0(a-b) + B F1(a-b),
con F0(y) = int e^{-alpha t} P(X_t >= y) dt, F1 lo mismo para X_t + xi, xi ~ Exp(eta1).
F0 y F1 son explícitas en las cuatro raíces (fracciones simples de 1/(alpha - G)).
"""
from __future__ import annotations

import mpmath as mp


def pesos_AB(b, rs, eta1):
    """(A_alpha, B_alpha) del Teo. 4.4, ecs. (4.5)-(4.6), con b > 0."""
    b1, b2 = rs[0], rs[1]
    e1, e2_ = mp.exp(-b * b1), mp.exp(-b * b2)
    A = (eta1 - b1) / (b2 - b1) * e1 + (b2 - eta1) / (b2 - b1) * e2_
    B = (eta1 - b1) * (b2 - eta1) / (eta1 * (b2 - b1)) * (e1 - e2_)
    return A, B


def _residuos(rho, num, sigma):
    out = []
    for j, r in enumerate(rho):
        den = sigma ** 2 / 2
        for k, s in enumerate(rho):
            if k != j:
                den *= (r - s)
        out.append(num(r) / den)
    return out


def F_res(y, rs, num, sigma):
    """F(y) = int_0^inf e^{-alpha t} P(Z >= y) dt para num = num0 (F0) o num1 (F1).

    rs = (b1, b2, b3, b4); rho = (b1, b2, -b3, -b4).
    """
    b1, b2, b3, b4 = rs
    rho = [b1, b2, -b3, -b4]
    R = _residuos(rho, num, sigma)
    pos = [(Rj, r) for Rj, r in zip(R, rho) if r > 0]
    neg = [(Rj, r) for Rj, r in zip(R, rho) if r < 0]
    if y >= 0:
        return -sum(Rj * mp.exp(-r * y) / r for Rj, r in pos)
    return (sum(Rj * (mp.exp(-r * y) - 1) / r for Rj, r in neg)
            - sum(Rj / r for Rj, r in pos))


def F0(y, rs, kou):
    e1, e2 = kou.eta1, kou.eta2
    return F_res(y, rs, lambda r: (e1 - r) * (e2 + r), kou.sigma)


def F1(y, rs, kou):
    e1, e2 = kou.eta1, kou.eta2
    return F_res(y, rs, lambda r: e1 * (e2 + r), kou.sigma)


def lt_conjunta(a, b, kou, rs):
    """Transformada de P(X_t >= a, tau_b <= t) (b > 0). Devuelve (L, A, B).

    Vale para cualquier a (la Prop. 4.7 no pide a <= b); Teo. B.35 sí.
    """
    k = kou.to_mp()
    a, b = mp.mpf(a), mp.mpf(b)
    A, B = pesos_AB(b, rs, k.eta1)
    y = a - b
    return A * F0(y, rs, k) + B * F1(y, rs, k), A, B


def lt_marginal(y, kou, rs):
    """Transformada de P(X_t >= y)."""
    return F0(mp.mpf(y), rs, kou.to_mp())


def lt_cruce(b, kou, rs, alpha):
    """(A_alpha/alpha, B_alpha/alpha): transformadas de P(tau<=t, Z=0) y P(tau<=t, Z>0)."""
    A, B = pesos_AB(mp.mpf(b), rs, kou.to_mp().eta1)
    return A / alpha, B / alpha

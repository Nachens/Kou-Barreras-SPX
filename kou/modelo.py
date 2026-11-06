"""Modelo de Kou: X_t = gamma t + sigma W_t + sum_{i<=N_t} Y_i, Y doble exponencial.

Notación de la tesis (Cap. 2): (gamma, sigma, lambda, p, eta1, eta2), q = 1 - p.
Las funciones sirven con float, numpy (arrays complejos) y mpmath: usan solo + - * /.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mpmath as mp


@dataclass(frozen=True)
class Kou:
    gamma: object
    sigma: object
    lam: object
    p: object
    eta1: object
    eta2: object

    @property
    def q(self):
        return 1 - self.p

    # ---- exponente de Laplace y de Lévy-Khintchine ----
    def G(self, beta):
        """G(beta) = log E[exp(beta X_1)], para -eta2 < beta < eta1."""
        p, q = self.p, self.q
        return (self.gamma * beta + self.sigma ** 2 * beta ** 2 / 2
                + self.lam * (p * self.eta1 / (self.eta1 - beta)
                              + q * self.eta2 / (self.eta2 + beta) - 1))

    def dG(self, beta):
        p, q = self.p, self.q
        return (self.gamma + self.sigma ** 2 * beta
                + self.lam * (p * self.eta1 / (self.eta1 - beta) ** 2
                              - q * self.eta2 / (self.eta2 + beta) ** 2))

    def Psi(self, u):
        """Psi(u) = G(iu), con u real o complejo (Prop. 2.7 de la tesis)."""
        return self.G(1j * u)

    def phi(self, u, T):
        """Función característica de X_T."""
        import numpy as np
        return np.exp(T * self.Psi(u))

    def to_mp(self):
        return Kou(*[mp.mpf(x) for x in (self.gamma, self.sigma, self.lam, self.p,
                                          self.eta1, self.eta2)])

    def to_float(self):
        return Kou(*[float(x) for x in (self.gamma, self.sigma, self.lam, self.p,
                                         self.eta1, self.eta2)])

    # ---- transformaciones ----
    def esscher(self, theta) -> "Kou":
        """Transformada de Esscher de parámetro theta (Prop. 2.9 / B.3)."""
        p, q, e1, e2 = self.p, self.q, self.eta1, self.eta2
        a1 = p * e1 / (e1 - theta)
        a2 = q * e2 / (e2 + theta)
        return Kou(self.gamma + self.sigma ** 2 * theta, self.sigma, self.lam * (a1 + a2),
                   a1 / (a1 + a2), e1 - theta, e2 + theta)

    def negar(self) -> "Kou":
        """X' = -X es Kou con (-gamma, sigma, lambda, q, eta2, eta1) (Obs. 4.9)."""
        return Kou(-self.gamma, self.sigma, self.lam, self.q, self.eta2, self.eta1)

    def con_gamma(self, gamma) -> "Kou":
        return replace(self, gamma=gamma)


def gamma_Q(r, delta, sigma, lam, p, eta1, eta2):
    """gamma bajo Q con G^Q(1) = r - delta (Prop. 2.17). Exige eta1 > 1."""
    if not eta1 > 1:
        raise ValueError(f"eta1 debe ser > 1 (eta1={eta1})")
    q = 1 - p
    return (r - delta) - sigma ** 2 / 2 - lam * (p * eta1 / (eta1 - 1) + q * eta2 / (eta2 + 1) - 1)


def kou_Q(r, delta, sigma, lam, p, eta1, eta2) -> Kou:
    return Kou(gamma_Q(r, delta, sigma, lam, p, eta1, eta2), sigma, lam, p, eta1, eta2)

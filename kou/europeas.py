"""Opciones europeas bajo Kou (§5.2.1 de la tesis).

* CarrMadan: ec. (5.2), con la variante de Cont-Tankov que resta Black-Scholes con sigma_bar
  (CT ec. (11.21)); FFT + interpolación cúbica (la que usa la calibración) y cuadratura adaptativa
  en cada K (control). Precios en unidades de S0.
* prob_gil_pelaez / call_gil_pelaez: P(X_T >= k) por Gil-Pelaez y la call como
  S0 e^{-delta T} Q1(X_T >= k) - K e^{-rT} Q(X_T >= k) (Kou, Teo. 2 / Cor. 2; control, nota 9).
Las puts salen de la paridad (5.1).
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from .modelo import Kou


def paridad_put(call, S0, K, T, r, delta):
    return call - S0 * np.exp(-delta * T) + K * np.exp(-r * T)


class CarrMadan:
    def __init__(self, kouQ: Kou, T, r, delta, N=2 ** 15, Delta=0.2, sigma_bar="anula"):
        self.kou, self.T, self.r, self.delta = kouQ.to_float(), float(T), float(r), float(delta)
        self.g = (self.r - self.delta) * self.T
        self.N, self.Delta = int(N), float(Delta)
        k = self.kou
        # G(1) = r - delta (medida martingala)
        if abs(k.G(1.0) - (self.r - self.delta)) > 1e-8:
            raise ValueError("kouQ no cumple G(1) = r - delta")
        if sigma_bar == "anula":
            s2 = 2 * (k.dG(1.0) - k.G(1.0))      # anula zeta~(0), ver docstring de _zeta
            self.sigma_bar = float(np.sqrt(s2)) if s2 > 1e-8 else k.sigma
        else:
            self.sigma_bar = float(sigma_bar)
        self._grid = None

    # ---- transformada ----
    def _zeta(self, v):
        """zeta~_T(v) = e^{-rT}(phi(v-i) - phi_BS(v-i)) / (iv(1+iv)); en v=0 vale 0.

        Con sigma_bar^2 = 2(G'(1) - G(1)) el término lineal de phi(v-i) - phi_BS(v-i) se anula
        (E[X e^X] coincide), y zeta~(0) = 0.
        """
        T, k = self.T, self.kou
        b = 1.0 + 1j * v
        phi = np.exp(T * k.G(b))
        mu = (self.r - self.delta) - self.sigma_bar ** 2 / 2
        phib = np.exp(T * (mu * b + self.sigma_bar ** 2 * b ** 2 / 2))
        den = 1j * v * (1 + 1j * v)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.exp(-self.r * T) * (phi - phib) / den
        z = np.where(v == 0, 0.0, z)
        return z

    def _bs_unit(self, kk):
        """Black-Scholes con sigma_bar, en unidades de S0, k = log(K/S0)."""
        T, s = self.T, self.sigma_bar
        sT = s * np.sqrt(T)
        dp = (-kk + (self.r - self.delta + s * s / 2) * T) / sT
        dm = dp - sT
        return np.exp(-self.delta * T) * norm.cdf(dp) - np.exp(kk - self.r * T) * norm.cdf(dm)

    # ---- FFT ----
    def _fft(self):
        if self._grid is None:
            N, D = self.N, self.Delta
            j = np.arange(N)
            v = (j - N // 2) * D
            zeta = self._zeta(v)
            L = 2 * np.pi / D
            d = L / N
            sgn = (-1.0) ** (j - N // 2)                  # e^{i(j-N/2) pi}
            F = np.fft.fft(zeta * sgn)
            m = np.arange(N)
            z = (D / (2 * np.pi)) * ((-1.0) ** m) * F
            kgrid = -L / 2 + m * d
            self._grid = (kgrid, z.real)
        return self._grid

    def call_unit(self, kk):
        """Call/S0 en k = log(K/S0), por FFT + spline cúbico en una ventana."""
        kgrid, z = self._fft()
        kk = np.atleast_1d(np.asarray(kk, float))
        lo, hi = kk.min() - 0.05, kk.max() + 0.05
        sel = (kgrid >= lo) & (kgrid <= hi)
        spl = CubicSpline(kgrid[sel], z[sel])
        return spl(kk) + self._bs_unit(kk)

    def call(self, S0, K):
        K = np.atleast_1d(np.asarray(K, float))
        return S0 * self.call_unit(np.log(K / S0))

    def put(self, S0, K):
        K = np.atleast_1d(np.asarray(K, float))
        return paridad_put(self.call(S0, K), S0, K, self.T, self.r, self.delta)

    # ---- cuadratura adaptativa (control) ----
    def _vmax(self):
        s = min(self.kou.sigma, self.sigma_bar)
        return float(np.sqrt(2 * 50 / (s * s * self.T)))

    def call_unit_cuadratura(self, kk):
        kk = np.atleast_1d(np.asarray(kk, float))
        vmax = self._vmax()
        out = []
        for k_ in kk:
            f = lambda v: (np.exp(-1j * v * k_) * self._zeta(np.array([v]))[0]).real
            pts = np.linspace(0, vmax, max(8, int(vmax / 5)))
            val = sum(integrate.quad(f, a, b, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
                      for a, b in zip(pts[:-1], pts[1:]))
            out.append(val / np.pi)
        return np.array(out) + self._bs_unit(kk)

    def call_cuadratura(self, S0, K):
        K = np.atleast_1d(np.asarray(K, float))
        return S0 * self.call_unit_cuadratura(np.log(K / S0))


# --------------------------------------------------------------------- Gil-Pelaez
def prob_gil_pelaez(kou: Kou, T, kk, theta=0.0):
    """P(X_T >= kk) bajo Kou (Esscher de parámetro theta: Q_theta), por Gil-Pelaez.

    phi_theta(u) = exp(T (G(theta + iu) - G(theta))).
    """
    k = kou.to_float()
    G0 = k.G(theta)
    s = k.sigma
    umax = float(np.sqrt(2 * 50 / (s * s * T)))
    f = lambda u: (np.exp(-1j * u * kk) * np.exp(T * (k.G(theta + 1j * u) - G0))).imag / u
    pts = np.linspace(1e-12, umax, max(8, int(umax / 5)))
    val = sum(integrate.quad(f, a, b, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
              for a, b in zip(pts[:-1], pts[1:]))
    return 0.5 + val / np.pi


def call_gil_pelaez(kouQ: Kou, S0, K, T, r, delta):
    """S0 e^{-delta T} Q1(X_T >= k) - K e^{-rT} Q(X_T >= k), k = log(K/S0)."""
    kk = np.log(K / S0)
    P2 = prob_gil_pelaez(kouQ, T, kk, 0.0)
    P1 = prob_gil_pelaez(kouQ, T, kk, 1.0)
    return S0 * np.exp(-delta * T) * P1 - K * np.exp(-r * T) * P2

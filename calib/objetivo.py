"""Función objetivo de la calibración (§5.3.1 de la tesis).

R(theta) = sum_i w_i (C_i(theta) - C_i^mid)^2 sobre las opciones OTM que quedan (la put OTM como put).
Pesos: (a) w_i = 1/(ask_i - bid_i)^2, o (b) w_i = 1/V_i^2 con V_i la vega de Black-Scholes en la
volatilidad implícita de mercado. gamma no es libre: lo fija G^Q(1) = r - delta.
Reparametrización u = (log sigma, log lambda, logit p, log(eta1 - 1), log eta2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bs import formulas as bsf
from bs.vol_implicita import vol_implicita
from kou.europeas import CarrMadan, paridad_put
from kou.modelo import kou_Q

NOMBRES = ["sigma", "lam", "p", "eta1", "eta2"]
# cotas por defecto (A3: en la práctica vienen de la clave `cotas:` del YAML); en theta:
# sigma in [0,02; 1,5], lambda in [1e-3; 40], logit p in [-6; 6], eta1 - 1 in [0,05; 300], eta2 in [0,5; 300]
COTAS_DEFECTO = {"sigma": [0.02, 1.5], "lam": [1e-3, 40.0], "logit_p": [-6.0, 6.0],
                 "eta1_menos_1": [0.05, 300.0], "eta2": [0.5, 300.0]}


def cotas_u(cotas=None):
    """Cotas de la reparametrización u = (log sigma, log lambda, logit p, log(eta1 - 1), log eta2)."""
    c = cotas or COTAS_DEFECTO
    lo = np.array([np.log(c["sigma"][0]), np.log(c["lam"][0]), c["logit_p"][0], np.log(c["eta1_menos_1"][0]),
                   np.log(c["eta2"][0])])
    hi = np.array([np.log(c["sigma"][1]), np.log(c["lam"][1]), c["logit_p"][1], np.log(c["eta1_menos_1"][1]),
                   np.log(c["eta2"][1])])
    return lo, hi


U_MIN, U_MAX = cotas_u()


def theta_a_u(th):
    s, lam, p, e1, e2 = th
    return np.array([np.log(s), np.log(lam), np.log(p / (1 - p)), np.log(e1 - 1), np.log(e2)])


def u_a_theta(u):
    s, lam, lp, le1, le2 = u
    return np.array([np.exp(s), np.exp(lam), 1 / (1 + np.exp(-lp)), 1 + np.exp(le1), np.exp(le2)])


@dataclass
class Problema:
    df: object           # strike, tipo (OTM), bid, ask, mid
    S0: float
    T: float
    r: float
    delta: float
    pesos: str = "b"
    N_fft: int = 2 ** 15
    cotas: dict | None = None

    def __post_init__(self):
        self.u_min, self.u_max = cotas_u(self.cotas)
        d = self.df
        self.K = d.strike.to_numpy(float)
        self.tipo = d.tipo.to_numpy()
        self.bid, self.ask, self.mid = (d[c].to_numpy(float) for c in ("bid", "ask", "mid"))
        self.es_put = self.tipo == "P"
        self.iv_mkt = np.array([vol_implicita(m, t, self.S0, k, self.T, self.r, self.delta)
                                for m, t, k in zip(self.mid, self.tipo, self.K)])
        self.vega = np.array([bsf.vega(self.S0, k, self.T, self.r, self.delta, s) if np.isfinite(s) else np.nan
                              for k, s in zip(self.K, self.iv_mkt)])
        spr = np.maximum(self.ask - self.bid, 1e-12)
        if self.pesos == "a":
            w = 1.0 / spr ** 2
        elif self.pesos == "b":
            w = np.where(np.isfinite(self.vega) & (self.vega > 0), 1.0 / np.where(self.vega > 0, self.vega, 1) ** 2,
                         1.0 / spr ** 2)       # sin vol implícita: cae a (a) y se cuenta
        else:
            raise ValueError("pesos debe ser 'a' o 'b'")
        self.w = w
        self.n_sin_iv = int((~np.isfinite(self.iv_mkt)).sum())
        self.eps0_sq = float(np.sum(self.w * spr ** 2))

    # ---- precios del modelo ----
    def kou_Q(self, th):
        s, lam, p, e1, e2 = th
        return kou_Q(self.r, self.delta, s, lam, p, e1, e2)

    def precios_modelo(self, th, N=None):
        cm = CarrMadan(self.kou_Q(th), self.T, self.r, self.delta, N=N or self.N_fft)
        C = cm.call(self.S0, self.K)
        return np.where(self.es_put, paridad_put(C, self.S0, self.K, self.T, self.r, self.delta), C)

    def precios_bs(self, sigma):
        return np.where(self.es_put, bsf.put(self.S0, self.K, self.T, self.r, self.delta, sigma),
                        bsf.call(self.S0, self.K, self.T, self.r, self.delta, sigma))

    # ---- objetivo ----
    def residuos_theta(self, th, N=None):
        try:
            c = self.precios_modelo(th, N)
        except Exception:
            return np.full(len(self.K), 1e6)
        if not np.all(np.isfinite(c)):
            return np.full(len(self.K), 1e6)
        return np.sqrt(self.w) * (c - self.mid)

    def residuos_u(self, u):
        return self.residuos_theta(u_a_theta(np.clip(u, self.u_min, self.u_max)))

    def R(self, th, N=None):
        r = self.residuos_theta(th, N)
        return float(r @ r)

    def residuos_bs(self, sigma):
        return np.sqrt(self.w) * (self.precios_bs(sigma) - self.mid)

    def cerca_de_cota(self, th, tol=0.01):
        """Para cada parámetro, si theta está a menos de tol (relativo) de una cota. p se mide en p, eta1 en eta1 - 1."""
        s, lam, p, e1, e2 = th
        lo, hi = self.u_min, self.u_max
        val = [s, lam, p, e1 - 1, e2]
        cl = [np.exp(lo[0]), np.exp(lo[1]), 1 / (1 + np.exp(-lo[2])), np.exp(lo[3]), np.exp(lo[4])]
        ch = [np.exp(hi[0]), np.exp(hi[1]), 1 / (1 + np.exp(-hi[2])), np.exp(hi[3]), np.exp(hi[4])]
        return {n: bool(abs(v - a) <= tol * abs(a) or abs(v - b) <= tol * abs(b))
                for n, v, a, b in zip(NOMBRES, val, cl, ch)}

    # ---- métricas por modelo ----
    def metricas(self, precios):
        iv = np.array([vol_implicita(p_, t, self.S0, k, self.T, self.r, self.delta)
                       for p_, t, k in zip(precios, self.tipo, self.K)])
        ok = np.isfinite(iv) & np.isfinite(self.iv_mkt)
        e = iv[ok] - self.iv_mkt[ok]
        return dict(rmse_vol=float(np.sqrt(np.mean(e ** 2))) if ok.any() else np.nan,
                    max_vol=float(np.max(np.abs(e))) if ok.any() else np.nan,
                    frac_en_bidask=float(np.mean((precios >= self.bid) & (precios <= self.ask))),
                    n_sin_vol=int((~ok).sum()), n=int(len(precios)))

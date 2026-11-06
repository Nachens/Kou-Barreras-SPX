"""Optimización multi-start (§5.3.3 de la tesis).

least_squares (TRF) sobre los residuos sqrt(w_i)(C_i(theta) - C_i^mid), con las cotas de `prob` (clave
`cotas:` del YAML, A3); 64 puntos Sobol de arranque en la caja
sigma in [0,03; 0,8], lambda in [0,05; 10], p in [0,05; 0,95], eta1 in [1,5; 100], eta2 in [1; 60]
(sigma, lambda y los eta se muestrean en escala logarítmica; p, lineal). Se agrupan los mínimos
locales por distancia en u (< 0,05) y se conservan los de R <= factor_parecido * R_min.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import qmc

from .objetivo import NOMBRES, Problema, theta_a_u, u_a_theta

CAJA = dict(sigma=(0.03, 0.8), lam=(0.05, 10.0), p=(0.05, 0.95), eta1=(1.5, 100.0), eta2=(1.0, 60.0))


def puntos_sobol(n, semilla=0):
    s = qmc.Sobol(d=5, scramble=True, seed=semilla).random(n)
    lg = lambda lohi, x: np.exp(np.log(lohi[0]) + x * (np.log(lohi[1]) - np.log(lohi[0])))
    th = np.column_stack([lg(CAJA["sigma"], s[:, 0]), lg(CAJA["lam"], s[:, 1]),
                          CAJA["p"][0] + s[:, 2] * (CAJA["p"][1] - CAJA["p"][0]),
                          lg(CAJA["eta1"], s[:, 3]), lg(CAJA["eta2"], s[:, 4])])
    return th


def ajustar_desde(prob: Problema, u0, libres=None, fijos=None, max_nfev=150):
    """Minimiza R desde u0. `fijos`: dict {indice_u: valor} (para el perfil)."""
    U_MIN, U_MAX = prob.u_min, prob.u_max
    u0 = np.clip(np.asarray(u0, float), U_MIN + 1e-6, U_MAX - 1e-6)
    libres = [i for i in range(5) if not fijos or i not in fijos]

    def completo(v):
        u = u0.copy()
        if fijos:
            for i, val in fijos.items():
                u[i] = val
        u[libres] = v
        return u

    f = lambda v: prob.residuos_u(completo(v))
    try:
        sol = least_squares(f, u0[libres], bounds=(U_MIN[libres], U_MAX[libres]), method="trf",
                            x_scale=1.0, max_nfev=max_nfev, xtol=1e-10, ftol=1e-12, gtol=1e-10)
        u = completo(sol.x)
        return u, float(2 * sol.cost), int(sol.nfev)
    except Exception:
        return completo(u0[libres]), float("inf"), 0


def _una(args):
    prob, th0 = args
    u, R, nfev = ajustar_desde(prob, theta_a_u(th0))
    return u, R, nfev


def agrupar(us, Rs, tol=0.05):
    """Agrupa por distancia en u; devuelve lista (u, R) de representantes ordenada por R."""
    orden = np.argsort(Rs)
    rep = []
    for i in orden:
        if not np.isfinite(Rs[i]):
            continue
        if all(np.linalg.norm(us[i] - u_) >= tol for u_, _ in rep):
            rep.append((us[i], Rs[i]))
    return rep


def calibrar_kou(prob: Problema, n_inicios=64, factor_parecido=1.2, semilla=0, jobs=1):
    th0 = puntos_sobol(n_inicios, semilla)
    if jobs and jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_una, [(prob, t) for t in th0]))
    else:
        res = [_una((prob, t)) for t in th0]
    us = np.array([r[0] for r in res])
    Rs = np.array([r[1] for r in res])
    rep = agrupar(us, Rs)
    Rmin = rep[0][1]
    conservados = [(u, R) for u, R in rep if R <= factor_parecido * Rmin]
    minimos = [dict(theta=u_a_theta(u).tolist(), u=u.tolist(), R=float(R),
                    cerca_de_cota=prob.cerca_de_cota(u_a_theta(u))) for u, R in conservados]
    en_cota = {n: sum(m["cerca_de_cota"][n] for m in minimos) for n in NOMBRES}
    return dict(
        optimo_u=rep[0][0], optimo_theta=u_a_theta(rep[0][0]), R_min=Rmin, minimos=minimos,
        minimos_cerca_de_cota=en_cota, n_distintos=len(rep), n_inicios=n_inicios, R_todos=Rs.tolist())


def calibrar_bs(prob: Problema):
    """sigma_LS (mínimos cuadrados con los mismos pesos) y sigma_ATM (vol implícita del strike más cercano a F)."""
    f = lambda s: float(np.sum(prob.residuos_bs(s) ** 2))
    sol = minimize_scalar(f, bounds=(0.01, 3.0), method="bounded", options=dict(xatol=1e-10))
    F = prob.S0 * np.exp((prob.r - prob.delta) * prob.T)
    ok = np.isfinite(prob.iv_mkt)
    i = np.argmin(np.where(ok, np.abs(prob.K - F), np.inf))
    return dict(sigma_LS=float(sol.x), R_LS=float(sol.fun), sigma_ATM=float(prob.iv_mkt[i]), K_atm=float(prob.K[i]),
                i_atm=int(i))

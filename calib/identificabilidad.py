"""Identificabilidad (§5.3.4 de la tesis): SVD del jacobiano, perfil en lambda y conjunto C."""
from __future__ import annotations

import numpy as np

from .objetivo import Problema, u_a_theta
from .optimizar import ajustar_desde


def jacobiano(prob: Problema, u, h=1e-5):
    u = np.asarray(u, float)
    cols = []
    for i in range(5):
        e = np.zeros(5)
        e[i] = h
        cols.append((prob.residuos_u(u + e) - prob.residuos_u(u - e)) / (2 * h))
    return np.column_stack(cols)


def svd_jacobiano(prob: Problema, u):
    J = jacobiano(prob, u)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    return dict(valores=s.tolist(), vectores=Vt.tolist(), cociente=float(s[0] / s[-1]) if s[-1] > 0 else float("inf"))


def perfil_lambda(prob: Problema, u_opt, n=25, lam_min=0.05, lam_max=40.0, reinicios=4, semilla=0, lam_extra=()):
    """R_perf(lambda): en cada lambda de una grilla geométrica (más el óptimo) se minimiza en los otros 4.

    lam_max es por defecto la cota superior de lambda de la optimización (A2), así el perfil cubre el
    mismo rango que los mínimos locales. `lam_extra`: lambdas que se agregan a la grilla (los de mínimos
    conservados fuera de [lam_min, lam_max], para que el rango del perfil contenga el de todos).
    Se arranca del punto vecino (continuación hacia ambos lados desde el óptimo) y de `reinicios`
    puntos aleatorios; se conserva el mejor.
    """
    U_MIN, U_MAX = prob.u_min, prob.u_max
    rng = np.random.default_rng(semilla)
    lam_opt = float(np.exp(u_opt[1]))
    extra = [l for l in lam_extra if l < lam_min or l > lam_max]
    grid = np.unique(np.concatenate([np.geomspace(lam_min, lam_max, n), [lam_opt], extra]))
    i0 = int(np.argmin(np.abs(grid - lam_opt)))
    out = {}

    def resolver(lam, u_ini):
        mejor = (None, np.inf)
        cand = [np.asarray(u_ini, float)]
        for _ in range(reinicios):
            c = np.asarray(u_ini, float) + rng.normal(0, 0.5, 5)
            cand.append(np.clip(c, U_MIN, U_MAX))
        for c in cand:
            u, R, _ = ajustar_desde(prob, c, fijos={1: np.log(lam)})
            if R < mejor[1]:
                mejor = (u, R)
        return mejor

    for rango in (range(i0, len(grid)), range(i0, -1, -1)):
        u_prev = np.asarray(u_opt, float)
        for i in rango:
            if i in out and i == i0:
                u_prev = out[i][0]
                continue
            u, R = resolver(grid[i], u_prev)
            out[i] = (u, R)
            u_prev = u
    return [dict(lam=float(grid[i]), R=float(out[i][1]), theta=u_a_theta(out[i][0]).tolist(),
                 u=out[i][0].tolist()) for i in sorted(out)]


def en_C(prob: Problema, th, R_min, criterio="eps0", kappa=1.2, R=None):
    """¿theta está en C según el criterio? (eps0, bidask o kappa)."""
    th = np.asarray(th, float)
    if criterio == "bidask":
        c = prob.precios_modelo(th)
        return bool(np.all((c >= prob.bid) & (c <= prob.ask)))
    R = prob.R(th) if R is None else R
    if criterio == "eps0":
        return R <= prob.eps0_sq
    if criterio == "kappa":
        return R <= kappa ** 2 * R_min
    raise ValueError("criterio_C: eps0 | bidask | kappa")


def conjunto_C(prob: Problema, candidatos, R_min, criterio="eps0", kappa=1.2):
    """Lista de candidatos (dict con theta y R) que pertenecen a C. Vacío => se reporta, no se cambia el criterio."""
    return [c for c in candidatos if en_C(prob, c["theta"], R_min, criterio, kappa, c.get("R"))]

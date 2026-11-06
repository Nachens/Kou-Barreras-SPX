"""Monte Carlo exacto bajo Kou (puente browniano entre saltos) para controlar las barreras.

Los kmax tiempos uniformes de cada trayectoria se **enmascaran** (los que sobran van en T) y recién
después se ordenan. Ordenar antes de enmascarar da los N menores de kmax uniformes y concentra los
saltos al principio (error de una versión anterior). `tiempos_de_salto` expone
los tiempos para el test de Kolmogorov-Smirnov contra la uniforme.
"""
from __future__ import annotations

import numpy as np

from .modelo import Kou


def simular(kou: Kou, T, n, rng, con_tiempos=False):
    k = kou.to_float()
    N = rng.poisson(k.lam * T, n)
    kmax = max(int(N.max()), 1)
    mask = np.arange(kmax)[None, :] < N[:, None]
    U = rng.uniform(0, T, (n, kmax))
    times = np.sort(np.where(mask, U, T), axis=1)
    J = np.where(rng.uniform(size=(n, kmax)) < k.p, rng.exponential(1 / k.eta1, (n, kmax)),
                 -rng.exponential(1 / k.eta2, (n, kmax)))
    J = np.where(mask, J, 0.0)
    x = np.zeros(n)
    mx = np.zeros(n)
    mn = np.zeros(n)
    tp = np.zeros(n)
    s = k.sigma
    for j in range(kmax + 1):
        tj = times[:, j] if j < kmax else np.full(n, T)
        dt = tj - tp
        x1 = x + k.gamma * dt + s * np.sqrt(dt) * rng.standard_normal(n)
        d = x1 - x
        # máximo y mínimo del puente browniano entre x y x1 (muestreo exacto)
        e1 = -np.log(rng.uniform(size=n))
        e2 = -np.log(rng.uniform(size=n))
        Mx = 0.5 * (x + x1 + np.sqrt(d * d + 2 * s * s * dt * e1))
        Mn = 0.5 * (x + x1 - np.sqrt(d * d + 2 * s * s * dt * e2))
        mx = np.maximum(mx, Mx)
        mn = np.minimum(mn, Mn)
        x = x1
        if j < kmax:
            x = x + J[:, j]
            mx = np.maximum(mx, x)
            mn = np.minimum(mn, x)
        tp = tj
    out = dict(x=x, mx=mx, mn=mn)
    if con_tiempos:
        out["N"], out["times"] = N, times
    return out


def tiempos_de_salto(kou: Kou, T, n, seed=0):
    """Todos los tiempos de salto simulados (vector), para el test KS contra Uniforme(0, T)."""
    r = simular(kou, T, n, np.random.default_rng(seed), con_tiempos=True)
    N, times = r["N"], r["times"]
    mask = np.arange(times.shape[1])[None, :] < N[:, None]
    return times[mask]


def precios_mc(kouQ: Kou, S0, K, T, r, delta, L_up, L_down, n, seed=0, chunk=500_000):
    """Medias y errores estándar: call, put, up-and-out call (L_up) y down-and-out put (L_down)."""
    rng = np.random.default_rng(seed)
    bU, bD = np.log(L_up / S0), np.log(L_down / S0)
    df = np.exp(-r * T)
    acc = [[], [], [], []]
    hecho = 0
    while hecho < n:
        m = min(chunk, n - hecho)
        s = simular(kouQ, T, m, rng)
        ST = S0 * np.exp(s["x"])
        acc[0].append(df * np.maximum(ST - K, 0))
        acc[1].append(df * np.maximum(K - ST, 0))
        acc[2].append(df * np.maximum(ST - K, 0) * (s["mx"] < bU))
        acc[3].append(df * np.maximum(K - ST, 0) * (s["mn"] > bD))
        hecho += m
    out = {}
    for nombre, a in zip(("call", "put", "uoc", "dop"), acc):
        v = np.concatenate(a)
        out[nombre] = (float(v.mean()), float(v.std() / np.sqrt(len(v))))
    return out


def prob_cruce_mc(kou: Kou, T, b, a, n, seed=0, chunk=500_000):
    """P(tau_b <= T) y P(tau_b <= T, X_T >= a) por simulación (b > 0)."""
    rng = np.random.default_rng(seed)
    c = j = 0.0
    hecho = 0
    while hecho < n:
        m = min(chunk, n - hecho)
        s = simular(kou, T, m, rng)
        cr = s["mx"] >= b
        c += cr.sum()
        j += (cr & (s["x"] >= a)).sum()
        hecho += m
    p1, p2 = c / n, j / n
    return (p1, float(np.sqrt(p1 * (1 - p1) / n))), (p2, float(np.sqrt(p2 * (1 - p2) / n)))


def precios_mc_barreras(kouQ: Kou, S0, K, T, r, delta, L_up, L_down, n, seed=0, chunk=500_000):
    """Como precios_mc, pero con varias barreras evaluadas sobre las mismas trayectorias.

    Devuelve dict(uoc=[(media, error estándar), ...] en el orden de L_up, dop=[...] en el orden de L_down,
    call=(media, ee), put=(media, ee)).
    """
    rng = np.random.default_rng(seed)
    bU = np.log(np.asarray(L_up, float) / S0)
    bD = np.log(np.asarray(L_down, float) / S0)
    df = np.exp(-r * T)
    s1 = {k: np.zeros(n_) for k, n_ in (("uoc", len(bU)), ("dop", len(bD)), ("call", 1), ("put", 1))}
    s2 = {k: np.zeros_like(v) for k, v in s1.items()}
    hecho = 0
    while hecho < n:
        m = min(chunk, n - hecho)
        s = simular(kouQ, T, m, rng)
        ST = S0 * np.exp(s["x"])
        c = df * np.maximum(ST - K, 0)
        p_ = df * np.maximum(K - ST, 0)
        pagos = dict(call=c[:, None], put=p_[:, None], uoc=c[:, None] * (s["mx"][:, None] < bU[None, :]),
                     dop=p_[:, None] * (s["mn"][:, None] > bD[None, :]))
        for k, v in pagos.items():
            s1[k] += v.sum(axis=0)
            s2[k] += (v ** 2).sum(axis=0)
        hecho += m
    out = {}
    for k in s1:
        media = s1[k] / n
        ee = np.sqrt(np.maximum(s2[k] / n - media ** 2, 0) / (n - 1))
        out[k] = [(float(a), float(b)) for a, b in zip(media, ee)]
    out["call"], out["put"] = out["call"][0], out["put"][0]
    return out

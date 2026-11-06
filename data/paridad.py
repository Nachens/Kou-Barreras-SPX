"""Tasa y dividendo por paridad put-call (§5.1.3 de la tesis).

Para cada strike con call y put, C^mid - P^mid = S0 e^{-delta T} - K e^{-rT} = D (F - K), con
D = e^{-rT} y F = S0 e^{(r - delta) T} el precio a término.

Dos fuentes para r (clave `r_fuente`):
  - 'paridad': regresión de dos parámetros sobre K, pendiente = -D y ordenada = S0 e^{-delta T}, así que
    r = -ln(-pendiente)/T y delta = -ln(ordenada/S0)/T;
  - 'externa': r viene de afuera (letra del Tesoro, data/tasa.py), D = e^{-rT} queda fijo y se ajusta
    solo F; después delta = r - log(F/S0)/T.

Pesos (clave `paridad_pesos`): 'ols', 'ponderada' (1/s^2) o 'robusta' (1/s^2 con pérdida soft_l1), con
s = sqrt(s_C^2 + s_P^2) el spread del par (los dos spreads sumados en cuadratura: es la escala del
error de C^mid - P^mid si el error de cada punto medio es proporcional a su spread). En la robusta
el residuo que ve least_squares es (y - modelo) * m / s, con m la mediana de s, y f_scale = m: la
pérdida deja de ser cuadrática cuando el residuo de un par pasa de un spread de par mediano.

Siempre se reporta además la regresión de dos parámetros con ventanas ±5 %, ±10 % y ±15 % en
log(K/S0), con su desvío por bootstrap (los pares se remuestrean con reposición): documenta qué
identifica la paridad (el cociente ordenada/pendiente, o sea F) y qué no (r y delta por separado).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

VENTANAS_DIAGNOSTICO = (0.05, 0.10, 0.15)


def pares(df: pd.DataFrame, S0, ventana):
    """Strikes con call y put y |log(K/S0)| <= ventana: (K, y = C^mid - P^mid, s del par)."""
    c = df[df.tipo == "C"].set_index("strike")
    p = df[df.tipo == "P"].set_index("strike")
    K = np.array(sorted(set(c.index) & set(p.index)))
    K = K[np.abs(np.log(K / S0)) <= ventana]
    if len(K) < 3:
        raise ValueError(f"paridad: solo {len(K)} strikes con call y put en la ventana (±{ventana}); hacen falta ≥ 3")
    y = (c.loc[K, "mid"] - p.loc[K, "mid"]).to_numpy(float)
    sC = (c.loc[K, "ask"] - c.loc[K, "bid"]).to_numpy(float)
    sP = (p.loc[K, "ask"] - p.loc[K, "bid"]).to_numpy(float)
    s = np.maximum(np.sqrt(sC ** 2 + sP ** 2), 1e-12)
    return K.astype(float), y, s


def _pesos(s, pesos):
    if pesos == "ols":
        return np.ones_like(s)
    if pesos in ("ponderada", "robusta"):
        return 1.0 / s ** 2
    raise ValueError("paridad_pesos: ols | ponderada | robusta")


def recta(K, y, s, pesos="ols"):
    """y = ordenada + pendiente K. Devuelve (pendiente, ordenada)."""
    w = _pesos(s, pesos)
    X = np.column_stack([K, np.ones_like(K)])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    if pesos != "robusta":
        return float(coef[0]), float(coef[1])
    # robusta: K centrado para que los dos parámetros tengan escalas parecidas
    Kc = float(np.median(K))
    m = float(np.median(s))
    x0 = np.array([coef[1] + coef[0] * Kc, coef[0]])
    f = lambda x: (y - x[0] - x[1] * (K - Kc)) * m / s
    sol = least_squares(f, x0, loss="soft_l1", f_scale=m, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    c0, c1 = sol.x
    return float(c1), float(c0 - c1 * Kc)


def forward_con_r(K, y, s, r, T, pesos="ols"):
    """Con D = e^{-rT} fijo: y = D (F - K)  =>  z = y/D + K = F + error. Ajusta solo F."""
    D = np.exp(-r * T)
    z = y / D + K
    w = _pesos(s, pesos)
    F0 = float(np.sum(w * z) / np.sum(w))
    if pesos != "robusta":
        return F0
    m = float(np.median(s))
    f = lambda x: (y - D * (x[0] - K)) * m / s
    sol = least_squares(f, [F0], loss="soft_l1", f_scale=m, method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(sol.x[0])


def _R2(y, ajuste, s, pesos):
    w = _pesos(s, pesos)
    res = y - ajuste
    ybar = np.average(y, weights=w)
    return float(1 - np.sum(w * res ** 2) / np.sum(w * (y - ybar) ** 2))


def dos_parametros(K, y, s, S0, T, pesos="ols"):
    pend, ord_ = recta(K, y, s, pesos)
    if pend >= 0 or ord_ <= 0:
        raise ValueError(f"paridad: pendiente {pend:.4g} u ordenada {ord_:.4g} sin sentido (r, delta indefinidos)")
    r = -np.log(-pend) / T
    delta = -np.log(ord_ / S0) / T
    F = ord_ / (-pend)                                  # S0 e^{-delta T} / e^{-rT}
    return dict(r=float(r), delta=float(delta), r_menos_delta=float(r - delta), F=float(F), pendiente=pend,
                ordenada=ord_, R2=_R2(y, ord_ + pend * K, s, pesos))


def estimar_r_delta(df: pd.DataFrame, S0, T, ventana=0.05, pesos="ols"):
    """Regresión de dos parámetros (r_fuente = 'paridad'). Devuelve r, delta, F, R2, residuos y el residuo
    máximo en unidades del spread del par."""
    K, y, s = pares(df, S0, ventana)
    out = dos_parametros(K, y, s, S0, T, pesos)
    res = y - (out["ordenada"] + out["pendiente"] * K)
    out.update(n=int(len(K)), residuos=res.tolist(), strikes=K.tolist(),
               residuo_max_spreads=float(np.max(np.abs(res) / s)), fuente="paridad", pesos=pesos, ventana=ventana)
    return out


def estimar_con_r(df: pd.DataFrame, S0, T, r, ventana=0.10, pesos="robusta"):
    """r_fuente = 'externa': F de la paridad con D = e^{-rT} fijo; delta = r - log(F/S0)/T."""
    K, y, s = pares(df, S0, ventana)
    F = forward_con_r(K, y, s, r, T, pesos)
    D = np.exp(-r * T)
    res = y - D * (F - K)
    delta = r - np.log(F / S0) / T
    return dict(r=float(r), delta=float(delta), r_menos_delta=float(r - delta), F=float(F),
                n=int(len(K)), residuos=res.tolist(), strikes=K.tolist(),
                residuo_max_spreads=float(np.max(np.abs(res) / s)), fuente="externa", pesos=pesos, ventana=ventana,
                R2=_R2(y, D * (F - K), s, pesos))


def diagnostico(df, S0, T, pesos="robusta", ventanas=VENTANAS_DIAGNOSTICO, n_boot=300, semilla=0):
    """Regresión de dos parámetros en cada ventana, con desvío por bootstrap (n_boot réplicas, semilla fija)."""
    out = {}
    for v in ventanas:
        try:
            K, y, s = pares(df, S0, v)
            est = dos_parametros(K, y, s, S0, T, pesos)
        except ValueError as e:
            out[f"{v:.2f}"] = dict(error=str(e))
            continue
        rng = np.random.default_rng(semilla)
        reps = []
        for _ in range(n_boot):
            i = rng.integers(0, len(K), len(K))
            if len(np.unique(K[i])) < 3:
                continue
            try:
                e = dos_parametros(K[i], y[i], s[i], S0, T, pesos)
            except ValueError:
                continue
            reps.append((e["r"], e["delta"], e["r_menos_delta"], e["F"]))
        reps = np.array(reps)
        sd = reps.std(axis=0, ddof=1) if len(reps) > 1 else np.full(4, np.nan)
        out[f"{v:.2f}"] = dict(n=int(len(K)), r=est["r"], delta=est["delta"], r_menos_delta=est["r_menos_delta"],
                               F=est["F"], R2=est["R2"], sd_r=float(sd[0]), sd_delta=float(sd[1]),
                               sd_r_menos_delta=float(sd[2]), sd_F=float(sd[3]), n_boot_validas=int(len(reps)))
    return out


def estimar(df, S0, T, p: dict, r_externa: float | None = None):
    """Punto de entrada de la limpieza: aplica `r_fuente` y agrega el diagnóstico de tres ventanas."""
    ventana = p.get("ventana_paridad", 0.05)
    pesos = p.get("paridad_pesos", "ols")
    fuente = p.get("r_fuente", "paridad")
    if fuente == "paridad":
        est = estimar_r_delta(df, S0, T, ventana, pesos)
    elif fuente == "externa":
        if r_externa is None:
            raise ValueError("r_fuente = 'externa' pide el valor de r (data/tasa.py)")
        est = estimar_con_r(df, S0, T, r_externa, ventana, pesos)
    else:
        raise ValueError("r_fuente: paridad | externa")
    est["diagnostico"] = diagnostico(df, S0, T, pesos, n_boot=p.get("paridad_n_boot", 300),
                                     semilla=p.get("paridad_semilla", 0))
    return est

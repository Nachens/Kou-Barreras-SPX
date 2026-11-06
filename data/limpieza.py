"""Limpieza de la cadena (§5.1.2 de la tesis), con el conteo de eliminadas por filtro (Tabla 5.2).

Orden: (1) bid > 0 y bid <= ask, (2) spread relativo, (3) paridad -> r, delta (no elimina),
(4) moneyness, (5) solo OTM, (6) arbitraje estático sobre las calls (las puts OTM pasadas a call
por paridad): se descarta iterativamente la cotización cuya remoción más reduce la violación total.

Moneyness (clave `moneyness_unidad`): 'log' lee el intervalo `moneyness` en log(K/S0) - (r - delta)T;
'sigma_atm' lo lee en unidades de sigma_atm sqrt(T), x_K = (log(K/S0) - (r - delta)T)/(sigma_atm sqrt(T)),
con sigma_atm la volatilidad implícita (nuestra, con r y delta de la paridad) de la opción OTM con strike
más cercano al precio a término, después de los filtros 1-3.

El filtro (6) se aplica con `arbitraje_estatico_sobre` y se cuenta además cuántas eliminaría el otro
modo (columna aparte de la Tabla 5.2). `bordes` registra, para el strike más alto de las calls OTM y el
más bajo de las puts OTM, qué filtro los fija: el último que los movió.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bs.vol_implicita import vol_implicita

from . import paridad

TOL = 1e-10


def violacion_total(K, C, S0, T, r, delta, modo="mid", bid=None, ask=None):
    """Suma de violaciones de cotas, monotonía y convexidad de una curva de calls ordenada por K."""
    K = np.asarray(K, float)
    base = S0 * np.exp(-delta * T)
    lb = np.maximum(base - K * np.exp(-r * T), 0.0)
    if modo == "bidask":
        lo, hi = np.asarray(bid, float), np.asarray(ask, float)
    else:
        lo = hi = np.asarray(C, float)
    v = np.sum(np.maximum(lb - hi, 0)) + np.sum(np.maximum(lo - base, 0))
    if len(K) >= 2:
        v += np.sum(np.maximum(lo[1:] - hi[:-1], 0))
    if len(K) >= 3:
        w = (K[1:-1] - K[:-2]) / (K[2:] - K[:-2])
        chord_hi = hi[:-2] + (hi[2:] - hi[:-2]) * w
        v += np.sum(np.maximum(lo[1:-1] - chord_hi, 0))
    return v


def arbitraje_estatico(K, bid, ask, S0, T, r, delta, modo="mid"):
    """Índices (en el orden de entrada) que se descartan; K ordenado creciente. bid/ask de CALLS."""
    K, bid, ask = np.asarray(K, float), np.asarray(bid, float), np.asarray(ask, float)
    mid = (bid + ask) / 2
    vivos = list(range(len(K)))
    fuera = []
    f = lambda idx: violacion_total(K[idx], mid[idx], S0, T, r, delta, modo, bid[idx], ask[idx])
    while True:
        v0 = f(vivos)
        if v0 <= TOL * max(1.0, S0 * 1e-3):
            break
        mejor, mejor_v = None, v0
        for i in vivos:
            idx = [j for j in vivos if j != i]
            vi = f(idx)
            if vi < mejor_v - 1e-15:
                mejor, mejor_v = i, vi
        if mejor is None:
            break
        vivos.remove(mejor)
        fuera.append(mejor)
    return fuera


def _bordes_otm(df, F):
    """(strike máximo de las calls con K >= F, strike mínimo de las puts con K < F)."""
    c = df[(df.tipo == "C") & (df.strike >= F)].strike
    p_ = df[(df.tipo == "P") & (df.strike < F)].strike
    return (float(c.max()) if len(c) else np.nan, float(p_.min()) if len(p_) else np.nan)


def sigma_atm(df, S0, T, r, delta):
    """Vol implícita de la OTM con strike más cercano a F (si no tiene, la siguiente más cercana)."""
    F = S0 * np.exp((r - delta) * T)
    otm = df[((df.tipo == "P") & (df.strike < F)) | ((df.tipo == "C") & (df.strike >= F))].copy()
    otm["dist"] = np.abs(otm.strike - F)
    for _, fila in otm.sort_values("dist").iterrows():
        s = vol_implicita(fila["mid"], fila["tipo"], S0, fila["strike"], T, r, delta)
        if np.isfinite(s):
            return float(s), float(fila["strike"]), fila["tipo"]
    raise ValueError("sigma_atm: ninguna opción OTM cerca del precio a término tiene volatilidad implícita")


def limpiar(cadena, p: dict, r_externa: float | None = None):
    """p: umbral_spread_rel, moneyness [mmin, mmax], moneyness_unidad, arbitraje_estatico_sobre, paridad_pesos,
    ventana_paridad, r_fuente. r_externa: la r de data/tasa.py si r_fuente = 'externa'.

    Devuelve (df_final, info). df_final tiene la opción OTM que queda por strike (strike, tipo, bid, ask,
    mid, spread, iv_prov). info tiene las cuentas por filtro, r, delta, el diagnóstico de la paridad,
    sigma_atm, el intervalo de moneyness en log y en strikes, y los bordes de la grilla de strikes.
    """
    S0, T = cadena.S0, cadena.T
    df = cadena.df.copy()
    n0 = len(df)
    cuentas = {}
    etapas = [("listado", df)]

    # (1) bid > 0 y bid <= ask
    cruz = df.bid > df.ask
    nulo = (df.bid <= 0) & ~cruz
    cuentas["bid_nulo"] = int(nulo.sum())
    cuentas["cruzadas"] = int(cruz.sum())
    df = df[~(nulo | cruz)]
    etapas.append(("bid", df))
    # (2) spread relativo
    rel = (df.ask - df.bid) / df.mid
    cuentas["spread"] = int((rel > p["umbral_spread_rel"]).sum())
    df = df[rel <= p["umbral_spread_rel"]]
    etapas.append(("spread", df))
    # (3) paridad
    par = paridad.estimar(df, S0, T, p, r_externa)
    r, delta = par["r"], par["delta"]
    F = S0 * np.exp((r - delta) * T)
    s_atm, K_atm, tipo_atm = sigma_atm(df, S0, T, r, delta)
    # (4) moneyness
    m = np.log(df.strike / S0) - (r - delta) * T
    mmin, mmax = p["moneyness"]
    unidad = p.get("moneyness_unidad", "log")
    if unidad == "sigma_atm":
        esc = s_atm * np.sqrt(T)
    elif unidad == "log":
        esc = 1.0
    else:
        raise ValueError("moneyness_unidad: log | sigma_atm")
    ok = (m >= mmin * esc) & (m <= mmax * esc)
    cuentas["moneyness"] = int((~ok).sum())
    df = df[ok]
    etapas.append(("moneyness", df))
    # (5) solo OTM: put si K < F, call si K > F (K = F: call)
    otm = ((df.tipo == "P") & (df.strike < F)) | ((df.tipo == "C") & (df.strike >= F))
    cuentas["no_OTM"] = int((~otm).sum())
    df = df[otm].sort_values("strike").reset_index(drop=True)
    # (6) arbitraje estático sobre las calls equivalentes
    adj = np.where(df.tipo == "P", S0 * np.exp(-delta * T) - df.strike * np.exp(-r * T), 0.0)
    cb, ca = df.bid.to_numpy() + adj, df.ask.to_numpy() + adj
    modo = p.get("arbitraje_estatico_sobre", "mid")
    otro = "mid" if modo == "bidask" else "bidask"
    fuera = arbitraje_estatico(df.strike.to_numpy(), cb, ca, S0, T, r, delta, modo)
    fuera_otro = arbitraje_estatico(df.strike.to_numpy(), cb, ca, S0, T, r, delta, otro)
    cuentas["arbitraje"] = len(fuera)
    df = df.drop(index=fuera).reset_index(drop=True)
    etapas.append(("arbitraje", df))
    df["spread"] = df.ask - df.bid

    # qué filtro fija cada borde: el último que lo movió
    bordes = {}
    seq = [(n, _bordes_otm(d, F)) for n, d in etapas]
    for j, lado in ((0, "call_K_max"), (1, "put_K_min")):
        quien = "listado"
        for (n0_, b0), (n1, b1) in zip(seq[:-1], seq[1:]):
            if not (np.isnan(b0[j]) and np.isnan(b1[j])) and b0[j] != b1[j]:
                quien = n1
        bordes[lado] = dict(K=seq[-1][1][j], filtro=quien, por_etapa={n: b[j] for n, b in seq})

    info = dict(n_inicial=n0, n_final=int(len(df)), cuentas=cuentas, r=r, delta=delta, paridad=par, F=float(F),
                strikes_otm_min=float(df.strike.min()), strikes_otm_max=float(df.strike.max()),
                sigma_atm=s_atm, K_atm=K_atm, tipo_atm=tipo_atm, moneyness_unidad=unidad,
                moneyness_log=[float(mmin * esc), float(mmax * esc)],
                moneyness_strikes=[float(S0 * np.exp((r - delta) * T + mmin * esc)),
                                   float(S0 * np.exp((r - delta) * T + mmax * esc))],
                arbitraje_modo=modo, arbitraje_otro_modo=otro, cuentas_otro_modo=dict(arbitraje=len(fuera_otro)),
                bordes=bordes)
    assert n0 - sum(cuentas.values()) == len(df), "las cuentas de la limpieza no cierran"
    return df, info

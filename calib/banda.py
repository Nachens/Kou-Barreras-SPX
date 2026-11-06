"""Barreras sobre la calibración (§5.4 de la tesis): precios, métricas, cruce y banda sobre C.

K = S0. La distancia normalizada es x = b / (sigma_atm sqrt(T)); 25 valores entre x_min y x_max, con
x_max la mayor distancia con L dentro de [K_min, K_max] de los strikes que sobrevivieron (L <= K_max
para la call up-and-out, L >= K_min para la put down-and-out).

Este módulo solo calcula cantidades; el contraste de la hipótesis lo hace el autor.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bs import barreras_rr as rr
from bs import formulas as bsf
from kou.barreras import Barreras
from kou.europeas import CarrMadan
from kou.modelo import kou_Q

CONTRATOS = ("call_up", "put_down")


def grilla(contrato, S0, T, sigma_atm, Kmin, Kmax, x_min, n_x):
    s = sigma_atm * np.sqrt(T)
    b_max = np.log(Kmax / S0) if contrato == "call_up" else np.log(S0 / Kmin)
    x_max = b_max / s
    if x_max <= x_min:
        raise ValueError(f"{contrato}: x_max={x_max:.3f} <= x_min={x_min} (el rango de strikes es muy angosto)")
    xs = np.linspace(x_min, x_max, n_x)
    b = xs * s
    L = S0 * np.exp(b) if contrato == "call_up" else S0 * np.exp(-b)
    return xs, b, L, float(x_max)


def _una_theta(args):
    """Precios de las dos barreras bajo Kou(theta) en toda la grilla. Devuelve dict contrato -> lista de dicts."""
    th, S0, K, T, r, delta, Ls, ci, N_fft = args
    kq = kou_Q(r, delta, *th)
    cm = CarrMadan(kq, T, r, delta, N=N_fft)
    call_eu = float(cm.call(S0, [K])[0])
    put_eu = float(cm.put(S0, [K])[0])
    br = barreras_de(kq, S0, K, T, r, delta, ci)
    out = {}
    for ct in CONTRATOS:
        reg = []
        for L in Ls[ct]:
            if ct == "call_up":
                d = br.call_up(L)
                eu = call_eu
            else:
                d = br.put_down(L)
                eu = put_eu
            v_in = float(d["v_in"])
            pc, pj, fr = br.cruce(L, ct)
            uc = getattr(br, "ultimo_cruce", dict(n=br.n, convergio=True))
            reg.append(dict(L=float(L), europea=eu, v_in=v_in, v_out=eu - v_in,
                            ctrl_europea_inv=float(d["europea_inv"]) - eu,
                            q_cont=float(pc), q_salto=float(pj), frac_salto=float(fr),
                            n_precio=int(d.get("n_usado", br.n)), n_cruce=int(uc["n"]),
                            convergio=bool(d.get("convergio", True) and uc["convergio"])))
        out[ct] = reg
    return out


def barreras_de(kq, S0, K, T, r, delta, ci):
    """Barreras con las opciones de inversión de la configuración (n fijo o adaptativo)."""
    return Barreras(kq, S0, K, T, r, delta, ci["n_GS"], ci["B"], ci["dps"], adaptativo=ci.get("adaptativo", False),
                    tol=ci.get("tol", 1e-6), n_max=ci.get("n_max", 100))


def precios_kou(th, S0, K, T, r, delta, Ls, cfg_inv, N_fft):
    return _una_theta((tuple(th), S0, K, T, r, delta, Ls, dict(cfg_inv), N_fft))


def precios_bs(sigma, S0, K, T, r, delta, Ls):
    out = {}
    for ct in CONTRATOS:
        reg = []
        for L in Ls[ct]:
            if ct == "call_up":
                eu = float(bsf.call(S0, K, T, r, delta, sigma))
                v = float(rr.rr_up_out_call(S0, K, L, T, r, delta, sigma))
            else:
                eu = float(bsf.put(S0, K, T, r, delta, sigma))
                v = float(rr.rr_down_out_put(S0, K, L, T, r, delta, sigma))
            reg.append(dict(L=float(L), europea=eu, v_out=v))
        out[ct] = reg
    return out


def tabla(th_kou, sigma_bs, S0, K, T, r, delta, grillas, s_bar, s_atm, cfg_inv, N_fft):
    """DataFrame (contrato, x, b, L, V_kou, V_bs, métricas, cruce) con Kou en th_kou y BS con sigma_bs."""
    Ls = {ct: grillas[ct]["L"] for ct in CONTRATOS}
    k = precios_kou(th_kou, S0, K, T, r, delta, Ls, cfg_inv, N_fft)
    b = precios_bs(sigma_bs, S0, K, T, r, delta, Ls)
    filas = []
    for ct in CONTRATOS:
        g = grillas[ct]
        for i, (kk, bb) in enumerate(zip(k[ct], b[ct])):
            dif = kk["v_out"] - bb["v_out"]
            filas.append(dict(
                contrato=ct, x=g["x"][i], b=g["b"][i], L=kk["L"], V_kou=kk["v_out"], V_bs=bb["v_out"],
                europea_kou=kk["europea"], europea_bs=bb["europea"], dif=dif, dif_abs=abs(dif),
                dif_abs_S0=abs(dif) / S0, dif_rel=dif / bb["v_out"] if bb["v_out"] > 0 else np.nan,
                dif_sobre_europea=abs(dif) / kk["europea"], dif_sobre_sbar=abs(dif) / s_bar,
                dif_sobre_satm=abs(dif) / s_atm, ctrl_europea_inv=kk["ctrl_europea_inv"],
                q_cont=kk["q_cont"], q_salto=kk["q_salto"], frac_salto=kk["frac_salto"],
                n_precio=kk["n_precio"], n_cruce=kk["n_cruce"], convergio=kk["convergio"]))
    return pd.DataFrame(filas)


def banda(thetas, S0, K, T, r, delta, grillas, cfg_inv, N_fft, jobs=1):
    """Mín y máx de V_out y de la fracción por salto sobre los theta dados, para cada (contrato, x).

    Es una cota inferior del rango sobre C: se recorre un subconjunto de C (mínimos locales y puntos del perfil).
    """
    Ls = {ct: grillas[ct]["L"] for ct in CONTRATOS}
    args = [(tuple(t), S0, K, T, r, delta, Ls, dict(cfg_inv), N_fft) for t in thetas]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_una_theta, args))
    else:
        res = [_una_theta(a) for a in args]
    filas = []
    for ct in CONTRATOS:
        g = grillas[ct]
        for i in range(len(g["x"])):
            v = np.array([r_[ct][i]["v_out"] for r_ in res])
            f = np.array([r_[ct][i]["frac_salto"] for r_ in res])
            filas.append(dict(contrato=ct, x=g["x"][i], L=g["L"][i], V_min=v.min(), V_max=v.max(), ancho=v.max() - v.min(),
                              frac_min=f.min(), frac_max=f.max(), n_theta=len(res),
                              n_max_usado=max(r_[ct][i]["n_precio"] for r_ in res),
                              convergio=all(r_[ct][i]["convergio"] for r_ in res)))
    return pd.DataFrame(filas)


def cantidades_hipotesis(tab, S0, s_bar, s_atm, banda_df=None):
    """Δmax, distancia donde ocurre, s̄, spread ATM y ancho de la banda. Solo cantidades, sin contraste."""
    out = {"s_bar": s_bar, "s_atm": s_atm}
    for ct in CONTRATOS:
        t = tab[tab.contrato == ct]
        i = t.dif_abs.idxmax()
        out[ct] = dict(delta_max=float(t.dif_abs[i]), delta_max_S0=float(t.dif_abs_S0[i]), x_en_delta_max=float(t.x[i]),
                       frac_salto_en_delta_max=float(t.frac_salto[i]), delta_max_sobre_sbar=float(t.dif_abs[i] / s_bar),
                       delta_max_sobre_satm=float(t.dif_abs[i] / s_atm), x_max=float(t.x.max()))
        if banda_df is not None:
            bb = banda_df[banda_df.contrato == ct]
            j = bb.ancho.idxmax()
            out[ct].update(ancho_banda_max=float(bb.ancho[j]), x_en_ancho_max=float(bb.x[j]),
                           ancho_banda_sobre_sbar=float(bb.ancho[j] / s_bar))
    return out

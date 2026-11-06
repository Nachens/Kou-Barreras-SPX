"""Cantidades adicionales de la comparación (fracción de cruces por salto sobre C, Black-Scholes en sigma_atm,
probabilidad de cruce de Black-Scholes, Delta_max sobre la europea), desde las salidas de la corrida preregistrada.
No recalibra ni vuelve a invertir nada: lee barreras.csv, banda.csv, calibracion.json e
hipotesis_cantidades.json, y agrega solo Black-Scholes en sigma_atm (Rubinstein-Reiner, fórmula cerrada) y la
probabilidad de cruce de Black-Scholes (principio de reflexión del browniano con deriva).

  python scripts/cantidades_adicionales.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml

Escribe results/cantidades_adicionales/numeros.json y salida.txt.

fraccion_salto_C: rango sobre C de la fracción de cruces por salto (frac_min, frac_max de banda.csv) en la distancia de Delta_max
     y en la del ancho máximo de la banda.
BS_sigma_atm:  Delta_max con Black-Scholes en sigma_atm en lugar de la sigma de mínimos cuadrados, sobre la misma grilla.
cruce_call_calma: probabilidad de cruce antes de T de Kou (q_cont + q_salto de barreras.csv) y de Black-Scholes (sigma de mínimos
     cuadrados) en la distancia de Delta_max de la call de la fecha tranquila.
delta_sobre_europea: Delta_max como fracción de la europea de Kou con K = S0.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from _comun import RAIZ, cargar_calibracion, cargar_cfg, parser, salida
from bs import barreras_rr as rr

CTS = ("call_up", "put_down")


def prob_cruce_bs(b, T, r, delta, sigma):
    """Q(max_{t<=T} X_t >= b), b > 0, X_t = mu t + sigma W_t, mu = r - delta - sigma^2/2 (reflexión)."""
    mu = r - delta - sigma ** 2 / 2
    s = sigma * np.sqrt(T)
    return float(norm.cdf((-b + mu * T) / s) + np.exp(2 * mu * b / sigma ** 2) * norm.cdf((-b - mu * T) / s))


def fila_cercana(df, x):
    i = (df.x - x).abs().idxmin()
    return df.loc[i], float(abs(df.x[i] - x))


if __name__ == "__main__":
    a = parser("números de la auditoría del 29/10", multiple=True).parse_args()
    res = {}
    lineas = []
    for path in a.config:
        cfg = cargar_cfg(path)
        esc = cfg["escenario"]
        cal = cargar_calibracion(cfg)
        d = salida(cfg)
        tab = pd.read_csv(d / "barreras.csv")
        ban = pd.read_csv(d / "banda.csv")
        hq = json.load(open(d / "hipotesis_cantidades.json", encoding="utf-8"))
        S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
        K = S0 * cfg["barreras"]["K_relativo"]
        s_atm, s_ls, s_bar = cal["bs"]["sigma_ATM"], cal["bs"]["sigma_elegida"], hq["s_bar"]
        out = {}
        for ct in CTS:
            t = tab[tab.contrato == ct].reset_index(drop=True)
            b = ban[ban.contrato == ct].reset_index(drop=True)
            h = hq[ct]
            # fraccion_salto_C
            fd, dx1 = fila_cercana(b, h["x_en_delta_max"])
            fa, dx2 = fila_cercana(b, h["x_en_ancho_max"])
            g14 = dict(x_delta_max=h["x_en_delta_max"], frac_min_en_delta_max=float(fd.frac_min),
                       frac_max_en_delta_max=float(fd.frac_max), dist_x_banda_delta=dx1,
                       frac_opt_en_delta_max=h["frac_salto_en_delta_max"],
                       x_ancho_max=h["x_en_ancho_max"], frac_min_en_ancho_max=float(fa.frac_min),
                       frac_max_en_ancho_max=float(fa.frac_max), dist_x_banda_ancho=dx2,
                       frac_min_global=float(b.frac_min.min()), frac_max_global=float(b.frac_max.max()))
            # BS_sigma_atm
            f = rr.rr_up_out_call if ct == "call_up" else rr.rr_down_out_put
            v_atm = np.array([float(f(S0, K, L, T, r, dl, s_atm)) for L in t.L])
            v_ls = np.array([float(f(S0, K, L, T, r, dl, s_ls)) for L in t.L])
            dif_atm = np.abs(t.V_kou.to_numpy() - v_atm)
            j = int(dif_atm.argmax())
            ctrl_ls = float(np.abs(t.V_kou.to_numpy() - v_ls).max())
            b1 = dict(sigma_atm=s_atm, sigma_ls=s_ls, delta_max_atm=float(dif_atm[j]), x_atm=float(t.x[j]),
                      delta_max_atm_sbar=float(dif_atm[j] / s_bar), delta_max_ls=h["delta_max"],
                      x_ls=h["x_en_delta_max"], delta_max_ls_sbar=h["delta_max_sobre_sbar"],
                      control_delta_max_ls_recalculado=ctrl_ls, n_grilla=int(len(t)))
            # delta_sobre_europea
            eu = float(t.europea_kou.iloc[0])
            g16 = dict(europea_kou=eu, delta_max=h["delta_max"], fraccion=h["delta_max"] / eu)
            out[ct] = dict(fraccion_salto_C=g14, BS_sigma_atm=b1, delta_sobre_europea=g16)
            lineas.append(f"[{esc} {ct}] fraccion_salto_C: en x={g14['x_delta_max']:.3f} (Δmax) frac en C ∈ [{g14['frac_min_en_delta_max']:.4f}, "
                          f"{g14['frac_max_en_delta_max']:.4f}] (óptimo {g14['frac_opt_en_delta_max']:.4f}); en x={g14['x_ancho_max']:.3f} "
                          f"(ancho máx) ∈ [{g14['frac_min_en_ancho_max']:.4f}, {g14['frac_max_en_ancho_max']:.4f}]; "
                          f"dist. de grilla {dx1:.1e}, {dx2:.1e}")
            lineas.append(f"[{esc} {ct}] BS_sigma_atm: σ_atm={s_atm:.4f}: Δmax={b1['delta_max_atm']:.2f} en x={b1['x_atm']:.2f} "
                          f"({b1['delta_max_atm_sbar']:.1f} s̄); con σ_LS={s_ls:.4f}: {h['delta_max']:.2f} en x={h['x_en_delta_max']:.2f} "
                          f"({h['delta_max_sobre_sbar']:.1f} s̄; recalculado {ctrl_ls:.4f}); grilla de {len(t)} puntos")
            lineas.append(f"[{esc} {ct}] delta_sobre_europea: Δmax / europea Kou = {h['delta_max']:.2f} / {eu:.2f} = {g16['fraccion']:.3f}")
        if esc == "calma":
            t = tab[tab.contrato == "call_up"].reset_index(drop=True)
            x0 = hq["call_up"]["x_en_delta_max"]
            fr, _ = fila_cercana(t, x0)
            p_kou = float(fr.q_cont + fr.q_salto)
            p_bs = prob_cruce_bs(float(fr.b), T, r, dl, s_ls)
            out["cruce_call_calma"] = dict(x=float(fr.x), p_cruce_kou=p_kou, frac_salto_kou=float(fr.frac_salto),
                              p_cruce_bs=p_bs, sigma_kou=cal["kou"]["optimo"]["theta"][0], sigma_bs=s_ls)
            lineas.append(f"[calma call_up] cruce_call_calma: x={fr.x:.3f}: Q(τ≤T) Kou={p_kou:.4f} (por salto {fr.frac_salto:.4f}); "
                          f"BS(σ={s_ls:.4f})={p_bs:.4f}; σ Kou={cal['kou']['optimo']['theta'][0]:.4f}")
        res[esc] = out
    d = RAIZ / "results" / "cantidades_adicionales"
    d.mkdir(parents=True, exist_ok=True)
    (d / "numeros.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "salida.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))

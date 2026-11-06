"""Sensibilidad a r: se recalibran Black-Scholes y Kou con r = r-sombrero ± 0,01 y
r - delta fijo (el mismo precio a término), y se recalcula Delta_max de cada contrato en la misma grilla de
barreras. Con r_fuente = 'externa' se agrega el r de la paridad de dos parámetros (ventana ±10 %).

La limpieza no cambia (r - delta es el mismo, así que el filtro de moneyness y el de solo OTM dan lo mismo).
Cambian los precios descontados de los modelos y los pesos (b), que dependen de r por la vega.

  python scripts/sensibilidad_r.py --config config/real_2017-10-11.yaml --jobs 11

Escribe out/<config>/sensibilidad_r.json y sensibilidad_r.csv.
"""
import json
import time

import numpy as np

from _comun import C, cargar_calibracion, cargar_cfg, escribir_tabla, parser, preparar, salida
from calib import banda as bd
from calib.optimizar import calibrar_bs, calibrar_kou

if __name__ == "__main__":
    a = parser("sensibilidad a r").parse_args()
    cfg = cargar_cfg(a.config[0])
    t0 = time.time()
    cal = cargar_calibracion(cfg)
    S0, T, r0, d0 = cal["S0"], cal["T"], cal["r"], cal["delta"]
    rmd = r0 - d0
    K = S0 * cfg["barreras"]["K_relativo"]
    cc, ci, cb = cfg["calibracion"], cfg["inversion"], cfg["barreras"]
    grillas = {}
    for ct in bd.CONTRATOS:
        xs, b, L, xmax = bd.grilla(ct, S0, T, cal["sigma_atm_imp"], cal["K_min"], cal["K_max"], cb["x_min"], cb["n_x"])
        grillas[ct] = dict(x=xs, b=b, L=L, x_max=xmax)
    q0 = json.loads((salida(cfg) / "hipotesis_cantidades.json").read_text(encoding="utf-8"))
    variantes = [("r - 0,01", r0 - 0.01), ("r + 0,01", r0 + 0.01)]
    if cfg["limpieza"].get("r_fuente") == "externa":
        variantes.append(("r de la paridad (2 parámetros, ±10 %)", cal["limpieza"]["paridad"]["diagnostico"]["0.10"]["r"]))
    s_bar = cal["s_bar"]
    filas, detalle = [], []
    for nombre, r in variantes:
        dl = r - rmd
        cad, df, info, prob = preparar(cfg, r_delta=(r, dl))
        k = calibrar_kou(prob, cc["n_inicios"], cc["factor_parecido"], cc["semilla"], jobs=a.jobs)
        bs = calibrar_bs(prob)
        sig = bs["sigma_LS"] if cc["sigma_bs"] == "LS" else bs["sigma_ATM"]
        tab = bd.tabla(k["optimo_theta"], sig, S0, K, T, r, dl, grillas, s_bar, cal["s_atm"], ci, cc["N_fft_final"])
        reg = dict(variante=nombre, r=r, delta=dl, theta=list(map(float, k["optimo_theta"])), sigma_bs=sig, R_min=k["R_min"])
        for ct in bd.CONTRATOS:
            t = tab[tab.contrato == ct]
            i = t.dif_abs.idxmax()
            dm0 = q0[ct]["delta_max"]
            reg[ct] = dict(delta_max=float(t.dif_abs[i]), x_en_delta_max=float(t.x[i]),
                           cambio=float(t.dif_abs[i] - dm0), cambio_sobre_sbar=float((t.dif_abs[i] - dm0) / s_bar))
            filas.append(dict(variante=nombre, r=r, contrato=ct, delta_max=reg[ct]["delta_max"], delta_max_base=dm0,
                              cambio_sobre_sbar=reg[ct]["cambio_sobre_sbar"]))
        detalle.append(reg)
        print(f"  {nombre}: r={r:.5f}, delta={dl:.5f}, theta={np.round(k['optimo_theta'], 4)}, sigma_BS={sig:.4f}; "
              + "; ".join(f"{ct}: Δmax={reg[ct]['delta_max']:.4f} ({reg[ct]['cambio_sobre_sbar']:+.3f} s̄)" for ct in bd.CONTRATOS))
    d = salida(cfg)
    res = dict(base=dict(r=r0, delta=d0, delta_max={ct: q0[ct]["delta_max"] for ct in bd.CONTRATOS}), s_bar=s_bar,
               variantes=detalle,
               max_cambio_sobre_sbar=max(abs(f["cambio_sobre_sbar"]) for f in filas))
    (d / "sensibilidad_r.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    escribir_tabla("sensibilidad_r", filas, ["variante", "r", "contrato", "delta_max", "delta_max_base", "cambio_sobre_sbar"], d,
                   tex_cols=["Variante", "$r$", "Contrato", "$\\Delta_{\\max}$", "$\\Delta_{\\max}$ base", "Cambio / $\\bar s$"],
                   spec="lrlrrr", fmt=dict(r="{:.4f}", delta_max="{:.4f}", delta_max_base="{:.4f}", cambio_sobre_sbar="{:+.3f}"))
    C.escribir_resumen(cfg, t0, extra=dict(max_cambio_sobre_sbar=res["max_cambio_sobre_sbar"]), nombre="resumen_sensibilidad_r.json")
    print(f"  máximo |cambio| = {res['max_cambio_sobre_sbar']:.3f} s̄; {time.time() - t0:.0f} s")

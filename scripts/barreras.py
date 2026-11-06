"""Barreras (§5.4): precios de Kou y Black-Scholes en la grilla de distancias, cruce por continuidad y por salto,
y banda sobre C. Escribe barreras.csv, banda.csv y hipotesis_cantidades.json (solo cantidades)."""
import json
import time

import numpy as np
import pandas as pd

from _comun import C, cargar_calibracion, cargar_cfg, escribir_tabla, parser, preparar, salida
from calib import banda as bd

if __name__ == "__main__":
    a = parser("barreras").parse_args()
    cfg = cargar_cfg(a.config[0])
    t0 = time.time()
    cal = cargar_calibracion(cfg)
    S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
    K = S0 * cfg["barreras"]["K_relativo"]
    cb, ci = cfg["barreras"], cfg["inversion"]
    sig_atm = cal["sigma_atm_imp"]
    grillas = {}
    for ct in bd.CONTRATOS:
        xs, b, L, xmax = bd.grilla(ct, S0, T, sig_atm, cal["K_min"], cal["K_max"], cb["x_min"], cb["n_x"])
        borde = cal["limpieza"]["bordes"]["call_K_max" if ct == "call_up" else "put_K_min"]
        grillas[ct] = dict(x=xs, b=b, L=L, x_max=xmax, K_borde=borde["K"], filtro_borde=borde["filtro"])
        print(f"  {ct}: x en [{xs[0]:.3f}, {xmax:.3f}], L en [{L.min():.1f}, {L.max():.1f}]; "
              f"x_max lo fija el strike {borde['K']:.0f} (filtro: {borde['filtro']})")
    th = cal["kou"]["optimo"]["theta"]
    sig_bs = cal["bs"]["sigma_elegida"]
    N = cfg["calibracion"]["N_fft_final"]
    tab = bd.tabla(th, sig_bs, S0, K, T, r, dl, grillas, cal["s_bar"], cal["s_atm"], ci, N)
    tab.to_csv(salida(cfg) / "barreras.csv", index=False, float_format="%.12g")
    nc = int((~tab.convergio).sum())
    print(f"  inversión: n usado entre {tab.n_precio.min()} y {tab.n_precio.max()} (precios), "
          f"{tab.n_cruce.min()} y {tab.n_cruce.max()} (cruce)" + (f"; AVISO: {nc} filas sin converger" if nc else ""))
    print(f"  tabla de barreras lista ({time.time() - t0:.0f} s); control europea (inversión vs FFT) máx = "
          f"{tab.ctrl_europea_inv.abs().max():.2e}")
    # la banda recorre solo los theta de C (theta-sombrero entra si está en C, no por ser el óptimo)
    pts, criterio_banda = cal["C"]["puntos"], cal["C"]["criterio"]
    if not pts and cal["C"].get("respaldo"):
        pts, criterio_banda = cal["C"]["respaldo"]["puntos"], cal["C"]["respaldo"]["criterio"]
        print(f"  C está vacío con el criterio {cal['C']['criterio']}: la banda usa el de respaldo ({criterio_banda})")
    bnd = None
    if pts:
        thetas = [p["theta"] for p in pts]
        bnd = bd.banda(thetas, S0, K, T, r, dl, grillas, ci, N, jobs=a.jobs)
        bnd.to_csv(salida(cfg) / "banda.csv", index=False, float_format="%.12g")
        print(f"  banda sobre {len(thetas)} theta ({time.time() - t0:.0f} s)")
    else:
        print("  C está vacío (también con el respaldo): no hay banda")
    q = bd.cantidades_hipotesis(tab, S0, cal["s_bar"], cal["s_atm"], bnd)
    q["nota_banda"] = "la banda es una cota inferior del rango sobre C: se recorre un subconjunto (mínimos locales y perfil)"
    q["x_max"] = {ct: grillas[ct]["x_max"] for ct in bd.CONTRATOS}
    q["borde_grilla"] = {ct: dict(K=grillas[ct]["K_borde"], filtro=grillas[ct]["filtro_borde"]) for ct in bd.CONTRATOS}
    q["sigma_atm"] = sig_atm
    q["n_theta_banda"] = len(pts)
    q["criterio_C"] = cal["C"]["criterio"]
    q["criterio_banda"] = criterio_banda if pts else None
    q["inversion"] = dict(adaptativo=bool(ci.get("adaptativo", False)), n_min=int(tab.n_precio.min()),
                          n_max=int(max(tab.n_precio.max(), tab.n_cruce.max())), filas_sin_converger=nc,
                          banda_sin_converger=int((~bnd.convergio).sum()) if bnd is not None else 0,
                          n_max_banda=int(bnd.n_max_usado.max()) if bnd is not None else None)
    for ct in bd.CONTRATOS:
        t_ = tab[tab.contrato == ct]
        q["inversion"][f"n_max_{ct}"] = int(max(t_.n_precio.max(), t_.n_cruce.max()))
    q["theta_opt_en_C"] = cal["C"]["theta_opt_en_C"]
    (salida(cfg) / "hipotesis_cantidades.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
    C.escribir_resumen(cfg, t0, extra=dict(cantidades=q), nombre="resumen_barreras.json")
    print(json.dumps({k: v for k, v in q.items() if k != "nota_banda"}, indent=1)[:900])

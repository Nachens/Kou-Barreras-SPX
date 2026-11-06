"""Banda sobre los subconjuntos de C con R <= kappa eps0^2 y europeas de sus extremos.

No recalibra ni optimiza: toma los theta de C que ya están en calibracion.json (mínimos locales y perfil en lambda),
los filtra por R <= kappa eps0^2 para kappa = 1/4, 1/2 y 1, y recalcula los precios de las barreras en la grilla con la
inversión de la configuración, como scripts/barreras.py. Con kappa = 1 reproduce banda.csv (se controla). Para cada
(contrato, kappa) da el ancho máximo de la banda, la distancia donde se alcanza, los dos theta de los extremos, cuántas
opciones dejan fuera del bid-ask y la mayor diferencia entre sus precios de las europeas usadas para calibrar
(FFT con N_fft_final), además de la diferencia de sus europeas con K = S0.

  python scripts/banda_kappa.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml --jobs 11

Escribe out/<config>/banda_kappa.json.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from _comun import C, cargar_calibracion, cargar_cfg, parser, preparar, salida
from calib import banda as bd
from kou.europeas import CarrMadan
from kou.modelo import kou_Q

KAPPAS = (0.25, 0.5, 1.0)

if __name__ == "__main__":
    a = parser("banda sobre los subconjuntos de C con R <= kappa eps0^2", multiple=True).parse_args()
    for path in a.config:
        cfg = cargar_cfg(path)
        t0 = time.time()
        cal = cargar_calibracion(cfg)
        cad, df, info, prob = preparar(cfg)
        S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
        K = S0 * cfg["barreras"]["K_relativo"]
        cb, ci = cfg["barreras"], cfg["inversion"]
        N = cfg["calibracion"]["N_fft_final"]
        grillas = {}
        for ct in bd.CONTRATOS:
            xs, b, L, _ = bd.grilla(ct, S0, T, cal["sigma_atm_imp"], cal["K_min"], cal["K_max"], cb["x_min"], cb["n_x"])
            grillas[ct] = dict(x=xs, L=L)
        pts, e0, sbar = cal["C"]["puntos"], cal["eps0_sq"], cal["s_bar"]
        Ls = {ct: grillas[ct]["L"] for ct in bd.CONTRATOS}
        args = [(tuple(p["theta"]), S0, K, T, r, dl, Ls, dict(ci), N) for p in pts]
        if a.jobs and a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                res = list(ex.map(bd._una_theta, args))
        else:
            res = [bd._una_theta(x) for x in args]
        Rrel = np.array([p["R"] / e0 for p in pts])
        spr = prob.ask - prob.bid
        cache = {}

        def precios(i):
            if i not in cache:
                cache[i] = prob.precios_modelo(np.asarray(pts[i]["theta"], float), N)
            return cache[i]

        def atm(i):
            cm = CarrMadan(kou_Q(r, dl, *pts[i]["theta"]), T, r, dl, N=N)
            return float(cm.call(S0, [K])[0]), float(cm.put(S0, [K])[0])

        banda_csv = pd.read_csv(salida(cfg) / "banda.csv")
        out = dict(eps0_sq=e0, s_bar=sbar, R_min_sobre_eps0_sq=float(Rrel.min()), n_opciones=int(len(spr)),
                   kappas=list(KAPPAS), contratos={})
        for ct in bd.CONTRATOS:
            V = np.array([[r_[ct][j]["v_out"] for j in range(len(grillas[ct]["x"]))] for r_ in res])
            bb = banda_csv[banda_csv.contrato == ct]
            control = float(max(np.abs(V.min(0) - bb.V_min.to_numpy()).max(), np.abs(V.max(0) - bb.V_max.to_numpy()).max()))
            filas = []
            for kap in KAPPAS:
                idx = np.where(Rrel <= kap * (1 + 1e-12))[0]
                if len(idx) == 0:
                    filas.append(dict(kappa=kap, n=0))
                    continue
                Vs = V[idx]
                ancho = Vs.max(0) - Vs.min(0)
                j = int(ancho.argmax())
                i0, i1 = int(idx[Vs[:, j].argmin()]), int(idx[Vs[:, j].argmax()])
                c0, c1 = precios(i0), precios(i1)
                d = np.abs(c0 - c1)
                m = int(d.argmax())
                a0, a1 = atm(i0), atm(i1)
                fuera = [int(((c < prob.bid) | (c > prob.ask)).sum()) for c in (c0, c1)]
                filas.append(dict(kappa=kap, n=int(len(idx)), ancho=float(ancho[j]), ancho_sobre_sbar=float(ancho[j] / sbar),
                                  x=float(grillas[ct]["x"][j]), indices_extremos=[i0, i1],
                                  R_extremos_sobre_eps0_sq=[float(Rrel[i0]), float(Rrel[i1])], fuera_bidask_extremos=fuera,
                                  dif_europeas_max=float(d[m]), dif_europeas_max_sobre_sbar=float(d[m] / sbar),
                                  K_dif_europeas_max=float(prob.K[m]), es_put_dif_europeas_max=bool(prob.es_put[m]),
                                  dif_call_atm=abs(a0[0] - a1[0]), dif_put_atm=abs(a0[1] - a1[1]),
                                  dif_atm_sobre_sbar=abs(a0[0] - a1[0]) / sbar))
            out["contratos"][ct] = dict(control_kappa_uno_vs_banda_csv=control, filas=filas)
        (salida(cfg) / "banda_kappa.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        C.escribir_resumen(cfg, t0, extra=dict(banda_kappa=out), nombre="resumen_banda_kappa.json")
        for ct, v in out["contratos"].items():
            s = "; ".join(f"κ={f['kappa']:g}: " + ("vacío" if f["n"] == 0 else
                          f"{f['n']} θ, ancho {f['ancho_sobre_sbar']:.2f} s̄, europeas {f['dif_europeas_max_sobre_sbar']:.2f} s̄")
                          for f in v["filas"])
            print(f"{cfg['nombre']} {ct}: {s} (control κ=1 vs banda.csv {v['control_kappa_uno_vs_banda_csv']:.1e})")

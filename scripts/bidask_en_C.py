"""Cuántas opciones quedan fuera del bid-ask para theta-sombrero y para los theta de C.

El criterio de C es R(theta) <= eps0^2: una cota sobre el error agregado, no sobre cada precio. Este script mide qué
tan lejos queda eso de "todos los precios dentro del bid-ask". No recalibra ni cambia ningún resultado: relee la
cadena, evalúa los precios del modelo (FFT con N_fft_final) en los theta que ya están en calibracion.json y, para los
extremos de la banda, recalcula V_out en la distancia del ancho máximo (con la inversión de la configuración) y
verifica que dé los V_min y V_max de banda.csv.

  python scripts/bidask_en_C.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml

Escribe out/<config>/bidask_en_C.json.
"""
import json
import time

import numpy as np
import pandas as pd

from _comun import C, cargar_calibracion, cargar_cfg, parser, preparar, salida
from calib import banda as bd
from kou.europeas import CarrMadan
from kou.modelo import kou_Q


def estadisticas(prob, th, N):
    c = prob.precios_modelo(np.asarray(th, float), N)
    spr = prob.ask - prob.bid
    fuera = (c < prob.bid) | (c > prob.ask)
    dev = np.maximum(np.maximum(prob.bid - c, c - prob.ask), 0.0) / spr
    return dict(R=float(prob.R(np.asarray(th, float), N)), n_fuera=int(fuera.sum()), n=int(len(c)),
                dev_max_spreads=float(dev.max()))


if __name__ == "__main__":
    a = parser("opciones fuera del bid-ask en C", multiple=True).parse_args()
    for path in a.config:
        cfg = cargar_cfg(path)
        t0 = time.time()
        cal = cargar_calibracion(cfg)
        cad, df, info, prob = preparar(cfg)
        N = cfg["calibracion"]["N_fft_final"]
        e0 = cal["eps0_sq"]
        th_hat = cal["kou"]["optimo"]["theta"]
        hat = dict(theta=th_hat, **estadisticas(prob, th_hat, N))
        pts = cal["C"]["puntos"]
        en_C = [dict(theta=p["theta"], origen=p.get("origen"), **estadisticas(prob, p["theta"], N)) for p in pts]
        S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
        K = S0 * cfg["barreras"]["K_relativo"]
        banda = pd.read_csv(salida(cfg) / "banda.csv")
        ext = {}
        for ct in bd.CONTRATOS:
            b = banda[banda.contrato == ct].reset_index(drop=True)
            j = int(b.ancho.idxmax())
            L = float(b.L[j])
            vals = []
            for p in pts:
                kq = kou_Q(r, dl, *p["theta"])
                cm = CarrMadan(kq, T, r, dl, N=N)
                br = bd.barreras_de(kq, S0, K, T, r, dl, dict(cfg["inversion"]))
                if ct == "call_up":
                    vals.append(float(cm.call(S0, [K])[0]) - float(br.call_up(L)["v_in"]))
                else:
                    vals.append(float(cm.put(S0, [K])[0]) - float(br.put_down(L)["v_in"]))
            i0, i1 = int(np.argmin(vals)), int(np.argmax(vals))
            ext[ct] = dict(x=float(b.x[j]), L=L, V_min=vals[i0], V_max=vals[i1],
                           dif_con_banda_min=abs(vals[i0] - float(b.V_min[j])), dif_con_banda_max=abs(vals[i1] - float(b.V_max[j])),
                           min=dict(indice_en_C=i0, **{k: v for k, v in en_C[i0].items()}),
                           max=dict(indice_en_C=i1, **{k: v for k, v in en_C[i1].items()}))
        extremos = [ext[ct][lado] for ct in bd.CONTRATOS for lado in ("min", "max")]
        out = dict(eps0_sq=e0, R_min=cal["kou"]["optimo"]["R"], R_min_sobre_eps0_sq=cal["kou"]["optimo"]["R"] / e0,
                   N=hat["n"], theta_hat=hat, en_C=en_C,
                   criterio_bidask_implica_R_menor_o_igual=e0 / 4,
                   criterio_bidask_vacio_por_R_min=bool(cal["kou"]["optimo"]["R"] > e0 / 4),
                   resumen_C=dict(n_fuera_min=min(f["n_fuera"] for f in en_C), n_fuera_max=max(f["n_fuera"] for f in en_C),
                                  dev_max_spreads=max(f["dev_max_spreads"] for f in en_C),
                                  R_max_sobre_eps0_sq=max(f["R"] for f in en_C) / e0),
                   extremos_banda=ext,
                   resumen_extremos=dict(n_fuera_min=min(f["n_fuera"] for f in extremos),
                                         n_fuera_max=max(f["n_fuera"] for f in extremos),
                                         dev_max_spreads=max(f["dev_max_spreads"] for f in extremos)))
        (salida(cfg) / "bidask_en_C.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        C.escribir_resumen(cfg, t0, extra=dict(bidask_en_C=out["resumen_C"]), nombre="resumen_bidask_en_C.json")
        r_ = out["resumen_C"]
        print(f"{cfg['nombre']}: eps0^2={e0:.4g}, R_min/eps0^2={out['R_min_sobre_eps0_sq']:.3f}; theta-sombrero fuera del bid-ask: "
              f"{hat['n_fuera']}/{hat['n']} (hasta {hat['dev_max_spreads']:.2f} spreads); C ({len(en_C)} puntos): "
              f"{r_['n_fuera_min']}..{r_['n_fuera_max']} fuera, hasta {r_['dev_max_spreads']:.2f} spreads; extremos de la banda: "
              f"{out['resumen_extremos']['n_fuera_min']}..{out['resumen_extremos']['n_fuera_max']}; "
              f"control V_min/V_max vs banda.csv: {max(e['dif_con_banda_min'] for e in ext.values()):.1e}, "
              f"{max(e['dif_con_banda_max'] for e in ext.values()):.1e}")

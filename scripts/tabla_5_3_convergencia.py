"""Tabla 5.3: f*_n(T) para n = 6..20 y dígitos en {20, 30, 45, 60}, call up-and-out de distancia media."""
import json
import time

import mpmath as mp
import numpy as np

from _comun import C, cargar_cfg, escribir_tabla, parser, preparar, salida

if __name__ == "__main__":
    a = parser("Tabla 5.3").parse_args()
    cfg = cargar_cfg(a.config[0])
    t0 = time.time()
    cad, df, info, prob = preparar(cfg)
    S0, T, r, dl = cad.S0, cad.T, info["r"], info["delta"]
    f = salida(cfg) / "calibracion.json"
    if not f.exists():
        raise SystemExit(f"falta {f}: corré antes scripts/calibrar.py")
    cal = json.load(open(f, encoding="utf-8"))
    th = cal["kou"]["optimo"]["theta"]
    origen = "calibración"
    from kou.barreras import Barreras
    from kou.modelo import kou_Q
    kq = kou_Q(r, dl, *th)
    sig_atm = prob.iv_mkt[np.nanargmin(np.abs(prob.K - S0 * np.exp((r - dl) * T)))]
    b = 1.0 * sig_atm * np.sqrt(T)                   # x = 1: distancia media de la grilla
    L = S0 * np.exp(b)
    ns, digs = list(range(6, 21)), [20, 30, 45, 60]
    filas = []
    for dps in digs:
        br = Barreras(kq, S0, S0, T, r, dl, n=max(ns), B=2, dps=dps)
        prev = None
        for n in ns:
            br.fijar_n(n)
            try:
                v = float(br.call_up(L)["v_out_inv"])
            except Exception:
                v = float("nan")
            filas.append(dict(n=n, digitos=dps, valor=v, dif=(v - prev) if prev is not None else float("nan")))
            prev = v
    escribir_tabla("tabla_5_3_convergencia", filas, ["n", "digitos", "valor", "dif"], salida(cfg),
                   tex_cols=["$n$", "Precisión (dígitos)", "Valor invertido $\\mathfrak f^*_n(T)$", "Diferencia con $n-1$"],
                   spec="cccc", fmt={"valor": "{:.10f}", "dif": "{:.3e}"})
    from calib.banda import barreras_de
    da = barreras_de(kq, S0, S0, T, r, dl, cfg["inversion"]).call_up(L)
    adapt = dict(n_usado=int(da["n_usado"]), convergio=bool(da["convergio"]), valor=float(da["v_out_inv"]))
    print(f"adaptativo: n = {adapt['n_usado']}, convergió: {adapt['convergio']}, valor = {adapt['valor']:.10f}")
    C.escribir_resumen(cfg, t0, extra=dict(theta=list(map(float, th)), origen_theta=origen, L=float(L), x=1.0,
                                           adaptativo=adapt),
                       nombre="resumen_tabla_5_3.json")
    print(f"L={L:.2f} (x=1), theta de {origen}")
    for dps in digs:
        print(dps, [f"{x['valor']:.7f}" for x in filas if x['digitos'] == dps][::2])

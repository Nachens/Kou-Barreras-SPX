"""Monte Carlo en los parámetros calibrados: con el theta-sombrero de Kou de la
fecha, la call up-and-out y la put down-and-out en x = 0,5; 1,0 y x_max/2 por simulación exacta (puente browniano
entre saltos, kou/mc.py), contra la inversión. Pasa si cada diferencia es de a lo sumo 3 errores estándar.

  python scripts/mc_real.py --config config/real_2017-10-11.yaml [--n 2000000] [--semilla 0]

Escribe out/<config>/mc_real.csv y mc_real.json.
"""
import json
import time

import numpy as np

from _comun import C, cargar_calibracion, cargar_cfg, escribir_tabla, parser, salida
from calib.banda import barreras_de
from kou.mc import precios_mc_barreras
from kou.modelo import kou_Q

if __name__ == "__main__":
    p = parser("Monte Carlo en los parámetros calibrados")
    p.add_argument("--n", type=int, default=None, help="por defecto, monte_carlo.n de la configuración (2·10^6)")
    p.add_argument("--semilla", type=int, default=None)
    a = p.parse_args()
    cfg = cargar_cfg(a.config[0])
    mcc = cfg.get("monte_carlo") or {}
    a.n = a.n or int(mcc.get("n", 2_000_000))
    a.semilla = a.semilla if a.semilla is not None else int(mcc.get("semilla", 0))
    t0 = time.time()
    cal = cargar_calibracion(cfg)
    q = json.loads((salida(cfg) / "hipotesis_cantidades.json").read_text(encoding="utf-8"))
    S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
    K = S0 * cfg["barreras"]["K_relativo"]
    th = cal["kou"]["optimo"]["theta"]
    s = cal["sigma_atm_imp"] * np.sqrt(T)
    xs = {ct: [0.5, 1.0, q["x_max"][ct] / 2] for ct in ("call_up", "put_down")}
    L_up = [S0 * np.exp(x * s) for x in xs["call_up"]]
    L_dn = [S0 * np.exp(-x * s) for x in xs["put_down"]]
    kq = kou_Q(r, dl, *th)
    ci = cfg["inversion"]
    br = barreras_de(kq, S0, K, T, r, dl, ci)
    du = [br.call_up(L) for L in L_up]
    dd = [br.put_down(L) for L in L_dn]
    inv = dict(call_up=[float(x["v_out_inv"]) for x in du], put_down=[float(x["v_out_inv"]) for x in dd])
    n_us = dict(call_up=[int(x["n_usado"]) for x in du], put_down=[int(x["n_usado"]) for x in dd])
    mc = precios_mc_barreras(kq, S0, K, T, r, dl, L_up, L_dn, a.n, a.semilla)
    filas = []
    for ct, clave, Ls in (("call_up", "uoc", L_up), ("put_down", "dop", L_dn)):
        for x, L, vi, (vm, ee), nu in zip(xs[ct], Ls, inv[ct], mc[clave], n_us[ct]):
            z = abs(vm - vi) / ee if ee > 0 else (0.0 if vm == vi else np.inf)
            filas.append(dict(contrato=ct, x=x, L=L, V_inversion=vi, V_mc=vm, error_estandar=ee, z=z, ok=bool(z <= 3),
                              n_usado=nu))
    d = salida(cfg)
    escribir_tabla("mc_real", filas, ["contrato", "x", "L", "V_inversion", "V_mc", "error_estandar", "z", "ok", "n_usado"], d,
                   tex_cols=["Contrato", "$x$", "$L$", "Inversión", "Monte Carlo", "Error estándar", "$|z|$", "Pasa", "$n$"],
                   spec="lrrrrrrcr", fmt=dict(x="{:.3f}", L="{:.2f}", V_inversion="{:.4f}", V_mc="{:.4f}",
                                             error_estandar="{:.4f}", z="{:.2f}"))
    res = dict(n=a.n, semilla=a.semilla, theta=th, filas=filas, z_max=max(f["z"] for f in filas),
               todos_ok=all(f["ok"] for f in filas), europeas_mc=dict(call=mc["call"], put=mc["put"]))
    (d / "mc_real.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    C.escribir_resumen(cfg, t0, extra=dict(z_max=res["z_max"], todos_ok=res["todos_ok"]), nombre="resumen_mc_real.json")
    for f in filas:
        print(("OK  " if f["ok"] else "FALLA"), f["contrato"], f"x={f['x']:.3f}", f"n={f['n_usado']}", f"inv={f['V_inversion']:.5f}",
              f"mc={f['V_mc']:.5f}±{f['error_estandar']:.5f}", f"z={f['z']:.2f}")
    print(f"  {time.time() - t0:.0f} s")

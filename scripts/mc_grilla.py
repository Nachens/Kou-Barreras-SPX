"""Monte Carlo en grilla (el "naive"), para mostrar por qué no sirve de control. Análisis exploratorio, no
preregistrado: no entra en ningún chequeo ni en ningún veredicto.

Con el theta-sombrero de Kou de la fecha, K = S0 y la barrera en x = 1 (la misma de mc_real.py), simula el proceso
solo en una grilla de m pasos iguales y toma el máximo (o el mínimo) sobre los puntos de la grilla. En los puntos de
la grilla la simulación es exacta (incremento gaussiano más la suma de los saltos del paso, Cont–Tankov Alg. 6.3,
Glasserman ec. (3.82)); el único error, además del estadístico, es no mirar el proceso entre dos puntos. Compara con
la inversión y con el Monte Carlo exacto de mc_real.json.

  python scripts/mc_grilla.py --config config/real_2017-10-11.yaml [--n 400000] [--semilla 1]

Escribe out/<config>/mc_grilla.json.
"""
import json
import math
import time

import numpy as np

from _comun import C, cargar_calibracion, cargar_cfg, parser, salida
from kou.modelo import kou_Q

PASOS_POR_DIA = (1, 10, 100)


def simular_grilla(k, T, m, n, rng, chunk=200_000):
    """X_T, máximo y mínimo sobre la grilla {0, T/m, ..., T}, para n trayectorias."""
    dt = T / m
    sx, mx, mn = [], [], []
    hecho = 0
    while hecho < n:
        c = min(chunk, n - hecho)
        x = np.zeros(c)
        a = np.zeros(c)
        b = np.zeros(c)
        for _ in range(m):
            inc = k.gamma * dt + k.sigma * math.sqrt(dt) * rng.standard_normal(c)
            cnt = rng.poisson(k.lam * dt, c)
            for j in range(1, int(cnt.max()) + 1):
                sel = cnt >= j
                q = int(sel.sum())
                inc[sel] += np.where(rng.uniform(size=q) < k.p, rng.exponential(1 / k.eta1, q),
                                     -rng.exponential(1 / k.eta2, q))
            x += inc
            np.maximum(a, x, out=a)
            np.minimum(b, x, out=b)
        sx.append(x)
        mx.append(a)
        mn.append(b)
        hecho += c
    return np.concatenate(sx), np.concatenate(mx), np.concatenate(mn)


if __name__ == "__main__":
    p = parser("Monte Carlo en grilla (exploratorio)")
    p.add_argument("--n", type=int, default=400_000)
    p.add_argument("--semilla", type=int, default=1)
    a = p.parse_args()
    cfg = cargar_cfg(a.config[0])
    t0 = time.time()
    cal = cargar_calibracion(cfg)
    d = salida(cfg)
    mc = json.loads((d / "mc_real.json").read_text(encoding="utf-8"))
    S0, T, r, dl = cal["S0"], cal["T"], cal["r"], cal["delta"]
    K = S0 * cfg["barreras"]["K_relativo"]
    kq = kou_Q(r, dl, *cal["kou"]["optimo"]["theta"]).to_float()
    ref = {f["contrato"]: f for f in mc["filas"] if abs(f["x"] - 1.0) < 1e-9}
    bU = math.log(ref["call_up"]["L"] / S0)
    bD = math.log(ref["put_down"]["L"] / S0)
    df = math.exp(-r * T)
    dias = round(T * 365)
    rng = np.random.default_rng(a.semilla)
    filas = []
    for ppd in PASOS_POR_DIA:
        m = dias * ppd
        x, mx, mn = simular_grilla(kq, T, m, a.n, rng)
        ST = S0 * np.exp(x)
        pagos = dict(call_up=df * np.maximum(ST - K, 0) * (mx < bU), put_down=df * np.maximum(K - ST, 0) * (mn > bD))
        for ct, v in pagos.items():
            media, ee = float(v.mean()), float(v.std(ddof=1) / math.sqrt(a.n))
            vi = ref[ct]["V_inversion"]
            filas.append(dict(contrato=ct, x=1.0, L=ref[ct]["L"], pasos_por_dia=ppd, m=m, V=media, error_estandar=ee,
                              sesgo=media - vi, z=abs(media - vi) / ee, sesgo_sobre_sbar=(media - vi) / cal["s_bar"]))
        print(f"  {ppd:>3} pasos por día (m = {m}): {time.time() - t0:.0f} s")
    # Si el sesgo bajara como la raíz del paso (Glasserman §6.4), pasos por día para que quede en un error estándar
    # del Monte Carlo exacto: ppd_max * (sesgo / ee_exacto)^2.
    extrap = {}
    for ct in ("call_up", "put_down"):
        f = [g for g in filas if g["contrato"] == ct and g["pasos_por_dia"] == PASOS_POR_DIA[-1]][0]
        ee_ex = ref[ct]["error_estandar"]
        extrap[ct] = dict(pasos_por_dia=PASOS_POR_DIA[-1] * (f["sesgo"] / ee_ex) ** 2, ee_exacto=ee_ex)
    res = dict(n=a.n, semilla=a.semilla, dias=dias, T=T, s_bar=cal["s_bar"], filas=filas,
               exacto={ct: dict(V=ref[ct]["V_mc"], error_estandar=ref[ct]["error_estandar"],
                                V_inversion=ref[ct]["V_inversion"], z=ref[ct]["z"]) for ct in ref},
               extrapolacion=extrap)
    (d / "mc_grilla.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    C.escribir_resumen(cfg, t0, extra=dict(n=a.n), nombre="resumen_mc_grilla.json")
    for f in filas:
        print(f["contrato"], f"{f['pasos_por_dia']:>3}/día", f"V={f['V']:.3f}±{f['error_estandar']:.3f}",
              f"inv={ref[f['contrato']]['V_inversion']:.3f}", f"sesgo={f['sesgo']:.3f} ({f['sesgo_sobre_sbar']:.2f} s̄)",
              f"z={f['z']:.1f}")
    print(extrap)
    print(f"  {time.time() - t0:.0f} s")

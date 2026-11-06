"""Auxiliar de las figuras 5.3, 5.4 y 5.5: grilla 2x2 (fecha x contrato) con dos configuraciones."""
import numpy as np
import pandas as pd

from _comun import C, RAIZ, cargar_cfg, fig_estilo, salida

TIT = {"call_up": "call up-and-out", "put_down": "put down-and-out"}
METRICAS = {"V": "precio de la knock-out (puntos del índice)", "dif_abs_S0": "|Kou − BS| (unidades de $S_0$)",
            "dif_sobre_sbar": "|Kou − BS| / spread medio", "dif_sobre_satm": "|Kou − BS| / spread ATM",
            "dif_sobre_europea": "|Kou − BS| / europea", "dif_rel": "(Kou − BS)/BS"}


def cargar(cfgs_paths):
    cfgs = [cargar_cfg(p) for p in cfgs_paths]
    datos = []
    for c in cfgs:
        d = salida(c)
        for f in ("barreras.csv", "calibracion.json"):
            if not (d / f).exists():
                raise SystemExit(f"falta {d / f}: corré calibrar.py y barreras.py para {c['_archivo']}")
        banda = pd.read_csv(d / "banda.csv") if (d / "banda.csv").exists() else None
        import json
        datos.append(dict(cfg=c, tab=pd.read_csv(d / "barreras.csv"), banda=banda,
                          cal=json.load(open(d / "calibracion.json", encoding="utf-8"))))
    d_out = salida(cfgs[0]) if len(cfgs) == 1 else RAIZ / "out" / "__".join(c["nombre"] for c in cfgs)
    d_out.mkdir(parents=True, exist_ok=True)
    return datos, d_out


def titulo_fecha(cfg):
    """dd/mm/aaaa del quote_date (el nombre de la configuración, si no es una fecha)."""
    q = str(cfg.get("datos", {}).get("quote_date", ""))
    try:
        a, m, d = q.split("-")
        return f"{d}/{m}/{a}"
    except ValueError:
        return cfg["nombre"]


def grilla(datos, titulo_fn):
    plt = fig_estilo()
    n = len(datos)
    fig, axs = plt.subplots(n, 2, figsize=(9, 3.4 * n), squeeze=False)
    for i, d in enumerate(datos):
        for j, ct in enumerate(("call_up", "put_down")):
            titulo_fn(axs[i][j], d, ct)
            axs[i][j].set_title(f"{titulo_fecha(d['cfg'])}: {TIT[ct]}", fontsize=9)
            axs[i][j].set_xlabel("$x=b/(\\sigma_{\\mathrm{atm}}\\sqrt{T})$")
    return plt, fig, axs

"""Utilidades comunes de los scripts: argumentos, preparación de datos, tablas .tex y estilo de figuras."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from calib.objetivo import Problema                     # noqa: E402
from data import config as C                            # noqa: E402
from data.cboe import ErrorDatos, leer_cadena           # noqa: E402
from data.limpieza import limpiar                       # noqa: E402


def parser(descr, multiple=False):
    p = argparse.ArgumentParser(description=descr)
    p.add_argument("--config", required=True, nargs="+" if multiple else 1, help="archivo(s) YAML")
    p.add_argument("--jobs", type=int, default=1, help="procesos en paralelo")
    return p


def cargar_cfg(path):
    cfg = C.cargar(path)
    C.verificar_real(cfg)
    return cfg


def asegurar_datos(cfg):
    arch = Path(cfg["datos"]["archivo"])
    if not arch.is_absolute():
        arch = RAIZ / arch
    if not arch.exists():
        raise SystemExit(f"[datos] falta {arch}")
    return arch


def preparar(cfg, r_delta=None):
    """Lee la cadena, limpia y arma el problema de calibración.

    r_delta: (r, delta) para reemplazar los de la paridad en la calibración (solo el análisis de sensibilidad a r);
    la limpieza no cambia.
    """
    arch = asegurar_datos(cfg)
    d = cfg["datos"]
    try:
        cad = leer_cadena(arch, d["root"], d["expiration"], d.get("quote_date"), cfg.get("hora_liquidacion"))
    except ErrorDatos as e:
        raise SystemExit(f"[datos] {e}")
    from data.tasa import r_externa
    rext = r_externa(cfg)
    df, info = limpiar(cad, cfg["limpieza"], rext["r"] if rext else None)
    info["tasa_externa"] = rext
    info["archivo"] = dict(ruta=cad.meta["archivo"], sha256=cad.meta["sha256"], n_filas=cad.meta["n_filas_archivo"],
                           n_filas_root_venc=cad.meta["n_filas_root_venc"])
    cal = cfg["calibracion"]
    r, dl = (info["r"], info["delta"]) if r_delta is None else r_delta
    prob = Problema(df, cad.S0, cad.T, r, dl, cal["pesos"], cal["N_fft"], cfg.get("cotas"))
    return cad, df, info, prob


def salida(cfg):
    return C.dir_salida(cfg)


def cargar_calibracion(cfg):
    f = salida(cfg) / "calibracion.json"
    if not f.exists():
        raise SystemExit(f"falta {f}: corré primero scripts/calibrar.py --config {cfg['_archivo']}")
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ tablas
def _fmt(x, f=None):
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "--"
        return (f or "{:.6g}").format(x)
    return str(x)


def escribir_tabla(nombre, filas, columnas, d, tex_cols=None, spec=None, caption=None, fmt=None):
    """filas: lista de dicts. Escribe <nombre>.csv (todos los dígitos) y <nombre>.tex (booktabs, para \\input)."""
    import csv
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{nombre}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        for r in filas:
            w.writerow({c: repr(r.get(c)) if isinstance(r.get(c), float) else r.get(c) for c in columnas})
    fmt = fmt or {}
    cab = tex_cols or columnas
    spec = spec or "l" * len(columnas)
    L = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(cab) + r" \\", r"\midrule"]
    for r in filas:
        L.append(" & ".join(_fmt(r.get(c), fmt.get(c)).replace("_", r"\_").replace("%", r"\%") for c in columnas) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (d / f"{nombre}.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def fig_estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt

"""Configuración YAML con los valores por defecto de las decisiones abiertas.

Ninguna decisión del autor está en el código: todas vienen de acá o del YAML, y el YAML manda.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent

DEFECTOS = {
    "limpieza": {
        "umbral_spread_rel": 0.5,
        "moneyness": [-0.25, 0.10],
        "moneyness_unidad": "log",               # log | sigma_atm (D4)
        "arbitraje_estatico_sobre": "bidask",    # bidask | mid (A1)
        "paridad_pesos": "robusta",              # robusta | ols | ponderada (A4)
        "ventana_paridad": 0.05,
        "r_fuente": "paridad",                   # paridad | externa (D3)
        "paridad_n_boot": 300,
        "paridad_semilla": 0,
    },
    "r_externa": None,                           # {serie, archivo, fecha} si r_fuente = externa
    # cotas de la optimización (A3); p se acota en logit
    "cotas": {"sigma": [0.02, 1.5], "lam": [1e-3, 40.0], "logit_p": [-6.0, 6.0],
              "eta1_menos_1": [0.05, 300.0], "eta2": [0.5, 300.0]},
    "calibracion": {
        "pesos": "b",                            # a (1/spread^2) | b (1/vega^2)
        "sigma_bs": "LS",                        # LS | ATM
        "criterio_C": "eps0",                    # eps0 | bidask | kappa
        "criterio_C_respaldo": "kappa",          # si C queda vacío (D6); None: sin respaldo
        "kappa": 1.2,
        "factor_parecido": 1.2,
        "n_inicios": 64,
        "N_fft": 2 ** 15,
        "N_fft_final": 2 ** 17,
        "semilla": 0,
    },
    "barreras": {"x_min": 0.1, "n_x": 25, "K_relativo": 1.0},
    # adaptativo: n desde n_GS hasta que dos diferencias seguidas < tol, como mucho n_max
    "inversion": {"n_GS": 14, "B": 2, "dps": 45, "paralelo": False, "adaptativo": False, "tol": 1e-6, "n_max": 100},
    # lam_max None: la cota superior de lambda de la optimización (A2)
    "perfil_lambda": {"n": 25, "lam_min": 0.05, "lam_max": None, "reinicios": 4},
    "hora_liquidacion": None,                    # None: por defecto de la raíz (a verificar en el contrato)
}


def _merge(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def cargar(path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    if "zip" in (y.get("datos") or {}) and "archivo" not in y["datos"]:      # nombre viejo de la clave
        y["datos"]["archivo"] = y["datos"].pop("zip")
    if "perfil_lambda_max" in (y.get("calibracion") or {}):
        y.setdefault("perfil_lambda", {})["lam_max"] = y["calibracion"]["perfil_lambda_max"]
    cfg = _merge(copy.deepcopy(DEFECTOS), y)
    if cfg["perfil_lambda"].get("lam_max") is None:
        cfg["perfil_lambda"]["lam_max"] = float(cfg["cotas"]["lam"][1])
    cfg["_archivo"] = str(path)
    cfg.setdefault("nombre", path.stem)
    return cfg


def dir_salida(cfg, base="out") -> Path:
    d = RAIZ / base / cfg["nombre"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bin_git():
    import shutil
    return "git.exe" if shutil.which("git.exe") else "git"      # WSL sobre /mnt/c: git.exe; Linux: git


def info_git():
    def run(*a):
        return subprocess.run([_bin_git(), *a], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "limpio": run("status", "--porcelain") == "",
                "describe": run("describe", "--tags", "--always", "--dirty")}
    except Exception:
        return {"commit": None, "limpio": None}


def escribir_resumen(cfg, t0, extra=None, nombre="resumen.json"):
    """resumen.json: hash del commit, árbol limpio, configuración completa y tiempo."""
    d = dir_salida(cfg)
    res = {"git": info_git(), "config": {k: v for k, v in cfg.items()}, "segundos": round(time.time() - t0, 2)}
    arch = Path(cfg.get("datos", {}).get("archivo", ""))
    if arch and not arch.is_absolute():
        arch = RAIZ / arch
    if arch.is_file():
        from .cboe import sha256
        res["datos_sha256"] = {str(arch.name): sha256(arch)}
    if extra:
        res.update(extra)
    with open(d / nombre, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str, ensure_ascii=False)
    return res


# ---------------------------------------------------------------- datos reales y preregistro
TAG_PREREG = "prereg-real-v1"


def _git(*a):
    r = subprocess.run([_bin_git(), *a], cwd=RAIZ, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def verificar_real(cfg):
    """Antes de correr datos reales: el archivo existe, el árbol está limpio y HEAD desciende del tag de preregistro.

    Si la fuente no es real, no hace nada. Ante cualquier problema termina con un mensaje claro.
    """
    if cfg.get("fuente") != "real":
        return
    zip_ = Path(cfg["datos"]["archivo"])
    if not zip_.is_absolute():
        zip_ = RAIZ / zip_
    if not zip_.exists():
        raise SystemExit(f"[datos reales] no está el archivo {zip_}\n"
                         "Poné el CSV (o zip) de Cboe DataShop en esa ruta (la ruta sale de la configuración).")
    if os.environ.get("CAP5_SIN_PREREGISTRO") == "1":
        # Repositorio público: el tag de preregistro vive en el historial de desarrollo y acá no existe.
        # Con esta variable no se verifica el tag ni el árbol limpio, y se avisa en cada corrida.
        print(f"[datos reales] AVISO: CAP5_SIN_PREREGISTRO=1, no se verifica el tag '{TAG_PREREG}' ni el árbol de git.",
              file=sys.stderr)
        return
    rc, _ = _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG_PREREG}")
    if rc != 0:
        raise SystemExit(f"[datos reales] falta el tag '{TAG_PREREG}': fijá todas las decisiones de la §10 en las "
                         "dos configuraciones, commiteá y poné el tag antes de correr datos reales.")
    rc, _ = _git("merge-base", "--is-ancestor", TAG_PREREG, "HEAD")
    if rc != 0:
        raise SystemExit(f"[datos reales] el commit actual no desciende de '{TAG_PREREG}'.")
    rc, out = _git("status", "--porcelain")
    if out:
        raise SystemExit("[datos reales] el árbol de git no está limpio; commiteá antes de correr.")

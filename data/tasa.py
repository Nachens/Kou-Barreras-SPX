"""Tasa libre de riesgo de la letra del Tesoro a 4 semanas (FRED, serie DTB4WK), para r_fuente = 'externa'.

DTB4WK es una tasa de descuento bancario anual, en %, con base 360: una letra de 28 días que paga 1 al
vencimiento cuesta

    precio = 1 - d * 28/360,      d = DTB4WK / 100.

La tasa continua con la convención de días de la tesis (T en años de 365 días) es la que hace
precio = e^{-r * 28/365}:

    r = -ln(precio) / (28/365).

Ejemplo a mano: d = 1,02 % da precio = 1 - 0,0102 * 28/360 = 0,999206667 y
r = -ln(0,999206667) * 365/28 = 0,01034577... .

Si el día pedido no tiene dato (feriado: FRED escribe "." o deja vacío), se usa el hábil anterior con
dato y se registra cuál.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIAS_LETRA = 28


def tasa_continua(d_pct: float, dias: int = DIAS_LETRA) -> float:
    precio = 1.0 - (d_pct / 100.0) * dias / 360.0
    return float(-np.log(precio) / (dias / 365.0))


def leer_fred(path) -> pd.Series:
    df = pd.read_csv(path)
    col_fecha = df.columns[0]
    col_val = df.columns[1]
    v = pd.to_numeric(df[col_val], errors="coerce")
    return pd.Series(v.values, index=pd.to_datetime(df[col_fecha]).dt.date, name=col_val)


def r_del_dia(path, fecha: str, dias: int = DIAS_LETRA) -> dict:
    serie = leer_fred(path)
    f = pd.Timestamp(fecha).date()
    disponibles = serie.dropna()
    previos = disponibles[disponibles.index <= f]
    if previos.empty:
        raise ValueError(f"{path}: no hay dato de {serie.name} en {fecha} ni antes")
    f_usada = previos.index[-1]
    d = float(previos.iloc[-1])
    return dict(serie=serie.name, fecha_pedida=str(f), fecha_usada=str(f_usada), hubo_que_retroceder=f_usada != f,
                d_pct=d, precio=1.0 - (d / 100.0) * dias / 360.0, r=tasa_continua(d, dias), archivo=str(path))


def r_externa(cfg) -> dict | None:
    """Con limpieza.r_fuente = 'externa', lee r del CSV de FRED que indica cfg['r_externa']."""
    if cfg.get("limpieza", {}).get("r_fuente", "paridad") != "externa":
        return None
    re_ = cfg.get("r_externa") or {}
    if "valor" in re_ and "archivo" not in re_:          # valor fijado a mano
        return dict(serie=re_.get("serie"), fecha_usada=re_.get("fecha"), r=float(re_["valor"]), archivo=None)
    from .config import RAIZ
    p = Path(re_["archivo"])
    if not p.is_absolute():
        p = RAIZ / p
    if not p.exists():
        raise SystemExit(f"[tasa] falta {p}: bajá la serie {re_.get('serie')} de FRED (ver el README)")
    return r_del_dia(p, re_.get("fecha") or cfg["datos"]["quote_date"])

"""Lector del producto "Option EOD Summary" con Calcs de Cboe DataShop.

Archivos UnderlyingOptionsEODCalcs_YYYY-MM-DD.csv, o el mismo CSV dentro de un .zip; el formato se elige por la extensión. El zip se lee sin descomprimir a disco. Se usa el snapshot de las 15:45. `implied_volatility_1545` es del modelo del proveedor (tasa
y dividendos desconocidos): **no** se usa para calibrar, solo se reporta la diferencia.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .tiempo import T_anios

COLUMNAS_CBOE = [
    "underlying_symbol", "quote_date", "root", "expiration", "strike", "option_type", "open", "high", "low",
    "close", "trade_volume", "bid_size_1545", "bid_1545", "ask_size_1545", "ask_1545", "underlying_bid_1545",
    "underlying_ask_1545", "implied_underlying_price_1545", "active_underlying_price_1545",
    "implied_volatility_1545", "delta_1545", "gamma_1545", "theta_1545", "vega_1545", "rho_1545",
    "bid_size_eod", "bid_eod", "ask_size_eod", "ask_eod", "underlying_bid_eod", "underlying_ask_eod",
    "vwap", "open_interest", "delivery_code",
]
USADAS = ["underlying_symbol", "quote_date", "root", "expiration", "strike", "option_type", "bid_size_1545",
          "bid_1545", "ask_size_1545", "ask_1545", "active_underlying_price_1545", "implied_volatility_1545",
          "trade_volume", "open_interest"]


@dataclass
class Cadena:
    S0: float
    quote_date: str
    expiration: str
    root: str
    T: float
    df: pd.DataFrame          # strike, tipo, bid, ask, mid, iv_prov, volumen, oi
    meta: dict = field(default_factory=dict)


class ErrorDatos(RuntimeError):
    pass


_DTYPES = {"root": str, "option_type": str, "underlying_symbol": str, "expiration": str, "quote_date": str}


def _verificar_columnas(cab, origen):
    faltan = [c for c in USADAS if c not in cab]
    if faltan:
        raise ErrorDatos(f"{origen}: faltan columnas {faltan}")


def csv_en_zip(zip_path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        nombres = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not nombres:
            raise ErrorDatos(f"{zip_path}: el zip no tiene un CSV")
        with z.open(nombres[0]) as f:
            cab = pd.read_csv(io.BytesIO(f.read(4096).split(b"\n")[0] + b"\n"), nrows=0).columns
        _verificar_columnas(cab, zip_path)
        with z.open(nombres[0]) as f:
            return pd.read_csv(f, usecols=USADAS, dtype=_DTYPES)


def csv_plano(csv_path) -> pd.DataFrame:
    _verificar_columnas(pd.read_csv(csv_path, nrows=0).columns, csv_path)
    return pd.read_csv(csv_path, usecols=USADAS, dtype=_DTYPES)


def leer_archivo(path) -> pd.DataFrame:
    """.csv o .zip según la extensión."""
    ext = Path(path).suffix.lower()
    if ext == ".zip":
        return csv_en_zip(path)
    if ext == ".csv":
        return csv_plano(path)
    raise ErrorDatos(f"{path}: extensión {ext!r} no soportada (.csv o .zip)")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_cadena(path, root="SPX", expiration=None, quote_date=None, hora_liquidacion=None,
                simbolo="^SPX") -> Cadena:
    raw = leer_archivo(path)
    n_total = len(raw)
    raw = raw[raw["underlying_symbol"] == simbolo]
    if quote_date is not None:
        raw = raw[raw["quote_date"] == str(quote_date)]
    raices = sorted(raw["root"].unique())
    raw = raw[raw["root"] == root]
    if raw.empty:
        raise ErrorDatos(f"no hay filas con root={root} (raíces disponibles: {raices})")
    vencs = sorted(raw["expiration"].unique())
    if expiration is None or str(expiration) not in vencs:
        raise ErrorDatos(f"vencimiento {expiration} no está para root={root} (disponibles: {vencs})")
    raw = raw[raw["expiration"] == str(expiration)].copy()
    qd = raw["quote_date"].iloc[0]
    S = raw["active_underlying_price_1545"].astype(float)
    S0 = float(S.median())
    meta = dict(archivo=str(path), sha256=sha256(path), n_filas_archivo=n_total, raices_archivo=raices,
                vencimientos_root=vencs, n_filas_root_venc=int(len(raw)),
                S0_min=float(S.min()), S0_max=float(S.max()), S0_constante=bool(S.max() - S.min() < 1e-9))
    df = pd.DataFrame({
        "strike": raw["strike"].astype(float).values,
        "tipo": raw["option_type"].str.upper().values,
        "bid": raw["bid_1545"].astype(float).values,
        "ask": raw["ask_1545"].astype(float).values,
        "iv_prov": raw["implied_volatility_1545"].astype(float).values,
        "volumen": raw["trade_volume"].values,
        "oi": raw["open_interest"].values,
    })
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df.sort_values(["strike", "tipo"]).reset_index(drop=True)
    T = T_anios(qd, str(expiration), hora_liquidacion, root)
    return Cadena(S0=S0, quote_date=qd, expiration=str(expiration), root=root, T=T, df=df, meta=meta)

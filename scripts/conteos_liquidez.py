"""Conteos de liquidez y de moneyness. No cambia ningún
filtro ni recalibra: relee la cadena con la limpieza preregistrada y cuenta.

Liquidez: de las europeas que entran en la calibración, cuántas tienen volumen negociado nulo (columna `trade_volume` del
archivo de Cboe) y cuántas interés abierto nulo (`open_interest`), y cuánto pesan en los pesos 1/vega^2 y en eps0^2.
Moneyness: en qué rango de x_K = (log(K/S0) - (r - delta)T)/(sigma_atm sqrt T) hay cotizaciones out of the money con bid > 0
y spread relativo <= umbral (los dos filtros previos al de moneyness), de cada lado del precio a término.

  python scripts/conteos_liquidez.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml

Escribe out/<config>/conteos_liquidez.json.
"""
import json
import time

import numpy as np

from _comun import C, cargar_cfg, parser, preparar, salida
from data.cboe import leer_cadena

if __name__ == "__main__":
    a = parser("conteos de liquidez y de moneyness", multiple=True).parse_args()
    for path in a.config:
        cfg = cargar_cfg(path)
        t0 = time.time()
        cad, df, info, prob = preparar(cfg)
        # opciones de la calibración
        vol = df["volumen"].to_numpy(float)
        oi = df["oi"].to_numpy(float)
        w = prob.w
        spr = prob.ask - prob.bid
        e0 = w * spr ** 2
        res = dict(n=int(len(df)))
        for nombre, m in (("volumen_nulo", vol == 0), ("oi_nulo", oi == 0), ("ambos_nulos", (vol == 0) & (oi == 0))):
            res[nombre] = dict(n=int(m.sum()), frac_pesos=float(w[m].sum() / w.sum()), frac_eps0=float(e0[m].sum() / e0.sum()))
        # cotizaciones OTM que pasan bid > 0 y spread relativo, antes del filtro de moneyness
        S0, T = cad.S0, cad.T
        r, dl = info["r"], info["delta"]
        F = S0 * np.exp((r - dl) * T)
        s_atm = info["sigma_atm"]
        c = cad.df
        ok = (c.bid > 0) & (c.bid <= c.ask)
        ok &= (c.ask - c.bid) / c.mid <= cfg["limpieza"]["umbral_spread_rel"]
        otm = ((c.tipo == "P") & (c.strike < F)) | ((c.tipo == "C") & (c.strike > F))
        d = c[ok & otm]
        x = (np.log(d.strike / S0) - (r - dl) * T) / (s_atm * np.sqrt(T))
        mmin, mmax = cfg["limpieza"]["moneyness"]
        res_x = dict(x_min_puts=float(x[d.tipo == "P"].min()), x_max_calls=float(x[d.tipo == "C"].max()),
                     n_puts=int((d.tipo == "P").sum()), n_calls=int((d.tipo == "C").sum()),
                     n_debajo=int((x < mmin).sum()), n_encima=int((x > mmax).sum()), intervalo=[mmin, mmax])
        out = dict(calibracion=res, moneyness=res_x)
        (salida(cfg) / "conteos_liquidez.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        C.escribir_resumen(cfg, t0, extra=out, nombre="resumen_conteos_liquidez.json")
        print(cfg["nombre"], json.dumps(out))

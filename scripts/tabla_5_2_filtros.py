"""Tabla 5.2: cotizaciones eliminadas por cada filtro. Por configuración, dos columnas: el modo del filtro de arbitraje
que se usa (`arbitraje_estatico_sobre`, bidask por defecto) y lo que eliminaría el otro modo (A1). Los filtros 1-5 no
dependen del modo, así que las dos columnas difieren solo en el arbitraje y en las que quedan."""
import time

from _comun import C, cargar_cfg, escribir_tabla, parser, preparar, salida

JUST = {"bid_nulo": "Bid positivo", "cruzadas": "Cotizaciones cruzadas (bid > ask)", "spread": "Spread relativo",
        "moneyness": "Moneyness", "no_OTM": "Solo out of the money", "arbitraje": "Arbitraje estático"}

if __name__ == "__main__":
    a = parser("Tabla 5.2", multiple=True).parse_args()
    cfgs = [cargar_cfg(p) for p in a.config]
    t0 = time.time()
    infos = [preparar(c)[2] for c in cfgs]
    filas = [{"filtro": "Cotizaciones del vencimiento", **{f"fecha_{i + 1}": inf["n_inicial"] for i, inf in enumerate(infos)},
              **{f"fecha_{i + 1}_otro": inf["n_inicial"] for i, inf in enumerate(infos)}}]
    for k, nombre in JUST.items():
        f = {"filtro": nombre}
        for i, inf in enumerate(infos):
            f[f"fecha_{i + 1}"] = inf["cuentas"][k]
            f[f"fecha_{i + 1}_otro"] = inf["cuentas_otro_modo"].get(k, inf["cuentas"][k])
        filas.append(f)
    tot = {"filtro": "Quedan"}
    for i, inf in enumerate(infos):
        tot[f"fecha_{i + 1}"] = inf["n_final"]
        tot[f"fecha_{i + 1}_otro"] = inf["n_final"] + inf["cuentas"]["arbitraje"] - inf["cuentas_otro_modo"]["arbitraje"]
    filas.append(tot)
    cols = ["filtro"] + sum(([f"fecha_{i + 1}", f"fecha_{i + 1}_otro"] for i in range(len(infos))), [])
    d = salida(cfgs[0]) if len(cfgs) == 1 else C.RAIZ / "out" / "__".join(c["nombre"] for c in cfgs)
    cab = ["Filtro"]
    for c, inf in zip(cfgs, infos):
        cab += [f"{c['nombre']} ({inf['arbitraje_modo']})", f"{c['nombre']} ({inf['arbitraje_otro_modo']}, alternativa)"]
    escribir_tabla("tabla_5_2_filtros", filas, cols, d, tex_cols=cab, spec="l" + "r" * (2 * len(infos)))
    print(d / "tabla_5_2_filtros.tex")
    for f in filas:
        print(f)

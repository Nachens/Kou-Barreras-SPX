"""Tabla 5.1: fecha, S0, vencimiento, raíz, T, opciones antes y después de filtrar."""
import time

from _comun import C, cargar_cfg, escribir_tabla, parser, preparar, salida

if __name__ == "__main__":
    a = parser("Tabla 5.1").parse_args()
    cfg = cargar_cfg(a.config[0])
    t0 = time.time()
    cad, df, info, prob = preparar(cfg)
    fila = dict(fecha=cad.quote_date, S0=cad.S0, vencimiento=cad.expiration, raiz=cad.root, T=cad.T,
                antes=info["n_inicial"], despues=info["n_final"], r=info["r"], delta=info["delta"],
                S0_min=cad.meta["S0_min"], S0_max=cad.meta["S0_max"])
    cols = ["fecha", "S0", "vencimiento", "raiz", "T", "antes", "despues", "r", "delta", "S0_min", "S0_max"]
    escribir_tabla("tabla_5_1_datos", [fila], cols, salida(cfg),
                   tex_cols=["Fecha", "$S_0$", "Vencimiento", "Raíz", "$T$ (años)", "Antes de filtrar",
                             "Después de filtrar", "$r$", "$\\delta$", "min $S_0$", "max $S_0$"],
                   spec="cllcrrrrrrr", fmt={"S0": "{:.2f}", "T": "{:.6f}", "r": "{:.5f}", "delta": "{:.5f}",
                                            "S0_min": "{:.2f}", "S0_max": "{:.2f}"})
    if not cad.meta["S0_constante"]:
        print(f"AVISO: active_underlying_price_1545 varía entre {cad.meta['S0_min']} y {cad.meta['S0_max']}; se usó la mediana")
    C.escribir_resumen(cfg, t0, nombre="resumen_tabla_5_1.json")
    print(fila)

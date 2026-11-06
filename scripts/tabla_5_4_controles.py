"""Tabla 5.4: controles numéricos."""
import argparse
import time

from _comun import C, RAIZ, escribir_tabla

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=1_000_000, help="trayectorias del Monte Carlo (el test usa 10^7)")
    p.add_argument("--n-casos", type=int, default=6)
    p.add_argument("--b35", type=int, default=5, help="casos aleatorios del control B.35 (la tabla usa 20; cada caso tarda ~1 min)")
    p.add_argument("--jobs", type=int, default=10)
    a = p.parse_args()
    t0 = time.time()
    from kou import controles as ct
    filas = []
    f, errs = ct.control_lambda0(ct.casos_lambda0(a.n_casos))
    filas += f
    filas += ct.control_a_igual_b()
    filas += ct.control_kou_wang()
    filas += ct.control_europeas()
    f, mc = ct.control_mc(a.mc)
    filas += f
    from kou import b35
    filas += b35.control_b35(n_aleatorios=a.b35, jobs=a.jobs)
    d = RAIZ / "out" / "controles"
    escribir_tabla("tabla_5_4_controles", filas, ["control", "que", "error_max", "tolerancia", "ok"], d,
                   tex_cols=["Control", "Qué prueba", "Error máximo", "Tolerancia", "Pasa"],
                   spec="p{0.28\\textwidth}p{0.36\\textwidth}rrc", fmt={"error_max": "{:.3e}", "tolerancia": "{:.1e}"})
    import json
    from data.config import info_git
    (d / "resumen.json").write_text(json.dumps(dict(git=info_git(), segundos=round(time.time() - t0, 1), n_mc=a.mc, n_b35=a.b35,
                                                   lambda0_errores={str(k): v for k, v in errs.items()}, mc=mc),
                                              indent=2, default=str), encoding="utf-8")
    for f in filas:
        print(("OK  " if f["ok"] else "FALLA"), f["control"], "|", f["que"], "|", f"{f['error_max']:.3e}", "<=", f["tolerancia"])

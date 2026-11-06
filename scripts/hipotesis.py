"""Evaluación de la hipótesis de §5.4.1. Sin interpretación: cada parte con los números
que la deciden y un veredicto 'cumple', 'no cumple' o 'no evaluable'.

  python scripts/hipotesis.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml

Las configuraciones se identifican por la clave `escenario` (calma | estres). Definiciones (tesis, §5.4.1, y las
decisiones D5-D7 del 29/10):
  Delta_max: máximo sobre la grilla de |V_Kou - V_BS| (Kou en theta-sombrero), por contrato y fecha;
  s̄: spread medio (ask - bid) de las europeas usadas para calibrar en esa fecha;
  (a) en estrés, Delta_max > s̄ para los dos contratos;
  (b) para los dos contratos, Delta_max(estrés) > Delta_max(calma);
  (c) en cada fecha y contrato, la fracción de cruces por salto (en theta-sombrero) en la distancia de Delta_max es
      >= umbral_c (0,10);
  (d) en estrés, para los dos contratos, el ancho máximo de la banda de Kou sobre C supera s̄; si C es vacío con
      eps0 se usa el criterio de respaldo (kappa) y se registra.
Escribe out/hipotesis.json (o out/hipotesis_<sufijo>.json con --sufijo).
"""
import json
import time

from _comun import C, cargar_cfg, parser, salida

CONTRATOS = ("call_up", "put_down")


def _v(ok):
    return "cumple" if ok else "no cumple"


if __name__ == "__main__":
    p = parser("hipótesis", multiple=True)
    p.add_argument("--umbral-c", type=float, default=None, help="por defecto, el de la configuración (hipotesis.umbral_c)")
    p.add_argument("--sufijo", default="")
    a = p.parse_args()
    cfgs = [cargar_cfg(c) for c in a.config]
    t0 = time.time()
    por = {}
    for cfg in cfgs:
        esc = cfg.get("escenario")
        if esc not in ("calma", "estres"):
            raise SystemExit(f"{cfg['_archivo']}: falta la clave escenario (calma | estres)")
        q = json.loads((salida(cfg) / "hipotesis_cantidades.json").read_text(encoding="utf-8"))
        por[esc] = dict(cfg=cfg, q=q)
    if set(por) != {"calma", "estres"}:
        raise SystemExit("hacen falta una configuración de calma y una de estrés")
    umbral_c = a.umbral_c if a.umbral_c is not None else por["estres"]["cfg"].get("hipotesis", {}).get("umbral_c", 0.10)
    qc, qe = por["calma"]["q"], por["estres"]["q"]
    out = dict(configuraciones={k: v["cfg"]["nombre"] for k, v in por.items()}, umbral_c=umbral_c,
               s_bar=dict(calma=qc["s_bar"], estres=qe["s_bar"]), s_atm=dict(calma=qc["s_atm"], estres=qe["s_atm"]))

    # (a)
    na = {ct: dict(delta_max=qe[ct]["delta_max"], s_bar=qe["s_bar"], cociente=qe[ct]["delta_max"] / qe["s_bar"],
                   cumple=qe[ct]["delta_max"] > qe["s_bar"]) for ct in CONTRATOS}
    out["a"] = dict(enunciado="en estrés, Δmax > s̄ para los dos contratos", numeros=na,
                    veredicto=_v(all(v["cumple"] for v in na.values())))
    # (b)
    nb = {ct: dict(delta_max_estres=qe[ct]["delta_max"], delta_max_calma=qc[ct]["delta_max"],
                   cumple=qe[ct]["delta_max"] > qc[ct]["delta_max"]) for ct in CONTRATOS}
    out["b"] = dict(enunciado="para los dos contratos, Δmax es mayor en estrés que en calma", numeros=nb,
                    veredicto=_v(all(v["cumple"] for v in nb.values())))
    # (c)
    nc = {}
    for esc, qq in (("calma", qc), ("estres", qe)):
        for ct in CONTRATOS:
            f = qq[ct]["frac_salto_en_delta_max"]
            nc[f"{esc}_{ct}"] = dict(x_en_delta_max=qq[ct]["x_en_delta_max"], frac_salto=f, cumple=f >= umbral_c)
    out["c"] = dict(enunciado=f"en cada fecha y contrato, la fracción de cruces por salto en la distancia de Δmax es ≥ {umbral_c}",
                    numeros=nc, veredicto=_v(all(v["cumple"] for v in nc.values())))
    # (d)
    nd, evaluable = {}, True
    for ct in CONTRATOS:
        if "ancho_banda_max" not in qe[ct]:
            evaluable = False
            nd[ct] = dict(ancho_banda_max=None, s_bar=qe["s_bar"], cumple=None)
            continue
        nd[ct] = dict(ancho_banda_max=qe[ct]["ancho_banda_max"], x_en_ancho_max=qe[ct]["x_en_ancho_max"], s_bar=qe["s_bar"],
                      cociente=qe[ct]["ancho_banda_max"] / qe["s_bar"], cumple=qe[ct]["ancho_banda_max"] > qe["s_bar"])
    out["d"] = dict(enunciado="en estrés, para los dos contratos, el ancho máximo de la banda de Kou sobre C supera s̄",
                    criterio_C=qe.get("criterio_C"), criterio_banda=qe.get("criterio_banda"),
                    se_uso_respaldo=qe.get("criterio_banda") not in (None, qe.get("criterio_C")),
                    n_theta_banda=qe.get("n_theta_banda"), numeros=nd,
                    veredicto=_v(all(v["cumple"] for v in nd.values())) if evaluable else "no evaluable")
    out["segundos"] = round(time.time() - t0, 2)
    from data.config import info_git
    out["git"] = info_git()
    f = C.RAIZ / "out" / f"hipotesis{a.sufijo}.json"
    f.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in "abcd":
        print(f"({k}) {out[k]['veredicto']}: {out[k]['enunciado']}")
    print(f)

"""Tasa de la letra a 4 semanas (FRED, DTB4WK) del día de cada configuración con r_fuente = 'externa'.

Lee el CSV de FRED que indica `r_externa.archivo` (ver el README), toma el dato
de la fecha (o del hábil anterior, y lo dice) y lo pasa a tasa continua (data/tasa.py). Escribe
out/<config>/tasa_externa.json. El valor no se escribe en el YAML: el YAML queda congelado por el tag de
preregistro; la limpieza lo vuelve a leer del mismo CSV con la misma función.

  python scripts/tasa_externa.py --config config/real_2017-10-11.yaml config/real_2020-03-16.yaml
"""
import json

from _comun import cargar_cfg, parser, salida
from data.cboe import sha256
from data.tasa import r_externa

if __name__ == "__main__":
    a = parser("tasa externa", multiple=True).parse_args()
    for c in a.config:
        cfg = cargar_cfg(c)
        t = r_externa(cfg)
        if t is None:
            print(f"{cfg['nombre']}: r_fuente = paridad, no hace falta")
            continue
        t["sha256_archivo"] = sha256(t["archivo"]) if t.get("archivo") else None
        (salida(cfg) / "tasa_externa.json").write_text(json.dumps(t, indent=2, default=str), encoding="utf-8")
        print(f"{cfg['nombre']}: {t['serie']} del {t['fecha_usada']} = {t['d_pct']} % (descuento, base 360) -> "
              f"precio {t['precio']:.9f} -> r = {t['r']:.8f}" + ("  [hábil anterior]" if t["hubo_que_retroceder"] else ""))

"""Corre todo lo del Cap. 5 para una o dos configuraciones.

  python scripts/correr_todo.py --config config/real_2017-10-11.yaml [config/real_2020-03-16.yaml] [--jobs N]

Por configuración: Tablas 5.1 y 5.3, calibración (Tabla 5.5), barreras y banda, Figuras 5.1 y 5.2. Con dos configuraciones
además Tabla 5.2 y Figuras 5.3, 5.4 y 5.5. La Tabla 5.4 (controles) es independiente de los datos: tabla_5_4_controles.py.
Sin el archivo de datos (o sin el tag de preregistro, ver el README) termina con un mensaje claro.
"""
import subprocess
import sys
import time
from pathlib import Path

from _comun import RAIZ, cargar_cfg, parser

S = Path(__file__).resolve().parent


def correr(script, *args):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(S / script), *map(str, args)], cwd=str(RAIZ))
    if r.returncode != 0:
        raise SystemExit(f"falló {script} (código {r.returncode})")
    print(f"--- {script}: {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    a = parser("correr todo", multiple=True).parse_args()
    for c in a.config:
        cargar_cfg(c)        # falla acá con mensaje claro si faltan datos reales o el preregistro
    for c in a.config:
        for sc in ("tabla_5_1_datos.py", "calibrar.py", "barreras.py", "tabla_5_3_convergencia.py",
                   "figura_5_1_sonrisa.py", "figura_5_2_perfil.py"):
            correr(sc, "--config", c, "--jobs", a.jobs)
    correr("tabla_5_2_filtros.py", "--config", *a.config)
    for sc in ("figura_5_3_barreras.py", "figura_5_4_cruce.py", "figura_5_5_banda.py"):
        correr(sc, "--config", *a.config)

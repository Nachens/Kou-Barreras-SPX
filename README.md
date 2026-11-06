# Barreras bajo el modelo de Kou: calibración al SPX y valuación

Código que reproduce los resultados del Capítulo 5 de la tesis de Licenciatura en Ciencias de Datos *Valuación de
opciones barrera en modelos de difusión con saltos* (Ignacio Ezequiel Arroyo, Universidad de Buenos Aires, 2026):
calibración de Black–Scholes y de Kou a opciones europeas sobre el S&P 500 en dos fechas, y valuación de una call
up-and-out y una put down-and-out por inversión de Laplace (Gaver–Stehfest, precisión arbitraria).

## Datos

Los datos no se incluyen. Los de Cboe DataShop tienen licencia de uso propio. Van en una carpeta `datos/` (ignorada por
git), con estos nombres y SHA-256:

| Archivo | Fuente | SHA-256 |
|---|---|---|
| `datos/UnderlyingOptionsEODCalcs_2017-10-11.csv` | Cboe DataShop, *Option EOD Summary* con cálculos, `^SPX`, 11/10/2017 | `3954b3faa62938b323af7bcbcdf07814b95190f9cdaa88dd7f4ec40c8de9a735` |
| `datos/UnderlyingOptionsEODCalcs_2020-03-16.csv` | ídem, 16/03/2020 | `40c57176df42af70b4e8f0f35a3b63a099563b97f3eabab5dd57d33470fa146a` |
| `datos/externos/DTB4WK_consulta-2026-10-29.csv` | FRED, serie `DTB4WK`: <https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB4WK> | `d19b2712c65611ca0ceb1eb8c1310c75bec2a3d91463bf5cd093fc461c55434b` |
| `datos/externos/VIX_History_consulta-2026-10-29.csv` | Cboe: <https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv> | `b0d2e90197b4b5403fd689fc580eb2506f07c867f0115034810c7218fc986b7f` |

Las dos últimas series se actualizan a diario, así que un archivo
bajado después tiene más filas y otro hash; los valores de las dos fechas no cambian.

## Instalación

Python ≥ 3.11 (las salidas se corrieron con 3.14.4).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Cómo correrlo

```bash
export CAP5_SIN_PREREGISTRO=1       # ver la nota de abajo
R="config/real_2017-10-11.yaml config/real_2020-03-16.yaml"
python scripts/tasa_externa.py --config $R                      # r de la letra a 4 semanas
python scripts/correr_todo.py --config $R --jobs 4              # tablas 5.1 a 5.3, calibración, barreras, figuras
for c in $R; do python scripts/mc_real.py --config $c; python scripts/sensibilidad_r.py --config $c --jobs 4; done
python scripts/hipotesis.py --config $R
python scripts/banda_kappa.py --config $R --jobs 4
python scripts/bidask_en_C.py --config $R
python scripts/conteos_liquidez.py --config $R
python scripts/cantidades_adicionales.py --config $R
python scripts/tabla_5_4_controles.py --mc 10000000 --b35 20 --jobs 4     # controles numéricos, no usa datos
```

Todo sale en `out/<configuración>/` (CSV con todos los dígitos, `.tex` de las tablas y figuras en PDF) y en
`out/<config1>__<config2>/` lo que combina las dos fechas. Cada corrida guarda en un `resumen*.json` el commit, la
configuración completa y el SHA-256 del archivo de datos que usó.

## Estructura

```
kou/     modelo, raíces de G(β) = α, transformadas, inversión de Gaver–Stehfest, barreras, europeas (Carr–Madan),
         forma cerrada de Kou–Wang y Monte Carlo (controles)
bs/      Black–Scholes, volatilidad implícita, barreras de Rubinstein–Reiner
data/    lector de Cboe, limpieza, paridad put–call, tasa, tiempo, configuración
calib/   función objetivo, optimización con varios inicios, identificabilidad, banda
config/  una configuración por fecha
scripts/ un script por tabla o figura
```

## Nota sobre el preregistro

Las decisiones de especificación (fechas, filtros, pesos, criterios e hipótesis) están en `config/` y se fijaron antes
de correr datos reales. En el repositorio de desarrollo se congelaron con un tag de git, y el código se negaba a correr
datos reales si el tag no existía o si el árbol no estaba limpio. Este repositorio no tiene ese historial, así que
`CAP5_SIN_PREREGISTRO=1` omite esa verificación (y avisa en cada corrida). Es el único cambio de código respecto del
repositorio de desarrollo (`data/config.py`).

## Licencia y cómo citar

MIT (ver `LICENSE`). Los datos de Cboe tienen su propia licencia y no forman parte de este repositorio.

Arroyo, I. E. (2026). *Valuación de opciones barrera en modelos de difusión con saltos*. Tesis de Licenciatura en Ciencias de Datos, Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires.

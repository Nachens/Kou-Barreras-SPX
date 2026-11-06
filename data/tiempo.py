"""Tiempo hasta el vencimiento (§5.1.4 de la tesis).

T = (momento de liquidación - 15:45 ET del quote_date) / 365 días, con la diferencia medida en tiempo
transcurrido (en UTC): si entre las dos fechas cambia el horario de verano, la hora que se gana o se pierde
cuenta. (Restar dos datetime con la misma tzinfo ignora el desfase UTC; por eso se pasa a UTC antes.)
La hora de liquidación depende de la raíz y está **a verificar en la especificación del contrato**:
por defecto 09:30 ET del día de vencimiento para SPX (tercer viernes, liquidación AM) y 16:00 ET para
SPXW. Sale de la configuración (`hora_liquidacion`).
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HORA_COTIZACION = time(15, 45)
HORA_LIQUIDACION_DEFECTO = {"SPX": "09:30", "SPXW": "16:00"}   # a verificar en el contrato


def _hora(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def hora_liquidacion_defecto(root: str) -> str:
    return HORA_LIQUIDACION_DEFECTO.get(root, "16:00")


def T_anios(quote_date: str, expiration: str, hora_liquidacion: str | None = None, root: str = "SPX") -> float:
    hl = hora_liquidacion or hora_liquidacion_defecto(root)
    q = datetime.combine(datetime.fromisoformat(str(quote_date)).date(), HORA_COTIZACION, tzinfo=ET)
    e = datetime.combine(datetime.fromisoformat(str(expiration)).date(), _hora(hl), tzinfo=ET)
    dias = (e.astimezone(timezone.utc) - q.astimezone(timezone.utc)).total_seconds() / 86400.0
    if dias <= 0:
        raise ValueError(f"vencimiento {expiration} {hl} no es posterior a la cotización {quote_date}")
    return dias / 365.0

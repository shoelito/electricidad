"""Lecturas sintéticas en kW con patrón diario y picos, para probar el pipeline mientras no hay datos reales."""

from datetime import datetime

import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "America/Costa_Rica"
BASE_POWER_KW = 0.12
MORNING_PEAK_HOURS = (6.0, 7.5)
EVENING_PEAK_HOURS = (18.0, 20.5)
PEAK_EXTRA_POWER_KW = 1.08
DAYTIME_EXTRA_POWER_KW = 0.18
NOISE_STANDARD_DEVIATION_KW = 0.036


def generate_synthetic_readings(
    start: datetime, days: float, interval_seconds: int = 60, random_seed: int = 7
) -> pd.DataFrame:
    """DataFrame con columnas measured_at (UTC) y power_kw. El sensor real envía una lectura por segundo;
    por defecto se generan cada 60 s para que las pruebas sean rápidas."""
    random_generator = np.random.default_rng(random_seed)
    measured_at = pd.date_range(
        start=pd.Timestamp(start).tz_convert("UTC"),
        periods=int(days * 86_400 / interval_seconds),
        freq=f"{interval_seconds}s",
    )
    local_time = measured_at.tz_convert(LOCAL_TIMEZONE)
    hour_of_day = local_time.hour + local_time.minute / 60
    is_weekend = local_time.day_of_week >= 5

    power = np.full(len(measured_at), BASE_POWER_KW)
    power += np.where((hour_of_day >= 8) & (hour_of_day < 22), DAYTIME_EXTRA_POWER_KW, 0.0)
    morning_peak = (hour_of_day >= MORNING_PEAK_HOURS[0]) & (hour_of_day < MORNING_PEAK_HOURS[1]) & ~is_weekend
    evening_peak = (hour_of_day >= EVENING_PEAK_HOURS[0]) & (hour_of_day < EVENING_PEAK_HOURS[1])
    power += np.where(morning_peak | evening_peak, PEAK_EXTRA_POWER_KW, 0.0)
    power += random_generator.normal(0, NOISE_STANDARD_DEVIATION_KW, len(measured_at))

    return pd.DataFrame({"measured_at": measured_at, "power_kw": np.clip(power, 0.0, None)})

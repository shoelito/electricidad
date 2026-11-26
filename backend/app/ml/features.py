"""Transformación de lecturas a features. La usan TANTO el entrenamiento como la predicción en vivo,
para que ambos calculen las entradas del modelo exactamente igual."""

import pandas as pd

INTERVAL_MINUTES = 7
INTERVAL_SECONDS = INTERVAL_MINUTES * 60
RESAMPLING_INTERVAL = f"{INTERVAL_MINUTES}min"
INTERVAL_DURATION = pd.Timedelta(minutes=INTERVAL_MINUTES)
# Todos los intervalos se cuentan desde 1970-01-01 UTC ("epoch"), igual que la agregación en SQLite y el
# cálculo del intervalo actual. 7 min no divide exacto el día, así que los intervalos empiezan en minutos
# distintos cada día (por ejemplo 18:03, 18:10...).
RESAMPLING_ORIGIN = "epoch"

# 7 min no divide exacto la hora ni el día: las duraciones se redondean al número de intervalos más cercano.
INTERVALS_PER_HOUR = round(60 / INTERVAL_MINUTES)  # 9 intervalos = 63 min
INTERVALS_PER_DAY = round(24 * 60 / INTERVAL_MINUTES)  # 206 intervalos = 24 h 2 min
MAX_INTERPOLATED_INTERVALS = INTERVALS_PER_HOUR  # rellenar huecos de hasta ~1 h
FORECAST_HORIZON_INTERVALS = INTERVALS_PER_HOUR  # ~1 hora adelante (63 min)
LAG_INTERVALS = (1, 2, INTERVALS_PER_HOUR, INTERVALS_PER_DAY)  # 7 min, 14 min, ~1 h, ~misma hora de ayer
HISTORY_NEEDED_FOR_FEATURES = INTERVAL_DURATION * (INTERVALS_PER_DAY + INTERVALS_PER_HOUR) + pd.Timedelta(hours=1)

POWER_COLUMN = "power_kw"
TARGET_COLUMN = "target_power_kw"
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    POWER_COLUMN,
    *(f"power_lag_{lag}" for lag in LAG_INTERVALS),
    "power_rolling_mean_1h",
    "power_rolling_max_1h",
]


def resample_power_series(readings: pd.DataFrame) -> pd.Series:
    """Lecturas crudas (measured_at UTC, power_kw cada segundo) -> potencia media en kW cada 7 min.

    Solo interpola huecos cortos (hasta ~1 h); los huecos largos quedan como NaN.
    """
    if readings.empty:
        return pd.Series(dtype="float64", name=POWER_COLUMN, index=pd.DatetimeIndex([], tz="UTC"))
    power_by_time = readings.set_index("measured_at")[POWER_COLUMN].sort_index()
    resampled_power = power_by_time.resample(RESAMPLING_INTERVAL, origin=RESAMPLING_ORIGIN).mean()
    interpolated_power = resampled_power.interpolate(limit=MAX_INTERPOLATED_INTERVALS, limit_area="inside")
    return interpolated_power.rename(POWER_COLUMN)


def build_feature_frame(power_series: pd.Series, local_timezone: str) -> pd.DataFrame:
    """Una fila por intervalo de 7 min con todas las FEATURE_COLUMNS (pueden quedar NaN)."""
    local_index = power_series.index.tz_convert(local_timezone)
    features = pd.DataFrame(index=power_series.index)
    features["hour"] = local_index.hour + local_index.minute / 60
    features["day_of_week"] = local_index.day_of_week
    features["is_weekend"] = (local_index.day_of_week >= 5).astype(int)
    features[POWER_COLUMN] = power_series
    for lag in LAG_INTERVALS:
        features[f"power_lag_{lag}"] = power_series.shift(lag)
    rolling_window = power_series.rolling(INTERVALS_PER_HOUR, min_periods=1)
    features["power_rolling_mean_1h"] = rolling_window.mean()
    features["power_rolling_max_1h"] = rolling_window.max()
    return features[FEATURE_COLUMNS]


def build_training_dataset(power_series: pd.Series, local_timezone: str) -> tuple[pd.DataFrame, pd.Series]:
    """Features + objetivo (potencia ~1 h adelante). Descarta filas sin potencia actual o sin objetivo."""
    features = build_feature_frame(power_series, local_timezone)
    target = power_series.shift(-FORECAST_HORIZON_INTERVALS).rename(TARGET_COLUMN)
    usable_rows = features[POWER_COLUMN].notna() & target.notna()
    return features[usable_rows], target[usable_rows]

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON_INTERVALS,
    INTERVALS_PER_DAY,
    INTERVALS_PER_HOUR,
    LAG_INTERVALS,
    build_feature_frame,
    build_training_dataset,
    resample_power_series,
)

# 2026-11-25 17:00 UTC es inicio de un intervalo de 7 min contado desde 1970 (la cuadrícula del sistema).
# Ojo: no todas las 17:00 lo son; la cuadrícula se corre 5 min cada día porque 7 min no divide el día.
GRID_START = datetime(2026, 11, 25, 17, 0, tzinfo=UTC)
# Lunes 00:00 en Costa Rica (UTC-6), para probar las features de calendario.
LOCAL_MIDNIGHT_MONDAY = datetime(2026, 11, 2, 6, 0, tzinfo=UTC)


def readings_at(minutes_from_start: list[int], powers_kw: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measured_at": pd.to_datetime(
                [GRID_START + timedelta(minutes=minute) for minute in minutes_from_start], utc=True
            ),
            "power_kw": powers_kw,
        }
    )


def power_series_of(values: list[float], start: datetime = GRID_START) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="7min", tz="UTC")
    return pd.Series(values, index=index, dtype="float64", name="power_kw")


def test_durations_are_rounded_to_whole_7_minute_intervals():
    assert INTERVALS_PER_HOUR == 9  # 63 min
    assert INTERVALS_PER_DAY == 206  # 24 h 2 min
    assert LAG_INTERVALS == (1, 2, 9, 206)
    assert FORECAST_HORIZON_INTERVALS == 9


def test_resample_averages_kilowatts_in_each_7_minute_interval():
    readings = readings_at([0, 5, 8], [1.0, 3.0, 2.0])

    power = resample_power_series(readings)

    assert power.tolist() == [2.0, 2.0]
    assert power.index[1] == pd.Timestamp(GRID_START + timedelta(minutes=7))


def test_resample_uses_the_same_grid_as_sqlite_regardless_of_the_first_reading():
    """La agregación en SQL cuenta los intervalos desde 1970; pandas debe usar la misma cuadrícula aunque
    la primera lectura no caiga en un inicio de intervalo (con 7 min, la medianoche no es inicio)."""
    readings = pd.DataFrame(
        {"measured_at": pd.to_datetime(["2026-11-25T00:00:00Z", "2026-11-25T00:06:00Z"], utc=True), "power_kw": [1.0, 3.0]}
    )

    power = resample_power_series(readings)

    sql_style_buckets = [
        pd.Timestamp((moment.value // 10**9 // 420) * 420, unit="s", tz="UTC") for moment in readings["measured_at"]
    ]
    assert power.index.tolist() == sorted(set(sql_style_buckets))
    assert power.index[0] == pd.Timestamp("2026-11-24T23:58:00Z")


def test_short_gaps_are_interpolated_and_long_gaps_are_not():
    short_gap = resample_power_series(readings_at([0, 21], [1.0, 4.0]))  # faltan 2 intervalos
    long_gap = resample_power_series(readings_at([0, 7 * 11], [1.0, 4.0]))  # faltan 10 intervalos

    assert short_gap.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert long_gap.isna().sum() == 1  # se interpolan 9 (~1 h) y queda 1 sin rellenar


def test_empty_readings_give_empty_series():
    empty_readings = pd.DataFrame({"measured_at": pd.to_datetime([], utc=True), "power_kw": []})

    assert resample_power_series(empty_readings).empty


def test_feature_frame_has_expected_columns_and_lags():
    power = power_series_of([float(value) for value in range(220)])

    features = build_feature_frame(power, "America/Costa_Rica")

    assert list(features.columns) == FEATURE_COLUMNS
    last_row = features.iloc[-1]
    assert last_row["power_kw"] == 219
    assert last_row["power_lag_1"] == 218
    assert last_row["power_lag_9"] == 210
    assert last_row["power_lag_206"] == 13
    assert last_row["power_rolling_mean_1h"] == np.mean(range(211, 220))  # últimos 9 intervalos
    assert last_row["power_rolling_max_1h"] == 219


def test_calendar_features_use_local_time():
    features = build_feature_frame(power_series_of([1.0] * 3, start=LOCAL_MIDNIGHT_MONDAY), "America/Costa_Rica")

    assert features["hour"].tolist() == pytest.approx([0.0, 7 / 60, 14 / 60])
    assert features["day_of_week"].iloc[0] == 0
    assert features["is_weekend"].iloc[0] == 0


def test_features_never_use_future_values():
    power = power_series_of([float(value) for value in range(250)])
    features_full = build_feature_frame(power, "America/Costa_Rica")
    features_truncated = build_feature_frame(power.iloc[:230], "America/Costa_Rica")

    pd.testing.assert_frame_equal(features_full.iloc[:230], features_truncated)


def test_training_target_is_power_nine_intervals_ahead():
    power = power_series_of([float(value) for value in range(30)])

    features, target = build_training_dataset(power, "America/Costa_Rica")

    assert len(features) == 30 - FORECAST_HORIZON_INTERVALS
    assert (target.to_numpy() == features["power_kw"].to_numpy() + FORECAST_HORIZON_INTERVALS).all()


@pytest.mark.parametrize("missing_position", [3, 10])
def test_training_dataset_drops_rows_without_current_power(missing_position):
    values = [float(value) for value in range(30)]
    values[missing_position] = np.nan

    features, _ = build_training_dataset(power_series_of(values), "America/Costa_Rica")

    assert features["power_kw"].notna().all()

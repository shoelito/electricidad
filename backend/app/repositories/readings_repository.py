from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from app.database import Database, to_storage_timestamp
from app.ml.features import INTERVAL_SECONDS

# Misma cuadrícula que el modelo: intervalos contados desde 1970-01-01 UTC.
AGGREGATION_INTERVAL_SECONDS = INTERVAL_SECONDS


@dataclass(frozen=True)
class StoredReading:
    measured_at: datetime
    power_kw: float


@dataclass(frozen=True)
class PowerIntervalAggregate:
    interval_start: datetime
    average_power_kw: float
    max_power_kw: float
    reading_count: int


class ReadingsRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def insert_readings(self, device_id: str, readings: Sequence[StoredReading], received_at: datetime) -> int:
        """Inserta ignorando duplicados (misma marca de tiempo). Devuelve cuántas filas nuevas se guardaron."""
        received_at_text = to_storage_timestamp(received_at)
        rows = [
            (device_id, to_storage_timestamp(reading.measured_at), reading.power_kw, received_at_text)
            for reading in readings
        ]
        with self.database.connect() as connection:
            changes_before = connection.total_changes
            connection.executemany(
                "INSERT OR IGNORE INTO readings (device_id, measured_at, power_kw, received_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            return connection.total_changes - changes_before

    def aggregate_by_interval(self, device_id: str, start: datetime, end: datetime) -> list[PowerIntervalAggregate]:
        """Agrupa en intervalos de 7 min directamente en SQL para no transferir miles de filas."""
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT (CAST(strftime('%s', measured_at) AS INTEGER) / :interval) * :interval AS bucket_epoch,
                       AVG(power_kw) AS average_power,
                       MAX(power_kw) AS max_power,
                       COUNT(*) AS reading_count
                FROM readings
                WHERE device_id = :device_id AND measured_at >= :start AND measured_at < :end
                GROUP BY bucket_epoch
                ORDER BY bucket_epoch
                """,
                {
                    "interval": AGGREGATION_INTERVAL_SECONDS,
                    "device_id": device_id,
                    "start": to_storage_timestamp(start),
                    "end": to_storage_timestamp(end),
                },
            ).fetchall()
        return [
            PowerIntervalAggregate(
                interval_start=datetime.fromtimestamp(row["bucket_epoch"], tz=UTC),
                average_power_kw=row["average_power"],
                max_power_kw=row["max_power"],
                reading_count=row["reading_count"],
            )
            for row in rows
        ]

    def fetch_interval_averages_dataframe(self, device_id: str) -> pd.DataFrame:
        """Todo el historial ya promediado cada 7 min en SQL (columnas measured_at, power_kw).

        Con una lectura por segundo, semanas de datos son millones de filas; el modelo solo necesita los
        promedios de 7 min, así que agruparlos en SQLite evita cargar todo en memoria.
        """
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT (CAST(strftime('%s', measured_at) AS INTEGER) / ?) * ? AS bucket_epoch,
                       AVG(power_kw) AS average_power
                FROM readings
                WHERE device_id = ?
                GROUP BY bucket_epoch
                ORDER BY bucket_epoch
                """,
                (AGGREGATION_INTERVAL_SECONDS, AGGREGATION_INTERVAL_SECONDS, device_id),
            ).fetchall()
        return pd.DataFrame(
            {
                "measured_at": pd.to_datetime([row["bucket_epoch"] for row in rows], unit="s", utc=True),
                "power_kw": pd.Series([row["average_power"] for row in rows], dtype="float64"),
            }
        )

    def fetch_readings_dataframe(
        self, device_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Lecturas crudas como DataFrame con columnas measured_at (UTC) y power_kw."""
        conditions = ["device_id = ?"]
        parameters: list[str] = [device_id]
        if start is not None:
            conditions.append("measured_at >= ?")
            parameters.append(to_storage_timestamp(start))
        if end is not None:
            conditions.append("measured_at < ?")
            parameters.append(to_storage_timestamp(end))
        query = (
            "SELECT measured_at, power_kw FROM readings WHERE "
            + " AND ".join(conditions)
            + " ORDER BY measured_at"
        )
        with self.database.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        readings_frame = pd.DataFrame(
            {
                "measured_at": pd.to_datetime([row["measured_at"] for row in rows], utc=True),
                "power_kw": pd.Series([row["power_kw"] for row in rows], dtype="float64"),
            }
        )
        return readings_frame

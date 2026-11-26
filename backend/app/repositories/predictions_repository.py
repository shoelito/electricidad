from dataclasses import dataclass
from datetime import datetime

from app.database import Database, from_storage_timestamp, to_storage_timestamp


@dataclass(frozen=True)
class StoredPrediction:
    target_time: datetime
    made_at: datetime
    predicted_power_kw: float
    peak_threshold_kw: float
    is_peak: bool


class PredictionsRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_prediction(self, device_id: str, prediction: StoredPrediction) -> None:
        """Una predicción por (dispositivo, hora objetivo); si se recalcula, se reemplaza."""
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions
                    (device_id, target_time, made_at, predicted_power_kw, peak_threshold_kw, is_peak)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    to_storage_timestamp(prediction.target_time),
                    to_storage_timestamp(prediction.made_at),
                    prediction.predicted_power_kw,
                    prediction.peak_threshold_kw,
                    int(prediction.is_peak),
                ),
            )

    def list_predictions(self, device_id: str, since: datetime) -> list[StoredPrediction]:
        """Predicciones cuya hora objetivo es posterior a 'since', en orden cronológico."""
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT target_time, made_at, predicted_power_kw, peak_threshold_kw, is_peak
                FROM predictions
                WHERE device_id = ? AND target_time >= ?
                ORDER BY target_time
                """,
                (device_id, to_storage_timestamp(since)),
            ).fetchall()
        return [
            StoredPrediction(
                target_time=from_storage_timestamp(row["target_time"]),
                made_at=from_storage_timestamp(row["made_at"]),
                predicted_power_kw=row["predicted_power_kw"],
                peak_threshold_kw=row["peak_threshold_kw"],
                is_peak=bool(row["is_peak"]),
            )
            for row in rows
        ]

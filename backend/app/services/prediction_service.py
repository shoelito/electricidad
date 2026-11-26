import asyncio
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta

import pandas as pd

from app.config import Settings
from app.ml.features import (
    FORECAST_HORIZON_INTERVALS,
    HISTORY_NEEDED_FOR_FEATURES,
    INTERVAL_DURATION,
    POWER_COLUMN,
    RESAMPLING_INTERVAL,
    build_feature_frame,
    resample_power_series,
)
from app.ml.training import InvalidModelArtifactError, load_model_artifact
from app.repositories.predictions_repository import PredictionsRepository, StoredPrediction
from app.repositories.readings_repository import ReadingsRepository

logger = logging.getLogger(__name__)

KILOWATT_DECIMALS = 3  # precisión de 1 W


class PredictionService:
    def __init__(
        self, settings: Settings, readings: ReadingsRepository, predictions: PredictionsRepository
    ) -> None:
        self.settings = settings
        self.readings = readings
        self.predictions = predictions
        self._artifact: dict | None = None
        self._artifact_lock = threading.Lock()

    @property
    def is_model_loaded(self) -> bool:
        return self._artifact is not None

    def load_model(self) -> None:
        """Carga el modelo desde la ruta configurada. Si falta o es inválido, se sigue sin predicciones."""
        model_path = self.settings.model_path
        if not model_path.is_file():
            logger.warning("No existe el modelo en %s; no se harán predicciones", model_path)
            self._set_artifact(None)
            return
        try:
            self._set_artifact(load_model_artifact(model_path))
            logger.info("Modelo cargado desde %s", model_path)
        except InvalidModelArtifactError:
            logger.exception("Modelo inválido en %s", model_path)
            self._set_artifact(None)

    def predict_and_store(self, device_id: str, now: datetime) -> StoredPrediction | None:
        """Predice la potencia (kW) 1 h después del último intervalo completo. None si no hay modelo o datos."""
        artifact = self._artifact
        if artifact is None:
            return None

        current_interval_start = pd.Timestamp(now).floor(RESAMPLING_INTERVAL)
        history_start = current_interval_start - HISTORY_NEEDED_FOR_FEATURES
        readings = self.readings.fetch_readings_dataframe(
            device_id, start=history_start.to_pydatetime(), end=current_interval_start.to_pydatetime()
        )
        if readings.empty:
            return None
        newest_reading_age = pd.Timestamp(now) - readings["measured_at"].iloc[-1]
        if newest_reading_age > pd.Timedelta(minutes=self.settings.max_minutes_without_data_for_prediction):
            logger.info("Sin datos recientes de %s; se omite la predicción", device_id)
            return None

        power_series = resample_power_series(readings)
        features = build_feature_frame(power_series, self.settings.local_timezone)
        latest_features = features.iloc[[-1]]
        if latest_features[POWER_COLUMN].isna().iloc[0]:
            return None

        predicted_power = float(artifact["model"].predict(latest_features[artifact["feature_columns"]])[0])
        peak_threshold = float(artifact["peak_threshold"])
        latest_interval_start = latest_features.index[0]
        prediction = StoredPrediction(
            target_time=(latest_interval_start + FORECAST_HORIZON_INTERVALS * INTERVAL_DURATION).to_pydatetime(),
            made_at=now,
            predicted_power_kw=round(max(predicted_power, 0.0), KILOWATT_DECIMALS),
            peak_threshold_kw=round(peak_threshold, KILOWATT_DECIMALS),
            is_peak=predicted_power > peak_threshold,
        )
        self.predictions.save_prediction(device_id, prediction)
        return prediction

    async def run_periodically(
        self, list_device_ids: Callable[[], list[str]], clock: Callable[[], datetime]
    ) -> None:
        """Bucle en segundo plano: predice al arrancar y luego tras cerrar cada intervalo de 7 min."""
        delay_after_interval = timedelta(seconds=self.settings.prediction_delay_after_interval_seconds)
        while True:
            try:
                for device_id in await asyncio.to_thread(list_device_ids):
                    await asyncio.to_thread(self.predict_and_store, device_id, clock())
            except Exception:
                logger.exception("Error al generar predicciones periódicas")
            await asyncio.sleep(seconds_until_next_prediction(clock(), delay_after_interval))

    def _set_artifact(self, artifact: dict | None) -> None:
        with self._artifact_lock:
            self._artifact = artifact


def seconds_until_next_prediction(now: datetime, delay_after_interval: timedelta) -> float:
    """Espera hasta 'delay' después del próximo cierre de intervalo, para que lleguen los lotes del sensor."""
    next_run = pd.Timestamp(now).floor(RESAMPLING_INTERVAL) + delay_after_interval
    if next_run <= pd.Timestamp(now):
        next_run += INTERVAL_DURATION
    return (next_run - pd.Timestamp(now)).total_seconds()

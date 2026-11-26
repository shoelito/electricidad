"""Entrenamiento del modelo lanzado desde el dashboard, en un hilo aparte para no bloquear el servidor."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from app.config import Settings
from app.ml.features import resample_power_series
from app.ml.training import InsufficientDataError, TrainingResult, save_model_artifact, train_peak_model
from app.repositories.readings_repository import ReadingsRepository
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR_MESSAGE = "Error inesperado durante el entrenamiento; revise el log del servidor"


class TrainingState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingStatus:
    state: TrainingState = TrainingState.IDLE
    device_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class TrainingAlreadyRunningError(Exception):
    pass


def run_in_daemon_thread(job: Callable[[], None]) -> None:
    threading.Thread(target=job, name="entrenamiento-modelo", daemon=True).start()


def train_and_save_from_database(
    readings: ReadingsRepository, settings: Settings, device_id: str, now: datetime | None = None
) -> TrainingResult:
    """Entrena con todo el historial del dispositivo y guarda el modelo. Lo usan el dashboard y el script."""
    interval_averages = readings.fetch_interval_averages_dataframe(device_id)
    power_series = resample_power_series(interval_averages)
    result = train_peak_model(power_series, settings.local_timezone, now=now)
    save_model_artifact(result.artifact, settings.model_path)
    return result


class TrainingService:
    """Solo permite un entrenamiento a la vez. Al terminar, carga el modelo nuevo y genera una predicción."""

    def __init__(
        self,
        settings: Settings,
        readings: ReadingsRepository,
        prediction_service: PredictionService,
        clock: Callable[[], datetime],
        run_in_background: Callable[[Callable[[], None]], None] = run_in_daemon_thread,
    ) -> None:
        self.settings = settings
        self.readings = readings
        self.prediction_service = prediction_service
        self.clock = clock
        self.run_in_background = run_in_background
        self._status = TrainingStatus()
        self._status_lock = threading.Lock()

    @property
    def status(self) -> TrainingStatus:
        with self._status_lock:
            return self._status

    def start_training(self, device_id: str) -> TrainingStatus:
        with self._status_lock:
            if self._status.state == TrainingState.RUNNING:
                raise TrainingAlreadyRunningError(self._status.device_id)
            self._status = TrainingStatus(
                state=TrainingState.RUNNING, device_id=device_id, started_at=self.clock()
            )
            started_status = self._status
        self.run_in_background(lambda: self._train(device_id))
        return started_status

    def _train(self, device_id: str) -> None:
        try:
            result = train_and_save_from_database(self.readings, self.settings, device_id, now=self.clock())
            self.prediction_service.load_model()
            self.prediction_service.predict_and_store(device_id, self.clock())
        except InsufficientDataError as error:
            self._finish(TrainingState.FAILED, message=str(error))
        except Exception:
            logger.exception("Falló el entrenamiento del modelo para %s", device_id)
            self._finish(TrainingState.FAILED, message=UNEXPECTED_ERROR_MESSAGE)
        else:
            logger.info("Modelo entrenado para %s: %s", device_id, result.metrics)
            self._finish(TrainingState.SUCCEEDED, message="Modelo entrenado y cargado", metrics=result.metrics)

    def _finish(self, state: TrainingState, message: str, metrics: dict[str, float] | None = None) -> None:
        with self._status_lock:
            self._status = replace(
                self._status, state=state, finished_at=self.clock(), message=message, metrics=metrics or {}
            )

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import Request

from app.config import Settings
from app.database import Database
from app.repositories.devices_repository import DevicesRepository
from app.repositories.predictions_repository import PredictionsRepository
from app.repositories.readings_repository import ReadingsRepository
from app.services.prediction_service import PredictionService
from app.services.training_service import TrainingService


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class AppContext:
    """Objetos compartidos por toda la app. Se guarda en app.state para poder sustituirlos en tests."""

    settings: Settings
    database: Database
    devices: DevicesRepository
    readings: ReadingsRepository
    predictions: PredictionsRepository
    prediction_service: PredictionService
    training_service: TrainingService
    clock: Callable[[], datetime] = field(default=utc_now)


def build_app_context(settings: Settings, clock: Callable[[], datetime] = utc_now) -> AppContext:
    database = Database(settings.database_path)
    readings = ReadingsRepository(database)
    predictions = PredictionsRepository(database)
    prediction_service = PredictionService(settings, readings, predictions)
    return AppContext(
        settings=settings,
        database=database,
        devices=DevicesRepository(database),
        readings=readings,
        predictions=predictions,
        prediction_service=prediction_service,
        training_service=TrainingService(settings, readings, prediction_service, clock),
        clock=clock,
    )


def get_app_context(request: Request) -> AppContext:
    return request.app.state.context

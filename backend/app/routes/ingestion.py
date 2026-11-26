from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import AppContext, get_app_context
from app.repositories.readings_repository import StoredReading
from app.schemas import IngestionResult, ReadingBatchIn

router = APIRouter(prefix="/api", tags=["ingesta"])


@router.post(
    "/lecturas",
    status_code=status.HTTP_201_CREATED,
    response_model=IngestionResult,
    summary="Enviar un lote de lecturas del sensor",
    responses={422: {"description": "Datos fuera de rango o mal formados"}},
)
def ingest_readings(
    batch: ReadingBatchIn,
    context: Annotated[AppContext, Depends(get_app_context)],
) -> IngestionResult:
    """Recibe un lote de lecturas. Un `device_id` nuevo se registra solo en su primer envío.
    4xx: no reintentar. 5xx o error de red: reintentar."""
    now = context.clock()
    validation_errors = _find_out_of_range_readings(batch, context, now)
    if validation_errors:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, validation_errors)

    readings_to_store = [
        StoredReading(measured_at=reading.timestamp, power_kw=reading.power_kw) for reading in batch.lecturas
    ]
    context.devices.ensure_device_exists(batch.device_id, now)
    inserted_count = context.readings.insert_readings(batch.device_id, readings_to_store, received_at=now)
    context.devices.update_last_seen(batch.device_id, now)
    return IngestionResult(aceptadas=inserted_count, duplicadas=len(readings_to_store) - inserted_count)


def _find_out_of_range_readings(batch: ReadingBatchIn, context: AppContext, now: datetime) -> list[str]:
    settings = context.settings
    latest_allowed = now + timedelta(seconds=settings.max_future_skew_seconds)
    earliest_allowed = now - timedelta(days=settings.max_reading_age_days)
    errors: list[str] = []
    for position, reading in enumerate(batch.lecturas):
        if reading.power_kw > settings.max_power_kw:
            errors.append(f"lecturas[{position}].power_kw supera {settings.max_power_kw} kW")
        if reading.timestamp > latest_allowed:
            errors.append(f"lecturas[{position}].timestamp está en el futuro")
        if reading.timestamp < earliest_allowed:
            errors.append(f"lecturas[{position}].timestamp es más antiguo que {settings.max_reading_age_days} días")
    return errors

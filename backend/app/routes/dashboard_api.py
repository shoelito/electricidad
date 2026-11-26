from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import AwareDatetime

from app.dependencies import AppContext, get_app_context
from app.schemas import (
    DEVICE_ID_PATTERN,
    DeviceStatus,
    PowerInterval,
    PredictionOut,
    PredictionSummary,
    TrainingStatusOut,
)
from app.services.training_service import TrainingAlreadyRunningError

router = APIRouter(prefix="/api", tags=["gestión"])

MAX_QUERY_RANGE = timedelta(days=31)
KILOWATT_DECIMALS = 3  # precisión de 1 W
PREDICTION_HISTORY_WINDOW = timedelta(hours=24)

DeviceIdQuery = Annotated[str, Query(pattern=DEVICE_ID_PATTERN)]


@router.get("/dispositivos", response_model=list[DeviceStatus], summary="Listar dispositivos y su estado")
def list_devices(context: Annotated[AppContext, Depends(get_app_context)]) -> list[DeviceStatus]:
    now = context.clock()
    return [
        DeviceStatus(
            device_id=device_id,
            last_seen_at=last_seen_at,
            seconds_since_last_reading=int((now - last_seen_at).total_seconds()) if last_seen_at else None,
        )
        for device_id, last_seen_at in context.devices.list_last_seen()
    ]


@router.get("/consumo", response_model=list[PowerInterval], summary="Consumo agregado cada 7 min")
def get_power_by_interval(
    context: Annotated[AppContext, Depends(get_app_context)],
    device_id: DeviceIdQuery,
    desde: Annotated[AwareDatetime | None, Query()] = None,
    hasta: Annotated[AwareDatetime | None, Query()] = None,
) -> list[PowerInterval]:
    """Potencia promedio y máxima (kW) en intervalos de 7 min. Por defecto, las últimas 24 h."""
    range_end: datetime = hasta or context.clock()
    range_start: datetime = desde or range_end - timedelta(hours=24)
    if range_start >= range_end:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "'desde' debe ser anterior a 'hasta'")
    if range_end - range_start > MAX_QUERY_RANGE:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "El rango máximo es de 31 días")

    return [
        PowerInterval(
            interval_start=interval.interval_start,
            average_power_kw=round(interval.average_power_kw, KILOWATT_DECIMALS),
            max_power_kw=round(interval.max_power_kw, KILOWATT_DECIMALS),
            reading_count=interval.reading_count,
        )
        for interval in context.readings.aggregate_by_interval(device_id, range_start, range_end)
    ]


@router.get("/predicciones", response_model=PredictionSummary, summary="Predicciones de las últimas 24 h")
def get_predictions(
    context: Annotated[AppContext, Depends(get_app_context)],
    device_id: DeviceIdQuery,
) -> PredictionSummary:
    now = context.clock()
    history = context.predictions.list_predictions(device_id, since=now - PREDICTION_HISTORY_WINDOW)
    prediction_models = [PredictionOut.model_validate(prediction, from_attributes=True) for prediction in history]
    return PredictionSummary(
        modelo_cargado=context.prediction_service.is_model_loaded,
        latest=prediction_models[-1] if prediction_models else None,
        history=prediction_models,
    )


@router.post("/modelo/recargar", response_model=PredictionSummary, summary="Recargar el modelo desde disco")
def reload_model(
    context: Annotated[AppContext, Depends(get_app_context)],
    device_id: DeviceIdQuery,
) -> PredictionSummary:
    """Recarga el modelo desde disco (tras reentrenar) y genera una predicción inmediata."""
    context.prediction_service.load_model()
    if context.prediction_service.is_model_loaded:
        context.prediction_service.predict_and_store(device_id, context.clock())
    return get_predictions(context, device_id)


@router.post(
    "/modelo/entrenar",
    response_model=TrainingStatusOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Entrenar el modelo con el historial del dispositivo",
    responses={409: {"description": "Ya hay un entrenamiento en curso"}},
)
def start_training(
    context: Annotated[AppContext, Depends(get_app_context)],
    device_id: DeviceIdQuery,
) -> TrainingStatusOut:
    """Lanza el entrenamiento en segundo plano y responde de inmediato. Consultar el avance en
    `GET /api/modelo/entrenamiento`. Al terminar, el modelo nuevo se carga solo."""
    try:
        training_status = context.training_service.start_training(device_id)
    except TrainingAlreadyRunningError:
        raise HTTPException(status.HTTP_409_CONFLICT, "Ya hay un entrenamiento en curso") from None
    return TrainingStatusOut.model_validate(training_status, from_attributes=True)


@router.get("/modelo/entrenamiento", response_model=TrainingStatusOut, summary="Estado del último entrenamiento")
def get_training_status(context: Annotated[AppContext, Depends(get_app_context)]) -> TrainingStatusOut:
    return TrainingStatusOut.model_validate(context.training_service.status, from_attributes=True)

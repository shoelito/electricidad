from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import AppContext, get_app_context
from app.schemas import HealthStatus

router = APIRouter(prefix="/api", tags=["salud"])


@router.get("/salud", response_model=HealthStatus, summary="Estado del servicio")
def get_health(context: Annotated[AppContext, Depends(get_app_context)]) -> HealthStatus:
    """Indica si el servidor responde y si hay un modelo de predicción cargado."""
    return HealthStatus(estado="ok", modelo_cargado=context.prediction_service.is_model_loaded)

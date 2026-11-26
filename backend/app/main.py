import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Response

from app.config import Settings, get_settings
from app.dependencies import build_app_context, utc_now
from app.routes import dashboard_api, health, ingestion

API_DESCRIPTION = (
    "API del monitor eléctrico. El sensor envía lecturas a `POST /api/lecturas`; "
    "el resto de endpoints alimentan el dashboard (frontend en Streamlit)."
)
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def create_app(settings: Settings | None = None, clock: Callable[[], datetime] = utc_now) -> FastAPI:
    settings = settings or get_settings()
    context = build_app_context(settings, clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        context.database.initialize_schema()
        context.prediction_service.load_model()
        prediction_task = None
        if settings.enable_prediction_scheduler:
            prediction_task = asyncio.create_task(
                context.prediction_service.run_periodically(context.devices.list_device_ids, context.clock)
            )
        yield
        if prediction_task is not None:
            prediction_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await prediction_task

    application = FastAPI(
        title="Monitor eléctrico - predicción de picos",
        lifespan=lifespan,
        description=API_DESCRIPTION,
        docs_url=None,  # la documentación interactiva está en el frontend (Explorador de la API)
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    application.state.context = context

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    application.include_router(ingestion.router)
    application.include_router(health.router)
    application.include_router(dashboard_api.router)
    return application


logging.basicConfig(level=logging.INFO)

"""Contrato de la API. Lo que está aquí es lo que se acuerda con el equipo de hardware."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

DEVICE_ID_PATTERN = r"^[a-z0-9-]{3,40}$"
MAX_READINGS_PER_BATCH = 500


class ReadingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: AwareDatetime = Field(description="Momento de la medición, con zona horaria (UTC recomendado)")
    power_kw: float = Field(
        ge=0, allow_inf_nan=False, description="Potencia en kilovatios (kW) medida en ese segundo"
    )


class ReadingBatchIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "device_id": "medidor-01",
                    "lecturas": [
                        {"timestamp": "2026-11-25T18:00:00Z", "power_kw": 0.412},
                        {"timestamp": "2026-11-25T18:00:01Z", "power_kw": 0.408},
                    ],
                }
            ]
        },
    )

    device_id: str = Field(pattern=DEVICE_ID_PATTERN)
    lecturas: list[ReadingIn] = Field(min_length=1, max_length=MAX_READINGS_PER_BATCH)


class IngestionResult(BaseModel):
    aceptadas: int
    duplicadas: int


class HealthStatus(BaseModel):
    estado: str
    modelo_cargado: bool


class DeviceStatus(BaseModel):
    device_id: str
    last_seen_at: datetime | None
    seconds_since_last_reading: int | None


class PowerInterval(BaseModel):
    interval_start: datetime
    average_power_kw: float
    max_power_kw: float
    reading_count: int


class PredictionOut(BaseModel):
    target_time: datetime
    made_at: datetime
    predicted_power_kw: float
    peak_threshold_kw: float
    is_peak: bool


class TrainingStatusOut(BaseModel):
    state: Literal["idle", "running", "succeeded", "failed"]
    device_id: str | None
    started_at: datetime | None
    finished_at: datetime | None
    message: str | None
    metrics: dict[str, float]


class PredictionSummary(BaseModel):
    modelo_cargado: bool
    latest: PredictionOut | None
    history: list[PredictionOut]

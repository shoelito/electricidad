from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración leída de variables de entorno con prefijo ELECTRICIDAD_ o del archivo .env."""

    model_config = SettingsConfigDict(env_prefix="ELECTRICIDAD_", env_file=".env", extra="ignore")

    database_path: Path = Path("data/electricidad.db")
    model_path: Path = Path("models/modelo_picos.joblib")

    max_power_kw: float = Field(default=25.0, gt=0)
    max_future_skew_seconds: int = Field(default=120, ge=0)
    max_reading_age_days: int = Field(default=7, ge=1)

    local_timezone: str = "America/Costa_Rica"
    enable_prediction_scheduler: bool = True
    # Debe ser menor que el intervalo del modelo (7 min = 420 s) y mayor que el envío de lotes del ESP32 (5 min).
    prediction_delay_after_interval_seconds: int = Field(default=360, ge=0, lt=420)
    max_minutes_without_data_for_prediction: int = Field(default=30, ge=15)


@lru_cache
def get_settings() -> Settings:
    return Settings()

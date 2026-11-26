from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ml.features import resample_power_series
from app.ml.synthetic_data import generate_synthetic_readings
from app.ml.training import train_peak_model

TEST_DEVICE_ID = "medidor-01"
FIXED_NOW = datetime(2026, 11, 25, 18, 7, 0, tzinfo=UTC)


class FakeClock:
    """Reloj controlable para probar expiraciones y ventanas de tiempo."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, duration: timedelta) -> None:
        self.now += duration


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(FIXED_NOW)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        database_path=tmp_path / "test.db",
        model_path=tmp_path / "models" / "modelo.joblib",
        enable_prediction_scheduler=False,
    )


@pytest.fixture
def client(settings, clock):
    with TestClient(create_app(settings, clock)) as test_client:
        yield test_client


@pytest.fixture
def app_context(client):
    return client.app.state.context


@pytest.fixture
def registered_device(app_context) -> str:
    app_context.devices.ensure_device_exists(TEST_DEVICE_ID, FIXED_NOW)
    return TEST_DEVICE_ID


@pytest.fixture(scope="session")
def synthetic_readings():
    return generate_synthetic_readings(FIXED_NOW - timedelta(days=14), days=14)


@pytest.fixture(scope="session")
def trained_result(synthetic_readings):
    power_series = resample_power_series(synthetic_readings)
    return train_peak_model(power_series, "America/Costa_Rica", now=FIXED_NOW)


def build_reading_batch(timestamps: list[datetime], power_kw: float = 0.3, device_id: str = TEST_DEVICE_ID) -> dict:
    return {
        "device_id": device_id,
        "lecturas": [{"timestamp": moment.isoformat(), "power_kw": power_kw} for moment in timestamps],
    }

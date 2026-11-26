from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from monitor_frontend.api_client import ApiError, RawResponse

FRONTEND_DIRECTORY = Path(__file__).resolve().parent.parent
APP_TEST_TIMEOUT_SECONDS = 30


def iso_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FakeApiClient:
    """Backend falso para probar las páginas sin levantar FastAPI."""

    base_url: str = "http://backend-de-prueba"
    devices: list[dict] = field(default_factory=list)
    intervals: list[dict] = field(default_factory=list)
    prediction_summary: dict = field(default_factory=lambda: {"modelo_cargado": False, "latest": None, "history": []})
    training: dict = field(default_factory=lambda: {"state": "idle", "metrics": {}})
    model_is_loaded: bool = False
    specification: dict = field(default_factory=dict)
    error_to_raise: ApiError | None = None
    start_training_error: ApiError | None = None
    started_trainings: list[str] = field(default_factory=list)
    reloaded_models: list[str] = field(default_factory=list)
    raw_requests: list[tuple] = field(default_factory=list)

    def _check(self) -> None:
        if self.error_to_raise:
            raise self.error_to_raise

    def health(self) -> dict:
        self._check()
        return {"estado": "ok", "modelo_cargado": self.model_is_loaded}

    def list_devices(self) -> list[dict]:
        self._check()
        return self.devices

    def power_by_interval(self, device_id: str) -> list[dict]:
        self._check()
        return self.intervals

    def predictions(self, device_id: str) -> dict:
        self._check()
        return self.prediction_summary

    def start_training(self, device_id: str) -> dict:
        if self.start_training_error:
            raise self.start_training_error
        self.started_trainings.append(device_id)
        self.training = {"state": "running", "device_id": device_id, "started_at": iso_utc(datetime.now(UTC))}
        return self.training

    def training_status(self) -> dict:
        self._check()
        return self.training

    def reload_model(self, device_id: str) -> dict:
        self.reloaded_models.append(device_id)
        return self.prediction_summary

    def openapi_specification(self) -> dict:
        self._check()
        return self.specification

    def send_raw_request(self, method, path, query_parameters, json_body) -> RawResponse:
        self.raw_requests.append((method, path, query_parameters, json_body))
        return RawResponse(201, "Created", 12, '{"aceptadas": 1, "duplicadas": 0}')


@pytest.fixture
def fake_api(monkeypatch) -> FakeApiClient:
    fake = FakeApiClient()
    monkeypatch.setattr("monitor_frontend.api_client.get_api_client", lambda: fake)
    return fake


@pytest.fixture
def run_view():
    def run(view_file: str) -> AppTest:
        app_test = AppTest.from_file(str(FRONTEND_DIRECTORY / "views" / view_file), default_timeout=APP_TEST_TIMEOUT_SECONDS)
        return app_test.run()

    return run


def sample_intervals(now: datetime, count: int = 3) -> list[dict]:
    return [
        {
            "interval_start": iso_utc(now - timedelta(minutes=7 * (count - position))),
            "average_power_kw": 0.3 + position * 0.1,
            "max_power_kw": 0.5,
            "reading_count": 420,
        }
        for position in range(count)
    ]


def sample_prediction(target_time: datetime, predicted_kw: float, is_peak: bool) -> dict:
    return {
        "target_time": iso_utc(target_time),
        "made_at": iso_utc(target_time - timedelta(hours=1)),
        "predicted_power_kw": predicted_kw,
        "peak_threshold_kw": 1.2,
        "is_peak": is_peak,
    }

from datetime import timedelta

import pandas as pd
import pytest

from app.ml.synthetic_data import generate_synthetic_readings
from app.repositories.readings_repository import StoredReading
from app.services.training_service import (
    UNEXPECTED_ERROR_MESSAGE,
    TrainingAlreadyRunningError,
    TrainingState,
)
from tests.conftest import FIXED_NOW, TEST_DEVICE_ID


def run_immediately(job) -> None:
    job()


def never_run(job) -> None:
    """Simula un entrenamiento que sigue en curso: el trabajo nunca termina."""


@pytest.fixture
def synchronous_training(app_context):
    app_context.training_service.run_in_background = run_immediately
    return app_context.training_service


def store_synthetic_history(app_context, days: float) -> None:
    readings = generate_synthetic_readings(FIXED_NOW - timedelta(days=days), days=days)
    app_context.devices.ensure_device_exists(TEST_DEVICE_ID, FIXED_NOW)
    app_context.readings.insert_readings(
        TEST_DEVICE_ID,
        [StoredReading(row.measured_at.to_pydatetime(), row.power_kw) for row in readings.itertuples()],
        received_at=FIXED_NOW,
    )


def test_status_is_idle_before_any_training(app_context):
    assert app_context.training_service.status.state == TrainingState.IDLE


def test_successful_training_saves_loads_and_predicts(app_context, synchronous_training, settings):
    store_synthetic_history(app_context, days=10)

    synchronous_training.start_training(TEST_DEVICE_ID)

    status = synchronous_training.status
    assert status.state == TrainingState.SUCCEEDED
    assert status.started_at == FIXED_NOW and status.finished_at == FIXED_NOW
    assert status.metrics["mae_modelo_kw"] < status.metrics["mae_persistencia_kw"]
    assert settings.model_path.is_file()
    assert app_context.prediction_service.is_model_loaded
    assert app_context.predictions.list_predictions(TEST_DEVICE_ID, since=FIXED_NOW)


def test_training_with_too_little_data_fails_with_clear_message(app_context, synchronous_training, settings):
    store_synthetic_history(app_context, days=0.4)  # ~9,6 h: 74 intervalos útiles de 7 min

    synchronous_training.start_training(TEST_DEVICE_ID)

    status = synchronous_training.status
    assert status.state == TrainingState.FAILED
    assert "Se necesitan al menos" in status.message
    assert not settings.model_path.exists()
    assert not app_context.prediction_service.is_model_loaded


def test_unexpected_error_is_reported_without_internal_details(app_context, synchronous_training, monkeypatch):
    def explode(device_id):
        raise RuntimeError("detalle interno")

    monkeypatch.setattr(app_context.readings, "fetch_interval_averages_dataframe", explode)

    synchronous_training.start_training(TEST_DEVICE_ID)

    assert synchronous_training.status.state == TrainingState.FAILED
    assert synchronous_training.status.message == UNEXPECTED_ERROR_MESSAGE


def test_only_one_training_runs_at_a_time(app_context):
    app_context.training_service.run_in_background = never_run
    app_context.training_service.start_training(TEST_DEVICE_ID)

    with pytest.raises(TrainingAlreadyRunningError):
        app_context.training_service.start_training(TEST_DEVICE_ID)


def test_a_new_training_can_start_after_the_previous_one_finished(app_context, synchronous_training):
    synchronous_training.start_training(TEST_DEVICE_ID)  # falla por falta de datos, pero termina

    synchronous_training.start_training(TEST_DEVICE_ID)

    assert synchronous_training.status.state == TrainingState.FAILED


def test_interval_averages_match_raw_readings_resampled(app_context):
    store_synthetic_history(app_context, days=1)

    averaged_in_sql = app_context.readings.fetch_interval_averages_dataframe(TEST_DEVICE_ID)
    raw = app_context.readings.fetch_readings_dataframe(TEST_DEVICE_ID)
    averaged_in_pandas = raw.set_index("measured_at")["power_kw"].resample("7min", origin="epoch").mean().dropna()

    assert len(averaged_in_sql) == len(averaged_in_pandas) == 207  # 24 h desde las 18:07 tocan 207 intervalos de 7 min
    pd.testing.assert_series_equal(
        averaged_in_sql.set_index("measured_at")["power_kw"], averaged_in_pandas, check_names=False, check_freq=False
    )


def test_train_endpoint_returns_202_and_reports_progress(client, app_context, synchronous_training):
    store_synthetic_history(app_context, days=10)

    response = client.post("/api/modelo/entrenar", params={"device_id": TEST_DEVICE_ID})

    assert response.status_code == 202
    assert response.json()["state"] == "running"
    status = client.get("/api/modelo/entrenamiento").json()
    assert status["state"] == "succeeded"
    assert status["device_id"] == TEST_DEVICE_ID
    assert {"mae_modelo_kw", "recall_picos_modelo"} <= status["metrics"].keys()
    assert client.get("/api/salud").json()["modelo_cargado"] is True


def test_train_endpoint_rejects_concurrent_training(client, app_context):
    app_context.training_service.run_in_background = never_run
    client.post("/api/modelo/entrenar", params={"device_id": TEST_DEVICE_ID})

    response = client.post("/api/modelo/entrenar", params={"device_id": TEST_DEVICE_ID})

    assert response.status_code == 409


def test_train_endpoint_validates_device_id(client):
    assert client.post("/api/modelo/entrenar", params={"device_id": "x' OR 1=1"}).status_code == 422


def test_training_status_starts_idle(client):
    assert client.get("/api/modelo/entrenamiento").json() == {
        "state": "idle",
        "device_id": None,
        "started_at": None,
        "finished_at": None,
        "message": None,
        "metrics": {},
    }

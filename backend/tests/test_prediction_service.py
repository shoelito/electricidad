from datetime import UTC, datetime, timedelta

import pytest

from app.ml.synthetic_data import generate_synthetic_readings
from app.ml.training import save_model_artifact
from app.repositories.readings_repository import StoredReading
from app.services.prediction_service import seconds_until_next_prediction
from tests.conftest import FIXED_NOW, TEST_DEVICE_ID

# FIXED_NOW = 18:07 UTC cae en el intervalo de 7 min 18:03–18:10; el último completo es 17:56–18:03.
LAST_COMPLETE_INTERVAL_START = datetime(2026, 11, 25, 17, 56, tzinfo=UTC)


def store_recent_synthetic_readings(app_context, days: int = 2, end: datetime = FIXED_NOW) -> None:
    readings = generate_synthetic_readings(end - timedelta(days=days), days=days)
    app_context.readings.insert_readings(
        TEST_DEVICE_ID,
        [StoredReading(row.measured_at.to_pydatetime(), row.power_kw) for row in readings.itertuples()],
        received_at=end,
    )


@pytest.fixture
def loaded_model(app_context, trained_result, settings):
    save_model_artifact(trained_result.artifact, settings.model_path)
    app_context.prediction_service.load_model()
    assert app_context.prediction_service.is_model_loaded


def test_without_model_no_prediction_is_made(app_context, registered_device):
    store_recent_synthetic_readings(app_context)

    assert app_context.prediction_service.predict_and_store(TEST_DEVICE_ID, FIXED_NOW) is None


def test_prediction_targets_nine_intervals_after_last_complete_interval(app_context, registered_device, loaded_model):
    store_recent_synthetic_readings(app_context)

    prediction = app_context.prediction_service.predict_and_store(TEST_DEVICE_ID, FIXED_NOW)

    assert prediction is not None
    assert prediction.target_time == LAST_COMPLETE_INTERVAL_START + timedelta(minutes=63)  # 9 × 7 min
    assert prediction.predicted_power_kw >= 0
    assert prediction.is_peak == (prediction.predicted_power_kw > prediction.peak_threshold_kw)
    stored = app_context.predictions.list_predictions(TEST_DEVICE_ID, since=FIXED_NOW - timedelta(days=1))
    assert stored == [prediction]


def test_recomputing_same_target_replaces_previous_prediction(app_context, registered_device, loaded_model):
    store_recent_synthetic_readings(app_context)

    app_context.prediction_service.predict_and_store(TEST_DEVICE_ID, FIXED_NOW)
    app_context.prediction_service.predict_and_store(TEST_DEVICE_ID, FIXED_NOW + timedelta(minutes=1))

    assert len(app_context.predictions.list_predictions(TEST_DEVICE_ID, since=FIXED_NOW)) == 1


def test_no_prediction_when_sensor_stopped_sending(app_context, registered_device, loaded_model):
    store_recent_synthetic_readings(app_context, end=FIXED_NOW - timedelta(hours=2))

    assert app_context.prediction_service.predict_and_store(TEST_DEVICE_ID, FIXED_NOW) is None


def test_missing_model_file_leaves_service_without_model(app_context):
    app_context.prediction_service.load_model()

    assert not app_context.prediction_service.is_model_loaded


def test_reload_endpoint_loads_model_and_returns_prediction(
    client, app_context, registered_device, trained_result, settings
):
    store_recent_synthetic_readings(app_context)
    save_model_artifact(trained_result.artifact, settings.model_path)

    response = client.post("/api/modelo/recargar", params={"device_id": TEST_DEVICE_ID})

    assert response.status_code == 200
    summary = response.json()
    assert summary["modelo_cargado"] is True
    assert summary["latest"] is not None
    assert summary["history"] == [summary["latest"]]
    assert client.get("/api/salud").json()["modelo_cargado"] is True


@pytest.mark.parametrize(
    ("now", "expected_seconds"),
    [
        (datetime(2026, 11, 25, 18, 3, tzinfo=UTC), 360),  # inicio de intervalo 18:03 -> 18:09
        (datetime(2026, 11, 25, 18, 7, tzinfo=UTC), 120),  # 18:07 -> 18:09
        (datetime(2026, 11, 25, 18, 9, tzinfo=UTC), 420),  # justo a la hora -> siguiente intervalo (18:16)
        (datetime(2026, 11, 25, 18, 10, tzinfo=UTC), 360),  # inicio de intervalo 18:10 -> 18:16
    ],
)
def test_next_prediction_runs_after_each_interval_closes(now, expected_seconds):
    assert seconds_until_next_prediction(now, timedelta(minutes=6)) == expected_seconds

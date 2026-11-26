from datetime import timedelta

import pytest

from tests.conftest import FIXED_NOW, TEST_DEVICE_ID, build_reading_batch

RECENT_TIMESTAMPS = [FIXED_NOW - timedelta(seconds=10 * offset) for offset in range(3, 0, -1)]


def post_readings(client, batch: dict):
    return client.post("/api/lecturas", json=batch)


def test_valid_batch_is_stored(client, app_context):
    response = post_readings(client, build_reading_batch(RECENT_TIMESTAMPS))

    assert response.status_code == 201
    assert response.json() == {"aceptadas": 3, "duplicadas": 0}
    assert len(app_context.readings.fetch_readings_dataframe(TEST_DEVICE_ID)) == 3


def test_new_device_is_registered_on_first_batch(client, app_context):
    assert app_context.devices.list_device_ids() == []

    post_readings(client, build_reading_batch(RECENT_TIMESTAMPS, device_id="medidor-nuevo"))

    assert app_context.devices.list_device_ids() == ["medidor-nuevo"]


def test_existing_device_is_not_duplicated(client, registered_device, app_context):
    post_readings(client, build_reading_batch(RECENT_TIMESTAMPS))

    assert app_context.devices.list_device_ids() == [registered_device]


def test_five_minutes_of_one_second_readings_fit_in_one_batch(client, app_context):
    one_per_second = [FIXED_NOW - timedelta(seconds=offset) for offset in range(300, 0, -1)]

    response = post_readings(client, build_reading_batch(one_per_second, power_kw=0.412))

    assert response.json() == {"aceptadas": 300, "duplicadas": 0}
    stored = app_context.readings.fetch_readings_dataframe(TEST_DEVICE_ID)
    assert stored["power_kw"].eq(0.412).all()


def test_old_current_field_is_rejected(client):
    batch = {"device_id": TEST_DEVICE_ID, "lecturas": [{"timestamp": FIXED_NOW.isoformat(), "current_rms": 3.4}]}

    assert post_readings(client, batch).status_code == 422


def test_resent_batch_is_counted_as_duplicates(client):
    batch = build_reading_batch(RECENT_TIMESTAMPS)
    post_readings(client, batch)

    response = post_readings(client, batch)

    assert response.status_code == 201
    assert response.json() == {"aceptadas": 0, "duplicadas": 3}


def test_ingestion_updates_device_last_seen(client, app_context):
    post_readings(client, build_reading_batch(RECENT_TIMESTAMPS))

    assert app_context.devices.list_last_seen() == [(TEST_DEVICE_ID, FIXED_NOW)]


@pytest.mark.parametrize("power_kw", [-0.1, 25.5, "abc"])
def test_out_of_range_power_is_rejected(client, power_kw):
    response = post_readings(client, build_reading_batch(RECENT_TIMESTAMPS, power_kw=power_kw))

    assert response.status_code == 422


def test_future_timestamp_is_rejected(client):
    response = post_readings(client, build_reading_batch([FIXED_NOW + timedelta(minutes=10)]))

    assert response.status_code == 422


def test_too_old_timestamp_is_rejected(client):
    response = post_readings(client, build_reading_batch([FIXED_NOW - timedelta(days=8)]))

    assert response.status_code == 422


def test_timestamp_without_timezone_is_rejected(client):
    batch = {"device_id": TEST_DEVICE_ID, "lecturas": [{"timestamp": "2026-11-25T18:00:00", "power_kw": 1.0}]}

    assert post_readings(client, batch).status_code == 422


@pytest.mark.parametrize(
    "batch",
    [
        {"device_id": TEST_DEVICE_ID, "lecturas": []},
        {"device_id": "ID CON ESPACIOS", "lecturas": [{"timestamp": "2026-11-25T18:00:00Z", "power_kw": 1}]},
        {"device_id": TEST_DEVICE_ID, "lecturas": [{"timestamp": "2026-11-25T18:00:00Z", "power_kw": 1, "x": 1}]},
    ],
)
def test_malformed_batches_are_rejected(client, batch):
    assert post_readings(client, batch).status_code == 422


def test_rejected_batch_stores_nothing_and_registers_no_device(client, app_context):
    timestamps = [FIXED_NOW - timedelta(seconds=10), FIXED_NOW + timedelta(hours=1)]

    post_readings(client, build_reading_batch(timestamps))

    assert app_context.readings.fetch_readings_dataframe(TEST_DEVICE_ID).empty
    assert app_context.devices.list_device_ids() == []

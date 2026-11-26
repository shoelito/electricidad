from datetime import UTC, datetime, timedelta

from app.repositories.readings_repository import StoredReading
from tests.conftest import FIXED_NOW, TEST_DEVICE_ID

FIRST_INTERVAL_START = datetime(2026, 11, 25, 17, 0, tzinfo=UTC)  # coincide con un inicio de intervalo de 7 min
SECOND_INTERVAL_START = FIRST_INTERVAL_START + timedelta(minutes=7)


def store_readings(app_context, readings: list[tuple[datetime, float]]) -> None:
    """Cada tupla es (momento, potencia en kW)."""
    app_context.readings.insert_readings(
        TEST_DEVICE_ID, [StoredReading(moment, power_kw) for moment, power_kw in readings], received_at=FIXED_NOW
    )


def test_power_is_aggregated_per_7_minute_interval(client, app_context, registered_device):
    store_readings(
        app_context,
        [
            (FIRST_INTERVAL_START + timedelta(minutes=1), 0.1),
            (FIRST_INTERVAL_START + timedelta(minutes=6), 0.3),
            (SECOND_INTERVAL_START + timedelta(minutes=5), 1.2),
        ],
    )

    response = client.get("/api/consumo", params={"device_id": TEST_DEVICE_ID})

    assert response.status_code == 200
    intervals = response.json()
    assert [interval["reading_count"] for interval in intervals] == [2, 1]
    assert intervals[0]["average_power_kw"] == 0.2  # (0.1 kW + 0.3 kW) / 2
    assert intervals[0]["max_power_kw"] == 0.3
    assert intervals[1]["average_power_kw"] == 1.2
    assert datetime.fromisoformat(intervals[0]["interval_start"]) == FIRST_INTERVAL_START


def test_power_query_excludes_readings_outside_range(client, app_context, registered_device):
    store_readings(app_context, [(FIXED_NOW - timedelta(days=2), 5.0)])

    response = client.get("/api/consumo", params={"device_id": TEST_DEVICE_ID})

    assert response.json() == []


def test_power_query_rejects_inverted_or_too_long_ranges(client):
    inverted = {"device_id": TEST_DEVICE_ID, "desde": "2026-11-25T10:00:00Z", "hasta": "2026-11-24T10:00:00Z"}
    too_long = {"device_id": TEST_DEVICE_ID, "desde": "2026-09-01T00:00:00Z", "hasta": "2026-11-25T00:00:00Z"}

    assert client.get("/api/consumo", params=inverted).status_code == 422
    assert client.get("/api/consumo", params=too_long).status_code == 422


def test_power_query_rejects_invalid_device_id(client):
    assert client.get("/api/consumo", params={"device_id": "x' OR 1=1"}).status_code == 422


def test_device_list_reports_time_since_last_reading(client, app_context, registered_device):
    app_context.devices.update_last_seen(TEST_DEVICE_ID, FIXED_NOW - timedelta(minutes=3))

    devices = client.get("/api/dispositivos").json()

    assert devices == [
        {
            "device_id": TEST_DEVICE_ID,
            "last_seen_at": "2026-11-25T18:04:00Z",
            "seconds_since_last_reading": 180,
        }
    ]


def test_device_without_readings_has_no_last_seen(client, registered_device):
    devices = client.get("/api/dispositivos").json()

    assert devices[0]["last_seen_at"] is None
    assert devices[0]["seconds_since_last_reading"] is None


def test_predictions_are_empty_without_model(client, registered_device):
    summary = client.get("/api/predicciones", params={"device_id": TEST_DEVICE_ID}).json()

    assert summary == {"modelo_cargado": False, "latest": None, "history": []}

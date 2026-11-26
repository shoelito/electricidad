from datetime import UTC, datetime, timedelta

from monitor_frontend.api_client import ApiError
from tests.conftest import sample_intervals, sample_prediction

DEVICE = {"device_id": "medidor-01", "last_seen_at": None, "seconds_since_last_reading": 30}


def find_button(app_test, label: str):
    return next(button for button in app_test.button if button.label == label)


def texts(elements) -> list[str]:
    return [element.value for element in elements]


# --- Dashboard ---


def test_dashboard_without_devices_explains_how_they_appear(fake_api, run_view):
    app_test = run_view("dashboard.py")

    assert not app_test.exception
    assert any("Sin medidores todavía" in text for text in texts(app_test.info))


def test_dashboard_reports_backend_connection_errors(fake_api, run_view):
    fake_api.error_to_raise = ApiError("No se pudo conectar con el backend en http://backend-de-prueba")

    app_test = run_view("dashboard.py")

    assert any("No se pudo conectar con el backend" in text for text in texts(app_test.error))


def test_dashboard_shows_metrics_chart_and_device_selector(fake_api, run_view):
    now = datetime.now(UTC)
    fake_api.devices = [DEVICE]
    fake_api.intervals = sample_intervals(now)
    fake_api.prediction_summary = {
        "modelo_cargado": True,
        "latest": sample_prediction(now + timedelta(minutes=40), 0.45, is_peak=False),
        "history": [sample_prediction(now + timedelta(minutes=40), 0.45, is_peak=False)],
    }
    fake_api.model_is_loaded = True

    app_test = run_view("dashboard.py")

    assert not app_test.exception
    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics == {"Consumo reciente": "0,50 kW", "Predicción próxima hora": "0,45 kW", "Estado del sensor": "En línea"}
    assert app_test.sidebar.selectbox[0].value == "medidor-01"
    assert app_test.get("vega_lite_chart")
    assert not app_test.error


def test_dashboard_alerts_upcoming_peak(fake_api, run_view):
    now = datetime.now(UTC)
    fake_api.devices = [DEVICE]
    fake_api.intervals = sample_intervals(now)
    peak = sample_prediction(now + timedelta(minutes=40), 1.6, is_peak=True)
    fake_api.prediction_summary = {"modelo_cargado": True, "latest": peak, "history": [peak]}

    app_test = run_view("dashboard.py")

    assert any("Pico de consumo previsto" in text for text in texts(app_test.error))


def test_train_button_starts_training(fake_api, run_view):
    fake_api.devices = [DEVICE]

    app_test = run_view("dashboard.py")
    find_button(app_test, "Entrenar modelo").click().run()

    assert fake_api.started_trainings == ["medidor-01"]
    assert any("Entrenando con el historial de medidor-01" in text for text in texts(app_test.info))
    assert find_button(app_test, "Entrenar modelo").disabled


def test_train_button_reports_training_already_running(fake_api, run_view):
    fake_api.devices = [DEVICE]
    fake_api.start_training_error = ApiError("Ya hay un entrenamiento en curso", status_code=409)

    app_test = run_view("dashboard.py")
    find_button(app_test, "Entrenar modelo").click().run()

    assert "Ya hay un entrenamiento en curso." in texts(app_test.warning)


def test_dashboard_warns_when_model_does_not_beat_persistence(fake_api, run_view):
    fake_api.devices = [DEVICE]
    fake_api.training = {
        "state": "succeeded",
        "finished_at": "2026-11-25T18:00:00Z",
        "metrics": {"mae_modelo_kw": 0.5, "mae_persistencia_kw": 0.1, "recall_picos_modelo": 0.0, "picos_en_prueba": 0.0},
    }

    app_test = run_view("dashboard.py")

    assert any("todavía no supera a la persistencia" in text for text in texts(app_test.warning))


def test_reload_button_reloads_model(fake_api, run_view):
    fake_api.devices = [DEVICE]

    app_test = run_view("dashboard.py")
    find_button(app_test, "Recargar modelo").click().run()

    assert fake_api.reloaded_models == ["medidor-01"]


# --- Explorador de la API ---

SPECIFICATION = {
    "openapi": "3.1.0",
    "info": {"title": "Monitor eléctrico", "version": "0.1.0", "description": "API de prueba"},
    "paths": {
        "/api/lecturas": {
            "post": {
                "tags": ["ingesta"],
                "summary": "Enviar un lote de lecturas del sensor",
                "operationId": "ingest",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/ReadingBatchIn"}}}},
                "responses": {"201": {"description": "Creado"}},
            }
        },
        "/api/consumo": {
            "get": {
                "tags": ["gestión"],
                "summary": "Consumo agregado cada 7 min",
                "operationId": "consumption",
                "parameters": [
                    {"name": "device_id", "in": "query", "required": True, "schema": {"type": "string", "pattern": "^[a-z0-9-]{3,40}$"}}
                ],
                "responses": {"200": {"description": "OK"}},
            }
        },
    },
    "components": {
        "schemas": {
            "ReadingBatchIn": {
                "type": "object",
                "required": ["device_id"],
                "examples": [{"device_id": "medidor-01", "lecturas": [{"timestamp": "2026-11-25T18:00:00Z", "power_kw": 0.412}]}],
                "properties": {"device_id": {"type": "string"}},
            }
        }
    },
}


def test_api_explorer_lists_operations_and_schemas(fake_api, run_view):
    fake_api.specification = SPECIFICATION

    app_test = run_view("api_explorer.py")

    assert not app_test.exception
    labels = [expander.label for expander in app_test.expander]
    assert "**POST** `/api/lecturas` — Enviar un lote de lecturas del sensor" in labels
    assert "**GET** `/api/consumo` — Consumo agregado cada 7 min" in labels
    assert "ReadingBatchIn" in labels
    assert '"power_kw": 0.412' in app_test.text_area[0].value
    assert app_test.text_input(key="consumption-param-device_id").value == "medidor-01"


def test_api_explorer_sends_request_and_shows_response_and_curl(fake_api, run_view):
    fake_api.specification = SPECIFICATION

    app_test = run_view("api_explorer.py")
    app_test.button(key="consumption-send").click().run()

    assert fake_api.raw_requests == [("GET", "/api/consumo", {"device_id": "medidor-01"}, None)]
    assert any("201 Created" in text for text in texts(app_test.success))
    code_blocks = texts(app_test.code)
    assert any('"aceptadas": 1' in block for block in code_blocks)
    assert any(block.startswith("curl -X GET 'http://backend-de-prueba/api/consumo?device_id=medidor-01'") for block in code_blocks)


def test_api_explorer_filters_operations(fake_api, run_view):
    fake_api.specification = SPECIFICATION

    app_test = run_view("api_explorer.py")
    app_test.text_input[0].input("consumo").run()

    operation_labels = [expander.label for expander in app_test.expander if expander.label.startswith("**")]
    assert operation_labels == ["**GET** `/api/consumo` — Consumo agregado cada 7 min"]


def test_api_explorer_rejects_invalid_json_body(fake_api, run_view):
    fake_api.specification = SPECIFICATION

    app_test = run_view("api_explorer.py")
    app_test.text_area(key="ingest-body").input("{ no es json").run()
    app_test.button(key="ingest-send").click().run()

    assert "El cuerpo de la petición no es JSON válido." in texts(app_test.error)
    assert fake_api.raw_requests == []

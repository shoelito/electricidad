from datetime import UTC, datetime, timedelta

import pytest

from monitor_frontend.formatting import (
    describe_device_status,
    describe_model_status,
    describe_prediction,
    describe_training_metrics,
    describe_training_status,
    format_kilowatts,
    format_local_clock,
    format_percent,
    model_beats_persistence,
)

NOW = datetime(2026, 11, 25, 18, 7, tzinfo=UTC)
GOOD_METRICS = {"mae_modelo_kw": 0.02, "mae_persistencia_kw": 0.2, "recall_picos_modelo": 0.98, "picos_en_prueba": 30.0}


@pytest.mark.parametrize(("kilowatts", "expected"), [(0.3125, "0,31 kW"), (1.0, "1,00 kW"), (12.346, "12,35 kW")])
def test_kilowatts_use_comma_decimal(kilowatts, expected):
    assert format_kilowatts(kilowatts) == expected


@pytest.mark.parametrize(
    ("utc_text", "expected"),
    [("2026-11-25T22:59:00Z", "04:59 p. m."), ("2026-11-25T06:05:00Z", "12:05 a. m."), ("2026-11-25T18:00:00Z", "12:00 p. m.")],
)
def test_clock_is_shown_in_costa_rica_time(utc_text, expected):
    assert format_local_clock(utc_text) == expected


def test_percent_is_rounded():
    assert format_percent(0.976) == "98 %"


@pytest.mark.parametrize(
    ("device", "label", "is_online"),
    [
        (None, "Sin datos", None),
        ({"seconds_since_last_reading": None}, "Sin datos", None),
        ({"seconds_since_last_reading": 120}, "En línea", True),
        ({"seconds_since_last_reading": 16 * 60}, "Desconectado", False),
    ],
)
def test_device_status(device, label, is_online):
    status = describe_device_status(device)

    assert (status.label, status.is_online) == (label, is_online)


def test_prediction_without_data():
    view = describe_prediction({"latest": None}, NOW)

    assert (view.value_text, view.peak_alert_text) == ("—", None)


def test_upcoming_peak_shows_alert():
    latest = {"target_time": "2026-11-25T19:03:00Z", "predicted_power_kw": 1.5, "peak_threshold_kw": 1.2, "is_peak": True}

    view = describe_prediction({"latest": latest}, NOW)

    assert view.value_text == "1,50 kW"
    assert view.detail_text == "Para las 01:03 p. m. · umbral 1,20 kW"
    assert "Pico de consumo previsto a las 01:03 p. m." in view.peak_alert_text


def test_past_peak_does_not_alert():
    latest = {"target_time": iso(NOW - timedelta(minutes=5)), "predicted_power_kw": 1.5, "peak_threshold_kw": 1.2, "is_peak": True}

    assert describe_prediction({"latest": latest}, NOW).peak_alert_text is None


def test_model_status_text():
    assert "Modelo cargado" in describe_model_status(True)
    assert "100 intervalos de 7 min" in describe_model_status(False)


def test_model_must_beat_persistence():
    assert model_beats_persistence(GOOD_METRICS)
    assert not model_beats_persistence({**GOOD_METRICS, "mae_modelo_kw": 0.3})
    assert model_beats_persistence({})


def test_training_metrics_text():
    text = describe_training_metrics(GOOD_METRICS)

    assert "Error promedio 0,02 kW (persistencia 0,20 kW)" in text
    assert "recall de picos 98 %" in text


def test_recall_is_not_evaluable_without_peaks_in_test_period():
    assert "no evaluable" in describe_training_metrics({**GOOD_METRICS, "picos_en_prueba": 0.0})


@pytest.mark.parametrize(
    ("status", "level", "is_running", "text"),
    [
        ({"state": "idle"}, None, False, ""),
        ({"state": "running", "device_id": "medidor-01", "started_at": "2026-11-25T18:00:00Z"}, "info", True, "inició 12:00 p. m."),
        (
            {"state": "succeeded", "finished_at": "2026-11-25T18:00:00Z", "metrics": GOOD_METRICS},
            "success",
            False,
            "Entrenamiento completado (12:00 p. m.). Error promedio",
        ),
        (
            {"state": "succeeded", "finished_at": "2026-11-25T18:00:00Z", "metrics": {**GOOD_METRICS, "mae_modelo_kw": 0.5}},
            "warning",
            False,
            "todavía no supera a la persistencia",
        ),
        ({"state": "failed", "message": "Se necesitan al menos 100"}, "error", False, "No se pudo entrenar: Se necesitan"),
    ],
)
def test_training_status_views(status, level, is_running, text):
    view = describe_training_status(status)

    assert (view.level, view.is_running) == (level, is_running)
    assert text in view.message


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")

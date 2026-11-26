import pandas as pd

from monitor_frontend.charts import (
    AVERAGE_SERIES,
    PREDICTION_SERIES,
    build_chart_data,
    build_power_chart,
)

INTERVALS = [
    {"interval_start": "2026-11-25T18:03:00Z", "average_power_kw": 0.3},
    {"interval_start": "2026-11-25T18:10:00Z", "average_power_kw": 0.4},
]
PREDICTIONS = [
    {"target_time": "2026-11-25T19:06:00Z", "predicted_power_kw": 1.5, "peak_threshold_kw": 1.2},
]


def test_chart_data_uses_costa_rica_local_time():
    data = build_chart_data(INTERVALS, PREDICTIONS)

    assert data["hora"].tolist() == [
        pd.Timestamp("2026-11-25 12:03"),
        pd.Timestamp("2026-11-25 12:10"),
        pd.Timestamp("2026-11-25 13:06"),
    ]
    assert data["serie"].tolist() == [AVERAGE_SERIES, AVERAGE_SERIES, PREDICTION_SERIES]


def test_no_chart_without_data():
    assert build_power_chart([], []) is None


def test_chart_has_average_prediction_and_threshold_layers():
    chart = build_power_chart(INTERVALS, PREDICTIONS)

    specification = chart.to_dict()
    marks = [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"] for layer in specification["layer"]]
    assert marks == ["line", "point", "rule"]


def test_chart_without_predictions_has_no_threshold():
    chart = build_power_chart(INTERVALS, [])

    assert len(chart.to_dict()["layer"]) == 2

"""Gráfica de potencia del dashboard (Altair, que viene incluido con Streamlit)."""

import altair as alt
import pandas as pd

from monitor_frontend.formatting import to_local_time

AVERAGE_SERIES = "Promedio 7 min"
PREDICTION_SERIES = "Predicción"
THRESHOLD_SERIES = "Umbral de pico"
SERIES_COLORS = {AVERAGE_SERIES: "#2563eb", PREDICTION_SERIES: "#d97706", THRESHOLD_SERIES: "#dc2626"}


def build_chart_data(intervals: list[dict], predictions: list[dict]) -> pd.DataFrame:
    """Tabla larga (hora local, kW, serie). Las horas se pasan a Costa Rica para que la gráfica no dependa
    de la zona horaria del navegador."""
    rows = [
        {"hora": _naive_local(interval["interval_start"]), "kw": interval["average_power_kw"], "serie": AVERAGE_SERIES}
        for interval in intervals
    ] + [
        {"hora": _naive_local(prediction["target_time"]), "kw": prediction["predicted_power_kw"], "serie": PREDICTION_SERIES}
        for prediction in predictions
    ]
    return pd.DataFrame(rows, columns=["hora", "kw", "serie"])


def build_power_chart(intervals: list[dict], predictions: list[dict]) -> alt.LayerChart | None:
    data = build_chart_data(intervals, predictions)
    if data.empty:
        return None
    color = alt.Color(
        "serie:N",
        title=None,
        scale=alt.Scale(domain=list(SERIES_COLORS), range=list(SERIES_COLORS.values())),
        legend=alt.Legend(orient="top"),
    )
    x_axis = alt.X("hora:T", title=None, axis=alt.Axis(format="%H:%M"))
    y_axis = alt.Y("kw:Q", title="kW")
    tooltip = [alt.Tooltip("hora:T", title="Hora", format="%H:%M"), alt.Tooltip("kw:Q", title="kW", format=".2f")]

    average_line = (
        alt.Chart(data[data["serie"] == AVERAGE_SERIES]).mark_line().encode(x=x_axis, y=y_axis, color=color, tooltip=tooltip)
    )
    prediction_points = (
        alt.Chart(data[data["serie"] == PREDICTION_SERIES])
        .mark_point(filled=True, size=50)
        .encode(x=x_axis, y=y_axis, color=color, tooltip=tooltip)
    )
    layers = [average_line, prediction_points]
    if predictions:
        threshold = pd.DataFrame({"kw": [predictions[-1]["peak_threshold_kw"]], "serie": [THRESHOLD_SERIES]})
        layers.append(alt.Chart(threshold).mark_rule(strokeDash=[6, 4]).encode(y="kw:Q", color=color))
    return alt.layer(*layers).properties(height=320)


def _naive_local(value: str) -> pd.Timestamp:
    return pd.Timestamp(to_local_time(value).replace(tzinfo=None))

"""Dashboard: consumo, predicción, alerta de pico, gráfica y entrenamiento del modelo."""

from datetime import UTC, datetime

import streamlit as st

from monitor_frontend.api_client import ApiError, get_api_client
from monitor_frontend.charts import build_power_chart
from monitor_frontend.formatting import (
    TrainingStatusView,
    describe_device_status,
    describe_model_status,
    describe_prediction,
    describe_training_status,
    format_kilowatts,
)

# Cada 5 min: alineado con el envío de lotes del sensor y cuida el límite de peticiones de ngrok.
LIVE_DATA_REFRESH_INTERVAL = "5m"
# Solo mientras hay un entrenamiento en curso.
TRAINING_POLL_INTERVAL = "3s"
CONFLICT_STATUS_CODE = 409

client = get_api_client()


def show_training_message(view: TrainingStatusView) -> None:
    show_by_level = {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}
    if view.level:
        show_by_level[view.level](view.message)


@st.fragment(run_every=LIVE_DATA_REFRESH_INTERVAL)
def render_live_data(device_id: str) -> None:
    try:
        device = next((item for item in client.list_devices() if item["device_id"] == device_id), None)
        intervals = client.power_by_interval(device_id)
        summary = client.predictions(device_id)
    except ApiError as error:
        st.error(f"No se pudieron cargar los datos del medidor: {error}")
        return

    prediction = describe_prediction(summary, datetime.now(UTC))
    if prediction.peak_alert_text:
        st.error(prediction.peak_alert_text)

    recent_power, predicted_power, sensor_status = st.columns(3, border=True)
    with recent_power:
        st.metric("Consumo reciente", format_kilowatts(intervals[-1]["average_power_kw"]) if intervals else "—")
        st.caption("Promedio del último intervalo de 7 min")
    with predicted_power:
        st.metric("Predicción próxima hora", prediction.value_text)
        st.caption(prediction.detail_text)
    with sensor_status:
        status = describe_device_status(device)
        st.metric("Estado del sensor", status.label)
        st.caption(status.detail)

    st.subheader("Potencia últimas 24 h (kW)")
    chart = build_power_chart(intervals, summary["history"])
    if chart is None:
        st.caption("No hay lecturas en las últimas 24 h.")
    else:
        st.altair_chart(chart, width="stretch")


@st.fragment(run_every=TRAINING_POLL_INTERVAL)
def render_training_progress() -> None:
    try:
        view = describe_training_status(client.training_status())
    except ApiError as error:
        st.error(f"No se pudo consultar el entrenamiento: {error}")
        return
    if not view.is_running:
        st.rerun()  # terminó: recargar toda la página para mostrar el modelo y la predicción nuevos
    show_training_message(view)


def render_model_section(device_id: str) -> None:
    st.subheader("Modelo de predicción")
    try:
        training_view = describe_training_status(client.training_status())
        model_is_loaded = client.health()["modelo_cargado"]
    except ApiError as error:
        st.error(f"No se pudo consultar el estado del modelo: {error}")
        return

    st.caption(describe_model_status(model_is_loaded))
    if training_view.is_running:
        render_training_progress()
    else:
        show_training_message(training_view)

    train_column, reload_column, _ = st.columns([1, 1, 3])
    if train_column.button("Entrenar modelo", type="primary", disabled=training_view.is_running, width="stretch"):
        try:
            client.start_training(device_id)
        except ApiError as error:
            if error.status_code == CONFLICT_STATUS_CODE:
                st.warning("Ya hay un entrenamiento en curso.")
            else:
                st.error(f"No se pudo iniciar el entrenamiento: {error}")
        else:
            st.rerun()
    if reload_column.button("Recargar modelo", width="stretch"):
        try:
            client.reload_model(device_id)
        except ApiError as error:
            st.error(f"No se pudo recargar el modelo: {error}")
        else:
            st.rerun()


st.title("Monitor eléctrico")

try:
    devices = client.list_devices()
except ApiError as error:
    st.error(f"No se pudo conectar con el backend: {error}")
    st.stop()

if not devices:
    st.info(
        "Sin medidores todavía. Un medidor aparece solo cuando el ESP32 envía su primer lote de lecturas a "
        "`POST /api/lecturas`. Para probar, envíe una lectura desde el **Explorador de la API**."
    )
    st.stop()

with st.sidebar:
    selected_device_id = st.selectbox("Medidor", [device["device_id"] for device in devices])
    if st.button("Actualizar", width="stretch"):
        st.rerun()

render_live_data(selected_device_id)
render_model_section(selected_device_id)

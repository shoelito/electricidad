"""Textos y formatos que muestra el dashboard. Funciones puras, sin Streamlit, para poder probarlas."""

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = ZoneInfo("America/Costa_Rica")
STALE_DEVICE_SECONDS = 15 * 60


@dataclass(frozen=True)
class DeviceStatusView:
    label: str
    is_online: bool | None
    detail: str


@dataclass(frozen=True)
class PredictionView:
    value_text: str
    detail_text: str
    peak_alert_text: str | None


@dataclass(frozen=True)
class TrainingStatusView:
    level: str | None  # "info", "success", "warning", "error" o None si no hay nada que mostrar
    message: str
    is_running: bool


def parse_api_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def to_local_time(value: str | datetime) -> datetime:
    return parse_api_datetime(value).astimezone(LOCAL_TIMEZONE)


def format_kilowatts(kilowatts: float) -> str:
    """Formato de Costa Rica: coma decimal. 0.3125 -> '0,31 kW'."""
    return f"{kilowatts:.2f} kW".replace(".", ",")


def format_local_clock(value: str | datetime) -> str:
    """Hora local de 12 h: '04:59 p. m.'."""
    local = to_local_time(value)
    suffix = "a. m." if local.hour < 12 else "p. m."
    return f"{local.hour % 12 or 12:02d}:{local.minute:02d} {suffix}"


def format_percent(fraction: float) -> str:
    return f"{round(fraction * 100)} %"


def describe_device_status(device: dict | None) -> DeviceStatusView:
    if not device or device.get("seconds_since_last_reading") is None:
        return DeviceStatusView("Sin datos", None, "El sensor aún no ha enviado lecturas")
    seconds = device["seconds_since_last_reading"]
    is_online = seconds <= STALE_DEVICE_SECONDS
    return DeviceStatusView(
        "En línea" if is_online else "Desconectado", is_online, f"Última lectura hace {round(seconds / 60)} min"
    )


def describe_prediction(summary: dict, now: datetime) -> PredictionView:
    latest = summary.get("latest")
    if not latest:
        return PredictionView("—", "Sin predicción todavía", None)
    target_clock = format_local_clock(latest["target_time"])
    peak_is_upcoming = latest["is_peak"] and parse_api_datetime(latest["target_time"]) > now
    peak_alert = (
        f"⚠ Pico de consumo previsto a las {target_clock}: {format_kilowatts(latest['predicted_power_kw'])}. "
        "Considere aplazar el uso de equipos de alto consumo."
        if peak_is_upcoming
        else None
    )
    return PredictionView(
        format_kilowatts(latest["predicted_power_kw"]),
        f"Para las {target_clock} · umbral {format_kilowatts(latest['peak_threshold_kw'])}",
        peak_alert,
    )


def describe_model_status(model_is_loaded: bool) -> str:
    if model_is_loaded:
        return "Modelo cargado. Las predicciones se generan cada 7 min."
    return "No hay modelo entrenado. Pulse «Entrenar modelo» cuando haya al menos 100 intervalos de 7 min (~12 h)."


def model_beats_persistence(metrics: dict) -> bool:
    if "mae_modelo_kw" not in metrics:
        return True
    return metrics["mae_modelo_kw"] < metrics["mae_persistencia_kw"]


def describe_training_metrics(metrics: dict) -> str:
    if "mae_modelo_kw" not in metrics:
        return ""
    if metrics.get("picos_en_prueba") == 0:
        recall_text = "recall de picos no evaluable (no hubo picos en el periodo de prueba)"
    else:
        recall_text = f"recall de picos {format_percent(metrics['recall_picos_modelo'])}"
    text = (
        f" Error promedio {format_kilowatts(metrics['mae_modelo_kw'])} "
        f"(persistencia {format_kilowatts(metrics['mae_persistencia_kw'])}) · {recall_text}."
    )
    if not model_beats_persistence(metrics):
        text += " ⚠ El modelo todavía no supera a la persistencia: espere a tener más días de datos y vuelva a entrenar."
    return text


def describe_training_status(status: dict) -> TrainingStatusView:
    state = status.get("state", "idle")
    if state == "running":
        started = format_local_clock(status["started_at"])
        return TrainingStatusView("info", f"Entrenando con el historial de {status['device_id']} (inició {started})…", True)
    if state == "succeeded":
        metrics = status.get("metrics") or {}
        message = f"Entrenamiento completado ({format_local_clock(status['finished_at'])})."
        message += describe_training_metrics(metrics)
        return TrainingStatusView("success" if model_beats_persistence(metrics) else "warning", message, False)
    if state == "failed":
        return TrainingStatusView("error", f"No se pudo entrenar: {status.get('message')}", False)
    return TrainingStatusView(None, "", False)

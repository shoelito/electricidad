"""Cliente HTTP del backend. El frontend solo se comunica con el backend a través de esta clase."""

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

API_URL_ENVIRONMENT_VARIABLE = "MONITOR_API_URL"
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 15


class ApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RawResponse:
    """Respuesta sin interpretar, para el explorador de la API."""

    status_code: int
    reason: str
    elapsed_milliseconds: int
    body_text: str

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


class MonitorApiClient:
    def __init__(self, base_url: str, http_client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client or httpx.Client(base_url=self.base_url, timeout=REQUEST_TIMEOUT_SECONDS)

    def health(self) -> dict:
        return self._request_json("GET", "/api/salud")

    def list_devices(self) -> list[dict]:
        return self._request_json("GET", "/api/dispositivos")

    def power_by_interval(self, device_id: str) -> list[dict]:
        return self._request_json("GET", "/api/consumo", params={"device_id": device_id})

    def predictions(self, device_id: str) -> dict:
        return self._request_json("GET", "/api/predicciones", params={"device_id": device_id})

    def start_training(self, device_id: str) -> dict:
        return self._request_json("POST", "/api/modelo/entrenar", params={"device_id": device_id})

    def training_status(self) -> dict:
        return self._request_json("GET", "/api/modelo/entrenamiento")

    def reload_model(self, device_id: str) -> dict:
        return self._request_json("POST", "/api/modelo/recargar", params={"device_id": device_id})

    def openapi_specification(self) -> dict:
        return self._request_json("GET", "/openapi.json")

    def send_raw_request(
        self, method: str, path: str, query_parameters: dict[str, str], json_body: Any | None
    ) -> RawResponse:
        """Envía una petición tal cual (explorador de la API). Los errores HTTP no lanzan excepción."""
        started_at = time.perf_counter()
        response = self._send(method, path, params=query_parameters or None, json=json_body)
        return RawResponse(
            status_code=response.status_code,
            reason=response.reason_phrase,
            elapsed_milliseconds=round((time.perf_counter() - started_at) * 1000),
            body_text=response.text,
        )

    def _request_json(self, method: str, path: str, **request_options: Any) -> Any:
        response = self._send(method, path, **request_options)
        if response.status_code >= 400:
            raise ApiError(_extract_error_detail(response), status_code=response.status_code)
        return response.json()

    def _send(self, method: str, path: str, **request_options: Any) -> httpx.Response:
        try:
            return self.http_client.request(method, path, **request_options)
        except httpx.HTTPError as error:
            raise ApiError(f"No se pudo conectar con el backend en {self.base_url} ({error.__class__.__name__})") from error


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail or f"Error {response.status_code} del backend")


@st.cache_resource
def get_api_client() -> MonitorApiClient:
    """Un cliente compartido por todas las sesiones; la URL del backend sale de MONITOR_API_URL."""
    return MonitorApiClient(os.environ.get(API_URL_ENVIRONMENT_VARIABLE, DEFAULT_API_URL))

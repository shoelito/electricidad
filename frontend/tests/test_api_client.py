import httpx
import pytest

from monitor_frontend.api_client import ApiError, MonitorApiClient

BASE_URL = "http://backend-de-prueba"


def client_with(handler) -> MonitorApiClient:
    return MonitorApiClient(BASE_URL, http_client=httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler)))


def test_requests_use_the_expected_paths_and_parameters():
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json=[])

    client = client_with(handler)
    client.list_devices()
    client.power_by_interval("medidor-01")
    client.start_training("medidor-01")

    assert seen_requests == [
        ("GET", "/api/dispositivos", {}),
        ("GET", "/api/consumo", {"device_id": "medidor-01"}),
        ("POST", "/api/modelo/entrenar", {"device_id": "medidor-01"}),
    ]


def test_backend_error_detail_becomes_api_error():
    client = client_with(lambda request: httpx.Response(409, json={"detail": "Ya hay un entrenamiento en curso"}))

    with pytest.raises(ApiError) as error:
        client.start_training("medidor-01")

    assert error.value.status_code == 409
    assert str(error.value) == "Ya hay un entrenamiento en curso"


def test_validation_error_list_is_joined():
    detail = [{"msg": "String should match pattern"}, {"msg": "Field required"}]
    client = client_with(lambda request: httpx.Response(422, json={"detail": detail}))

    with pytest.raises(ApiError, match="String should match pattern; Field required"):
        client.list_devices()


def test_connection_failure_becomes_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión", request=request)

    with pytest.raises(ApiError, match="No se pudo conectar con el backend en http://backend-de-prueba"):
        client_with(handler).health()


def test_raw_request_returns_error_responses_without_raising():
    client = client_with(lambda request: httpx.Response(422, json={"detail": "mal"}))

    response = client.send_raw_request("POST", "/api/lecturas", {}, {"device_id": "x"})

    assert (response.status_code, response.is_success) == (422, False)
    assert response.body_text == '{"detail":"mal"}'

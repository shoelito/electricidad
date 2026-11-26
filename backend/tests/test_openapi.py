import pytest

from app.schemas import ReadingBatchIn

EXPECTED_OPERATIONS = {
    ("post", "/api/lecturas"),
    ("get", "/api/salud"),
    ("get", "/api/dispositivos"),
    ("get", "/api/consumo"),
    ("get", "/api/predicciones"),
    ("post", "/api/modelo/recargar"),
    ("post", "/api/modelo/entrenar"),
    ("get", "/api/modelo/entrenamiento"),
}


@pytest.mark.parametrize(
    "removed_path", ["/", "/dashboard", "/api-docs", "/static/dashboard.js", "/login", "/api/sesion"]
)
def test_backend_serves_only_the_api(client, removed_path):
    """El frontend vive en Streamlit; el backend ya no sirve páginas ni archivos estáticos."""
    assert client.get(removed_path).status_code == 404


def test_openapi_lists_every_api_operation_with_summary(client):
    specification = client.get("/openapi.json").json()

    documented_operations = {
        (method, path) for path, path_item in specification["paths"].items() for method in path_item
    }
    assert documented_operations == EXPECTED_OPERATIONS
    for path, path_item in specification["paths"].items():
        for operation in path_item.values():
            assert operation.get("summary"), f"{path} no tiene summary"


def test_ingestion_requires_no_headers_or_parameters(client):
    ingestion = client.get("/openapi.json").json()["paths"]["/api/lecturas"]["post"]

    assert ingestion.get("parameters", []) == []


def test_openapi_body_example_is_a_valid_batch(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    example = schemas["ReadingBatchIn"]["examples"][0]
    assert ReadingBatchIn.model_validate(example).device_id == "medidor-01"


def test_swagger_from_cdn_is_disabled(client):
    assert client.get("/docs").status_code == 404

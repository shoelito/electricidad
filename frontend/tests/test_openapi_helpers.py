from monitor_frontend.openapi_helpers import (
    build_curl_command,
    build_example,
    describe_constraints,
    describe_type,
    filter_operations,
    format_response_body,
    group_operations_by_tag,
    list_operations,
    resolve_reference,
)

SPECIFICATION = {
    "openapi": "3.1.0",
    "paths": {
        "/api/lecturas": {
            "post": {
                "tags": ["ingesta"],
                "summary": "Enviar un lote de lecturas",
                "operationId": "ingest",
                "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Batch"}}}},
                "responses": {
                    "201": {"description": "Creado", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Result"}}}}
                },
            }
        },
        "/api/consumo": {
            "get": {
                "tags": ["gestión"],
                "summary": "Consumo agregado",
                "parameters": [
                    {"name": "device_id", "in": "query", "required": True, "schema": {"type": "string", "pattern": "^[a-z0-9-]{3,40}$"}},
                    {"name": "desde", "in": "query", "schema": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]}},
                ],
                "responses": {"200": {"description": "OK"}},
            }
        },
    },
    "components": {
        "schemas": {
            "Reading": {
                "type": "object",
                "properties": {"timestamp": {"type": "string", "format": "date-time"}, "power_kw": {"type": "number", "minimum": 0}},
            },
            "Batch": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}, "lecturas": {"type": "array", "items": {"$ref": "#/components/schemas/Reading"}}},
            },
            "BatchWithExample": {"type": "object", "examples": [{"device_id": "medidor-01", "lecturas": []}]},
            "Result": {"type": "object", "properties": {"aceptadas": {"type": "integer"}}},
        }
    },
}


def test_resolve_reference_follows_local_references():
    resolved = resolve_reference(SPECIFICATION, {"$ref": "#/components/schemas/Result"})

    assert list(resolved["properties"]) == ["aceptadas"]


def test_describe_type_handles_references_arrays_and_unions():
    batch = SPECIFICATION["components"]["schemas"]["Batch"]

    assert describe_type(SPECIFICATION, {"$ref": "#/components/schemas/Batch"}) == "Batch"
    assert describe_type(SPECIFICATION, batch["properties"]["lecturas"]) == "Reading[]"
    assert (
        describe_type(SPECIFICATION, {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]})
        == "string (date-time) | null"
    )


def test_describe_constraints_lists_limits_and_patterns():
    assert describe_constraints({"minimum": 0, "maximum": 25}) == "≥ 0, ≤ 25"
    assert describe_constraints({"pattern": "^[a-z]+$", "maxLength": 40}) == "longitud ≤ 40, patrón ^[a-z]+$"
    assert describe_constraints(None) == ""


def test_build_example_creates_nested_object_from_schema():
    assert build_example(SPECIFICATION, {"$ref": "#/components/schemas/Batch"}) == {
        "device_id": "texto",
        "lecturas": [{"timestamp": "1970-01-01T00:00:00Z", "power_kw": 0}],
    }


def test_build_example_prefers_declared_example():
    assert build_example(SPECIFICATION, {"$ref": "#/components/schemas/BatchWithExample"}) == {
        "device_id": "medidor-01",
        "lecturas": [],
    }


def test_build_example_stops_on_recursive_schemas():
    recursive = {"components": {"schemas": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/components/schemas/Node"}}}}}}

    build_example(recursive, {"$ref": "#/components/schemas/Node"})  # no debe entrar en bucle


def test_list_operations_extracts_method_path_parameters_and_body():
    operations = list_operations(SPECIFICATION)

    ingest = next(operation for operation in operations if operation.path == "/api/lecturas")
    assert (ingest.method, ingest.tag, ingest.operation_id) == ("POST", "ingesta", "ingest")
    assert ingest.request_schema == {"$ref": "#/components/schemas/Batch"}
    assert ingest.responses[0].status_code == "201"
    consumption = next(operation for operation in operations if operation.path == "/api/consumo")
    assert [parameter["name"] for parameter in consumption.parameters] == ["device_id", "desde"]


def test_group_and_filter_operations():
    operations = list_operations(SPECIFICATION)

    assert list(group_operations_by_tag(operations)) == ["ingesta", "gestión"]
    assert len(filter_operations(operations, "consumo")) == 1
    assert len(filter_operations(operations, "POST")) == 1
    assert len(filter_operations(operations, "   ")) == 2


def test_curl_command_encodes_query_and_quotes_body():
    command = build_curl_command(
        "GET", "http://localhost:8000/api/consumo", {"device_id": "medidor-01", "desde": "2026-11-25T10:00:00+00:00"}, None
    )

    assert command == "curl -X GET 'http://localhost:8000/api/consumo?device_id=medidor-01&desde=2026-11-25T10%3A00%3A00%2B00%3A00'"


def test_curl_command_with_json_body_escapes_single_quotes():
    command = build_curl_command("POST", "http://localhost:8000/api/lecturas", {}, {"nota": "it's"})

    assert "-H 'Content-Type: application/json'" in command
    assert """-d '{"nota": "it'"'"'s"}'""" in command


def test_format_response_body_indents_json_and_keeps_plain_text():
    assert format_response_body('{"a":1}') == ('{\n  "a": 1\n}', "json")
    assert format_response_body("no es json") == ("no es json", "text")

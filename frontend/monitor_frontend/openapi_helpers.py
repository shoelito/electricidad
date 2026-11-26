"""Lectura de la especificación OpenAPI del backend para el explorador de la API. Funciones puras."""

import json
import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

HTTP_METHODS = ("get", "post", "put", "patch", "delete")
MAX_SCHEMA_DEPTH = 8


@dataclass(frozen=True)
class ResponseDescription:
    status_code: str
    description: str
    schema: dict | None


@dataclass(frozen=True)
class Operation:
    operation_id: str
    method: str
    path: str
    tag: str
    summary: str
    description: str
    parameters: list[dict] = field(default_factory=list)
    request_schema: dict | None = None
    responses: list[ResponseDescription] = field(default_factory=list)


def resolve_reference(specification: dict, schema: dict | None) -> dict:
    if not schema or not isinstance(schema.get("$ref"), str):
        return schema or {}
    node: Any = specification
    for key in schema["$ref"].removeprefix("#/").split("/"):
        node = node.get(key) if isinstance(node, dict) else None
    return node or {}


def reference_name(schema: dict | None) -> str | None:
    if schema and isinstance(schema.get("$ref"), str):
        return schema["$ref"].rsplit("/", 1)[-1]
    return None


def describe_type(specification: dict, schema: dict | None) -> str:
    if not schema:
        return "any"
    name = reference_name(schema)
    if name:
        return name
    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        return " | ".join(describe_type(specification, variant) for variant in variants)
    if schema.get("type") == "array":
        return f"{describe_type(specification, schema.get('items'))}[]"
    if schema.get("format"):
        return f"{schema.get('type')} ({schema['format']})"
    return schema.get("type", "object")


CONSTRAINT_LABELS = (
    ("minimum", "≥ {}"),
    ("exclusiveMinimum", "> {}"),
    ("maximum", "≤ {}"),
    ("exclusiveMaximum", "< {}"),
    ("minLength", "longitud ≥ {}"),
    ("maxLength", "longitud ≤ {}"),
    ("minItems", "elementos ≥ {}"),
    ("maxItems", "elementos ≤ {}"),
    ("pattern", "patrón {}"),
)


def describe_constraints(schema: dict | None) -> str:
    schema = schema or {}
    return ", ".join(label.format(schema[key]) for key, label in CONSTRAINT_LABELS if key in schema)


def build_example(specification: dict, schema: dict | None, depth: int = 0) -> Any:
    if depth > MAX_SCHEMA_DEPTH:
        return None
    resolved = resolve_reference(specification, schema)
    if resolved.get("examples"):
        return resolved["examples"][0]
    for key in ("example", "default"):
        if key in resolved:
            return resolved[key]
    variants = resolved.get("anyOf") or resolved.get("oneOf")
    if variants:
        first_non_null = next((variant for variant in variants if variant.get("type") != "null"), variants[0])
        return build_example(specification, first_non_null, depth + 1)
    schema_type = resolved.get("type") or ("object" if "properties" in resolved else None)
    if schema_type == "object":
        return {
            name: build_example(specification, property_schema, depth + 1)
            for name, property_schema in resolved.get("properties", {}).items()
        }
    if schema_type == "array":
        return [build_example(specification, resolved.get("items"), depth + 1)]
    if schema_type in ("integer", "number"):
        return resolved.get("minimum", 0)
    if schema_type == "boolean":
        return True
    if schema_type == "string":
        if resolved.get("format") == "date-time":
            return "1970-01-01T00:00:00Z"
        if resolved.get("enum"):
            return resolved["enum"][0]
        return "texto"
    return None


def list_operations(specification: dict) -> list[Operation]:
    operations = []
    for path, path_item in specification.get("paths", {}).items():
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not operation:
                continue
            request_body = operation.get("requestBody") or {}
            operations.append(
                Operation(
                    operation_id=operation.get("operationId") or f"{method}_{path}",
                    method=method.upper(),
                    path=path,
                    tag=(operation.get("tags") or ["general"])[0],
                    summary=operation.get("summary", ""),
                    description=operation.get("description", ""),
                    parameters=[resolve_reference(specification, parameter) for parameter in operation.get("parameters", [])],
                    request_schema=request_body.get("content", {}).get("application/json", {}).get("schema"),
                    responses=[
                        ResponseDescription(
                            status_code=status_code,
                            description=response.get("description", ""),
                            schema=response.get("content", {}).get("application/json", {}).get("schema"),
                        )
                        for status_code, response in operation.get("responses", {}).items()
                    ],
                )
            )
    return operations


def group_operations_by_tag(operations: Iterable[Operation]) -> dict[str, list[Operation]]:
    groups: dict[str, list[Operation]] = {}
    for operation in operations:
        groups.setdefault(operation.tag, []).append(operation)
    return groups


def filter_operations(operations: list[Operation], search_text: str) -> list[Operation]:
    normalized_search = search_text.strip().lower()
    if not normalized_search:
        return operations
    return [
        operation
        for operation in operations
        if normalized_search in " ".join([operation.method, operation.path, operation.summary, operation.tag]).lower()
    ]


def build_curl_command(method: str, url: str, query_parameters: dict[str, str], json_body: Any | None) -> str:
    """Comando curl equivalente. Los valores van entre comillas simples de shell (shlex.quote)."""
    query_string = urlencode(query_parameters)
    full_url = f"{url}?{query_string}" if query_string else url
    parts = [f"curl -X {method} {shlex.quote(full_url)}"]
    if json_body is not None:
        parts.append(f"-H {shlex.quote('Content-Type: application/json')}")
        parts.append(f"-d {shlex.quote(json.dumps(json_body, ensure_ascii=False))}")
    return " \\\n  ".join(parts)


def format_response_body(text: str) -> tuple[str, str]:
    """Devuelve (texto, lenguaje para resaltar): JSON indentado si se puede, texto plano si no."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False), "json"
    except ValueError:
        return text, "text"

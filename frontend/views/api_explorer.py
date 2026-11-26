"""Explorador de la API: lee /openapi.json del backend y permite probar cada endpoint."""

import json

import pandas as pd
import streamlit as st

from monitor_frontend.api_client import ApiError, get_api_client
from monitor_frontend.openapi_helpers import (
    Operation,
    build_curl_command,
    build_example,
    describe_constraints,
    describe_type,
    filter_operations,
    format_response_body,
    group_operations_by_tag,
    list_operations,
    reference_name,
    resolve_reference,
)

DEVICE_ID_PATTERN = "^[a-z0-9-]{3,40}$"
EXAMPLE_DEVICE_ID = "medidor-01"

client = get_api_client()


def default_parameter_value(specification: dict, parameter: dict) -> str:
    schema = resolve_reference(specification, parameter.get("schema"))
    return EXAMPLE_DEVICE_ID if schema.get("pattern") == DEVICE_ID_PATTERN else ""


def render_parameter_inputs(specification: dict, operation: Operation) -> tuple[dict[str, str], dict[str, str]]:
    """Devuelve (valores de query, valores de path) escritos por el usuario."""
    query_values: dict[str, str] = {}
    path_values: dict[str, str] = {}
    st.markdown("**Parámetros**")
    for parameter in operation.parameters:
        schema = parameter.get("schema")
        required_mark = " *" if parameter.get("required") else ""
        label = f"{parameter['name']}{required_mark} · {parameter['in']} · {describe_type(specification, schema)}"
        value = st.text_input(
            label,
            value=default_parameter_value(specification, parameter),
            help=describe_constraints(resolve_reference(specification, schema)) or None,
            key=f"{operation.operation_id}-param-{parameter['name']}",
        ).strip()
        if not value:
            continue
        if parameter["in"] == "query":
            query_values[parameter["name"]] = value
        elif parameter["in"] == "path":
            path_values[parameter["name"]] = value
    return query_values, path_values


def send_request(operation: Operation, path: str, query_values: dict[str, str], body_text: str | None) -> None:
    json_body = None
    if body_text is not None:
        try:
            json_body = json.loads(body_text)
        except ValueError:
            st.error("El cuerpo de la petición no es JSON válido.")
            return
    try:
        response = client.send_raw_request(operation.method, path, query_values, json_body)
    except ApiError as error:
        st.error(str(error))
        return

    status_text = f"{response.status_code} {response.reason} · {response.elapsed_milliseconds} ms"
    (st.success if response.is_success else st.error)(f"Respuesta: {status_text}")
    body, language = format_response_body(response.body_text)
    st.code(body or "(sin contenido)", language=language)
    st.markdown("**Comando curl**")
    st.code(build_curl_command(operation.method, f"{client.base_url}{path}", query_values, json_body), language="bash")


def render_operation(specification: dict, operation: Operation) -> None:
    with st.expander(f"**{operation.method}** `{operation.path}` — {operation.summary}"):
        if operation.description:
            st.markdown(operation.description)

        query_values, path_values = ({}, {})
        if operation.parameters:
            query_values, path_values = render_parameter_inputs(specification, operation)

        body_text = None
        if operation.request_schema:
            st.markdown(f"**Cuerpo (JSON)** · `{reference_name(operation.request_schema) or 'objeto'}`")
            body_text = st.text_area(
                "Cuerpo de la petición",
                value=json.dumps(build_example(specification, operation.request_schema), indent=2, ensure_ascii=False),
                height=240,
                key=f"{operation.operation_id}-body",
                label_visibility="collapsed",
            )

        if st.button("Enviar petición", type="primary", key=f"{operation.operation_id}-send"):
            path = operation.path
            for name, value in path_values.items():
                path = path.replace(f"{{{name}}}", value)
            send_request(operation, path, query_values, body_text)

        st.markdown("**Respuestas posibles**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Código": response.status_code,
                        "Descripción": response.description,
                        "Tipo": describe_type(specification, response.schema) if response.schema else "—",
                    }
                    for response in operation.responses
                ]
            ),
            hide_index=True,
        )


def render_schemas(specification: dict) -> None:
    st.subheader("Esquemas")
    st.caption("Estructuras de datos que usan las peticiones y respuestas.")
    for schema_name, schema in sorted(specification.get("components", {}).get("schemas", {}).items()):
        with st.expander(schema_name):
            required_fields = set(schema.get("required", []))
            rows = [
                {
                    "Campo": name + (" *" if name in required_fields else ""),
                    "Tipo": describe_type(specification, property_schema),
                    "Restricciones": describe_constraints(property_schema),
                    "Descripción": property_schema.get("description", ""),
                }
                for name, property_schema in schema.get("properties", {}).items()
            ]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)


st.title("Explorador de la API")

try:
    specification = client.openapi_specification()
except ApiError as error:
    st.error(f"No se pudo cargar /openapi.json: {error}")
    st.stop()

information = specification.get("info", {})
st.subheader(information.get("title", "API"))
st.caption(
    f"Versión {information.get('version', '—')} · OpenAPI {specification.get('openapi', '—')} · "
    f"Backend: {client.base_url}"
)
st.markdown(information.get("description", ""))

search_text = st.text_input("Buscar endpoint", placeholder="Ruta, método o descripción")
operations = filter_operations(list_operations(specification), search_text)
if not operations:
    st.caption("Ningún endpoint coincide.")
for tag, tag_operations in group_operations_by_tag(operations).items():
    st.subheader(tag.capitalize())
    for operation in tag_operations:
        render_operation(specification, operation)

st.divider()
render_schemas(specification)

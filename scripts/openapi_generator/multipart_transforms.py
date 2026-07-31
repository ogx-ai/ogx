# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAPI transforms related to multipart/form-data request schemas."""

from typing import Any


def normalize_multipart_binary_fields(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize multipart binary fields to OpenAPI's ``format: binary`` style.

    FastAPI/Pydantic may emit JSON Schema 2020-12 style binary strings as:
    ``{"type": "string", "contentMediaType": "application/octet-stream"}``.
    Our v1 compatibility checks expect OpenAPI's legacy representation:
    ``{"type": "string", "format": "binary"}``.

    Apply this conversion only for multipart/form-data request schemas.
    """

    components = openapi_schema.get("components", {}).get("schemas", {})
    paths = openapi_schema.get("paths", {})
    if not isinstance(paths, dict):
        return openapi_schema

    ref_prefix = "#/components/schemas/"

    for path_item in paths.values():
        if not isinstance(path_item, dict):
            continue
        for method in ("post", "put", "patch"):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue

            request_body = operation.get("requestBody", {})
            if not isinstance(request_body, dict):
                continue
            content = request_body.get("content", {})
            if not isinstance(content, dict):
                continue

            multipart = content.get("multipart/form-data")
            if not isinstance(multipart, dict):
                continue
            schema = multipart.get("schema")
            if not isinstance(schema, dict):
                continue

            targets: list[dict[str, Any]] = []
            schema_ref = schema.get("$ref")
            if isinstance(schema_ref, str) and schema_ref.startswith(ref_prefix):
                schema_name = schema_ref[len(ref_prefix) :]
                component_schema = components.get(schema_name)
                if isinstance(component_schema, dict):
                    targets.append(component_schema)
            else:
                targets.append(schema)

            for target_schema in targets:
                properties = target_schema.get("properties")
                if not isinstance(properties, dict):
                    continue
                for field_schema in properties.values():
                    if not isinstance(field_schema, dict):
                        continue
                    if (
                        field_schema.get("type") == "string"
                        and field_schema.get("contentMediaType") == "application/octet-stream"
                    ):
                        field_schema.pop("contentMediaType", None)
                        field_schema["format"] = "binary"

    return openapi_schema

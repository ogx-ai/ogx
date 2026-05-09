# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema discovery and collection for OpenAPI generation.
"""

from typing import Any


def _ensure_components_schemas(openapi_schema: dict[str, Any]) -> None:
    """Ensure components.schemas exists in the schema."""
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}


def _extract_and_fix_defs(schema: dict[str, Any], openapi_schema: dict[str, Any]) -> None:
    """
    Extract $defs from a schema, move them to components/schemas, and fix references.
    This handles both TypeAdapter-generated schemas and model_json_schema() schemas.
    """
    if "$defs" in schema:
        defs = schema.pop("$defs")
        for def_name, def_schema in defs.items():
            if def_name not in openapi_schema["components"]["schemas"]:
                openapi_schema["components"]["schemas"][def_name] = def_schema
                # Recursively handle $defs in nested schemas
                _extract_and_fix_defs(def_schema, openapi_schema)

        # Fix any references in the main schema that point to $defs
        def fix_refs_in_schema(obj: Any) -> None:
            if isinstance(obj, dict):
                if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                    obj["$ref"] = obj["$ref"].replace("#/$defs/", "#/components/schemas/")
                for value in obj.values():
                    fix_refs_in_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    fix_refs_in_schema(item)

        fix_refs_in_schema(schema)


def _ensure_json_schema_types_included(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all registered schemas (decorated, explicit, and dynamic) are included in the OpenAPI schema.
    Relies on ogx_api's registry instead of recursively importing every module.
    """
    _ensure_components_schemas(openapi_schema)

    from pydantic import TypeAdapter

    from ogx_api.schema_utils import (
        get_json_schema_type_info,
        iter_dynamic_schema_types,
        iter_json_schema_types,
        iter_registered_schema_types,
    )

    # Handle explicitly registered schemas first (union types, Annotated structs, etc.)
    for registration_info in iter_registered_schema_types():
        schema_type = registration_info.type
        schema_name = registration_info.name
        if schema_name not in openapi_schema["components"]["schemas"]:
            try:
                adapter = TypeAdapter(schema_type)
                schema = adapter.json_schema(ref_template="#/components/schemas/{model}")
                _extract_and_fix_defs(schema, openapi_schema)
                openapi_schema["components"]["schemas"][schema_name] = schema
            except Exception as e:
                print(f"Warning: Failed to generate schema for registered type {schema_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Add @json_schema_type decorated models
    for model in iter_json_schema_types():
        schema_info = get_json_schema_type_info(model)
        schema_name = schema_info.name if schema_info else getattr(model, "__name__", None)
        if not schema_name:
            continue
        if schema_name not in openapi_schema["components"]["schemas"]:
            try:
                if hasattr(model, "model_json_schema"):
                    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
                else:
                    adapter = TypeAdapter(model)
                    schema = adapter.json_schema(ref_template="#/components/schemas/{model}")
                _extract_and_fix_defs(schema, openapi_schema)
                openapi_schema["components"]["schemas"][schema_name] = schema
            except Exception as e:
                print(f"Warning: Failed to generate schema for {schema_name}: {e}")
                continue

    # Include any dynamic models generated while building endpoints
    for model in iter_dynamic_schema_types():
        try:
            schema_name = model.__name__
            if schema_name not in openapi_schema["components"]["schemas"]:
                schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
                _extract_and_fix_defs(schema, openapi_schema)
                openapi_schema["components"]["schemas"][schema_name] = schema
        except Exception:
            continue

    return openapi_schema

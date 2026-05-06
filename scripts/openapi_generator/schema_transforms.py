# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema transformations and fixes for OpenAPI generation.
"""

import copy
from typing import Any

from . import endpoints, schema_collection
from ._schema_output import (
    _apply_legacy_sorting,
    _dedupe_create_response_request_input_union_for_stainless,
    _extract_duplicate_union_types,
    _write_yaml_file,
    validate_openapi_schema,
)
from .state import _extra_body_fields

# re-export so main.py can still access via schema_transforms.<func>
__all__ = [
    "_apply_legacy_sorting",
    "_dedupe_create_response_request_input_union_for_stainless",
    "_extract_duplicate_union_types",
    "_write_yaml_file",
    "validate_openapi_schema",
]


def _fix_ref_references(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix $ref references to point to components/schemas instead of $defs.
    This prevents the YAML dumper from creating a root-level $defs section.
    """

    def fix_refs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                # Replace #/$defs/ with #/components/schemas/
                obj["$ref"] = obj["$ref"].replace("#/$defs/", "#/components/schemas/")
            for value in obj.values():
                fix_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                fix_refs(item)

    fix_refs(openapi_schema)
    return openapi_schema


def _normalize_empty_responses(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert empty 200 responses into 204 No Content."""

    for path_item in openapi_schema.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for method in list(path_item.keys()):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            response_200 = responses.get("200") or responses.get(200)
            if response_200 is None:
                continue
            content = response_200.get("content")
            if content and any(
                isinstance(media, dict) and media.get("schema") not in ({}, None) for media in content.values()
            ):
                continue
            responses.pop("200", None)
            responses.pop(200, None)
            responses["204"] = {"description": response_200.get("description", "No Content")}
    return openapi_schema


def _eliminate_defs_section(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Eliminate $defs section entirely by moving all definitions to components/schemas.
    This matches the structure of the old pyopenapi generator for oasdiff compatibility.
    """
    schema_collection._ensure_components_schemas(openapi_schema)

    # First pass: collect all $defs from anywhere in the schema
    defs_to_move = {}

    def collect_defs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$defs" in obj:
                # Collect $defs for later processing
                for def_name, def_schema in obj["$defs"].items():
                    if def_name not in defs_to_move:
                        defs_to_move[def_name] = def_schema

            # Recursively process all values
            for value in obj.values():
                collect_defs(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_defs(item)

    # Collect all $defs
    collect_defs(openapi_schema)

    # Move all $defs to components/schemas
    for def_name, def_schema in defs_to_move.items():
        if def_name not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"][def_name] = def_schema

    # Also move any existing root-level $defs to components/schemas
    if "$defs" in openapi_schema:
        print(f"Found root-level $defs with {len(openapi_schema['$defs'])} items, moving to components/schemas")
        for def_name, def_schema in openapi_schema["$defs"].items():
            if def_name not in openapi_schema["components"]["schemas"]:
                openapi_schema["components"]["schemas"][def_name] = def_schema
        # Remove the root-level $defs
        del openapi_schema["$defs"]

    # Second pass: remove all $defs sections from anywhere in the schema
    def remove_defs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$defs" in obj:
                del obj["$defs"]

            # Recursively process all values
            for value in obj.values():
                remove_defs(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_defs(item)

    # Remove all $defs sections
    remove_defs(openapi_schema)

    return openapi_schema


def _add_error_responses(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add standard error response definitions to the OpenAPI schema.
    Uses the actual Error model from the codebase for consistency.
    """
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "responses" not in openapi_schema["components"]:
        openapi_schema["components"]["responses"] = {}

    try:
        from ogx_api.datatypes import Error

        schema_collection._ensure_components_schemas(openapi_schema)
        if "Error" not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"]["Error"] = Error.model_json_schema()
    except ImportError:
        pass

    schema_collection._ensure_components_schemas(openapi_schema)
    if "Response" not in openapi_schema["components"]["schemas"]:
        openapi_schema["components"]["schemas"]["Response"] = {"title": "Response", "type": "object"}

    # Define standard HTTP error responses
    error_responses = {
        400: {
            "name": "BadRequest400",
            "description": "The request was invalid or malformed",
            "example": {"status": 400, "title": "Bad Request", "detail": "The request was invalid or malformed"},
        },
        429: {
            "name": "TooManyRequests429",
            "description": "The client has sent too many requests in a given amount of time",
            "example": {
                "status": 429,
                "title": "Too Many Requests",
                "detail": "You have exceeded the rate limit. Please try again later.",
            },
        },
        500: {
            "name": "InternalServerError500",
            "description": "The server encountered an unexpected error",
            "example": {"status": 500, "title": "Internal Server Error", "detail": "An unexpected error occurred"},
        },
    }

    # Add each error response to the schema
    for _, error_info in error_responses.items():
        response_name = error_info["name"]
        openapi_schema["components"]["responses"][response_name] = {
            "description": error_info["description"],
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/Error"}, "example": error_info["example"]}
            },
        }

    # Add a default error response
    openapi_schema["components"]["responses"]["DefaultError"] = {
        "description": "An error occurred",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }

    return openapi_schema


def _fix_path_parameters(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix path parameter resolution issues by adding explicit parameter definitions.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    for path, path_item in openapi_schema["paths"].items():
        # Extract path parameters from the URL
        path_params = endpoints._extract_path_parameters(path)

        if not path_params:
            continue

        # Add parameters to each operation in this path
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method in path_item and isinstance(path_item[method], dict):
                operation = path_item[method]
                if "parameters" not in operation:
                    operation["parameters"] = []

                # Add path parameters that aren't already defined
                existing_param_names = {p.get("name") for p in operation["parameters"] if p.get("in") == "path"}
                for param in path_params:
                    if param["name"] not in existing_param_names:
                        operation["parameters"].append(param)

    return openapi_schema


def _get_schema_title(item: dict[str, Any]) -> str | None:
    """Extract a title for a schema item to use in union variant names."""
    if "$ref" in item:
        return item["$ref"].split("/")[-1]
    elif "type" in item:
        type_val = item["type"]
        if type_val == "null":
            return None
        if type_val == "array" and "items" in item:
            items = item["items"]
            if isinstance(items, dict):
                if "anyOf" in items or "oneOf" in items:
                    nested_union = items.get("anyOf") or items.get("oneOf")
                    if isinstance(nested_union, list) and len(nested_union) > 0:
                        nested_types = []
                        for nested_item in nested_union:
                            if isinstance(nested_item, dict):
                                if "$ref" in nested_item:
                                    nested_types.append(nested_item["$ref"].split("/")[-1])
                                elif "oneOf" in nested_item:
                                    one_of_items = nested_item.get("oneOf", [])
                                    if one_of_items and isinstance(one_of_items[0], dict) and "$ref" in one_of_items[0]:
                                        base_name = one_of_items[0]["$ref"].split("/")[-1].split("-")[0]
                                        nested_types.append(f"{base_name}Union")
                                    else:
                                        nested_types.append("Union")
                                elif "type" in nested_item and nested_item["type"] != "null":
                                    nested_types.append(nested_item["type"])
                        if nested_types:
                            unique_nested = list(dict.fromkeys(nested_types))
                            # Use more descriptive names for better code generation
                            if len(unique_nested) <= 3:
                                return f"list[{' | '.join(unique_nested)}]"
                            else:
                                # Include first few types for better naming
                                return f"list[{unique_nested[0]} | {unique_nested[1]} | ...]"
                        return "list[Union]"
                elif "$ref" in items:
                    return f"list[{items['$ref'].split('/')[-1]}]"
                elif "type" in items:
                    return f"list[{items['type']}]"
            return "array"
        return type_val
    elif "title" in item:
        return item["title"]
    return None


def _add_titles_to_unions(obj: Any, parent_key: str | None = None) -> None:
    """Recursively add titles to union schemas (anyOf/oneOf) to help code generators infer names."""
    if isinstance(obj, dict):
        # Check if this is a union schema (anyOf or oneOf)
        if "anyOf" in obj or "oneOf" in obj:
            union_type = "anyOf" if "anyOf" in obj else "oneOf"
            union_items = obj[union_type]

            if isinstance(union_items, list) and len(union_items) > 0:
                # Skip simple nullable unions (type | null) - these don't need titles
                is_simple_nullable = (
                    len(union_items) == 2
                    and any(isinstance(item, dict) and item.get("type") == "null" for item in union_items)
                    and any(
                        isinstance(item, dict) and "type" in item and item.get("type") != "null" for item in union_items
                    )
                    and not any(
                        isinstance(item, dict) and ("$ref" in item or "anyOf" in item or "oneOf" in item)
                        for item in union_items
                    )
                )

                if is_simple_nullable:
                    # Remove title from simple nullable unions if it exists
                    if "title" in obj:
                        del obj["title"]
                else:
                    # Add titles to individual union variants that need them
                    for item in union_items:
                        if isinstance(item, dict):
                            # Skip null types
                            if item.get("type") == "null":
                                continue
                            # Add title to complex variants (arrays with unions, nested unions, etc.)
                            # Also add to simple types if they're part of a complex union
                            needs_title = (
                                "items" in item
                                or "anyOf" in item
                                or "oneOf" in item
                                or ("$ref" in item and "title" not in item)
                            )
                            if needs_title and "title" not in item:
                                variant_title = _get_schema_title(item)
                                if variant_title:
                                    item["title"] = variant_title

                    # Try to infer a meaningful title from the union items for the parent
                    titles = []
                    for item in union_items:
                        if isinstance(item, dict):
                            title = _get_schema_title(item)
                            if title:
                                titles.append(title)

                    if titles:
                        # Create a title from the union items
                        unique_titles = list(dict.fromkeys(titles))  # Preserve order, remove duplicates
                        if len(unique_titles) <= 3:
                            title = " | ".join(unique_titles)
                        else:
                            title = f"{unique_titles[0]} | ... ({len(unique_titles)} variants)"
                        # Always set the title for unions to help code generators
                        # This will replace generic property titles with union-specific ones
                        obj["title"] = title
                    elif "title" not in obj and parent_key:
                        # Use parent key as fallback only if no title exists
                        obj["title"] = f"{parent_key.title()}Union"

        # Recursively process all values
        for key, value in obj.items():
            _add_titles_to_unions(value, key)
    elif isinstance(obj, list):
        for item in obj:
            _add_titles_to_unions(item, parent_key)


def _restore_const_enum_defaults(openapi_schema: dict[str, Any]) -> None:
    """Restore defaults on single-value enum ``object`` fields where the OpenAI spec expects them.

    ``_convert_standalone_const_to_enum`` strips all defaults from single-value
    enums because most OpenAI schemas omit them.  A handful of schemas (Conversations,
    Responses, Compact, Chat list) DO include a default on their ``object`` field.
    This function adds the default back for those specific schemas so the conformance
    checker doesn't flag them as mismatches.
    """
    schemas = openapi_schema.get("components", {}).get("schemas", {})

    # Mapping of our schema name → property name → expected default value.
    # These correspond to OpenAI schemas that have explicit defaults on
    # single-value enum fields (verified against openai-spec-2.3.0.yml).
    _defaults_to_restore: dict[str, dict[str, str]] = {
        "Conversation": {"object": "conversation"},
        "ListOpenAIChatCompletionResponse": {"object": "list"},
        "OpenAICompactedResponse": {"object": "response.compaction"},
        "OpenAIResponseObject": {"object": "response"},
        "OpenAIResponseObjectWithInput": {"object": "response"},
    }

    for schema_name, props_to_fix in _defaults_to_restore.items():
        schema_def = schemas.get(schema_name)
        if not isinstance(schema_def, dict) or "properties" not in schema_def:
            continue
        for prop_name, default_val in props_to_fix.items():
            prop = schema_def["properties"].get(prop_name)
            if isinstance(prop, dict) and "enum" in prop and prop["enum"] == [default_val] and "default" not in prop:
                prop["default"] = default_val


def _convert_standalone_const_to_enum(obj: Any) -> None:
    """Convert standalone const values to single-value enums to match OpenAI spec style.

    Converts: {const: "value", type: "string", default: "value"}
    To:       {enum: ["value"], type: "string"}

    OpenAI uses enum with a single value rather than const. Defaults are stripped
    because most OpenAI schemas omit defaults on single-value enums.  Schemas that
    need a default restored are handled individually in ``_fix_schema_issues``.
    """
    if isinstance(obj, dict):
        if "const" in obj and "anyOf" not in obj:
            const_val = obj.pop("const")
            obj["enum"] = [const_val]
            if "default" in obj and obj["default"] == const_val:
                del obj["default"]

        for value in obj.values():
            _convert_standalone_const_to_enum(value)
    elif isinstance(obj, list):
        for item in obj:
            _convert_standalone_const_to_enum(item)


def _convert_anyof_const_to_enum(obj: Any) -> None:
    """Convert anyOf with multiple const string values to a proper enum."""
    if isinstance(obj, dict):
        if "anyOf" in obj:
            any_of = obj["anyOf"]
            if isinstance(any_of, list):
                # Check if all items are const string values
                const_values = []
                has_null = False
                can_convert = True
                for item in any_of:
                    if isinstance(item, dict):
                        if item.get("type") == "null":
                            has_null = True
                        elif item.get("type") == "string" and "const" in item:
                            const_values.append(item["const"])
                        else:
                            # Not a simple const pattern, skip conversion for this anyOf
                            can_convert = False
                            break

                # If we have const values and they're all strings, convert to enum
                if can_convert and const_values and len(const_values) == len(any_of) - (1 if has_null else 0):
                    # Convert to enum
                    obj["type"] = "string"
                    obj["enum"] = const_values
                    # Preserve default if present, otherwise try to get from first const item
                    if "default" not in obj:
                        for item in any_of:
                            if isinstance(item, dict) and "const" in item:
                                obj["default"] = item["const"]
                                break
                    # Remove anyOf
                    del obj["anyOf"]
                    # Handle nullable
                    if has_null:
                        obj["nullable"] = True
                    # Remove title if it's just "string"
                    if obj.get("title") == "string":
                        del obj["title"]

        # Recursively process all values
        for value in obj.values():
            _convert_anyof_const_to_enum(value)
    elif isinstance(obj, list):
        for item in obj:
            _convert_anyof_const_to_enum(item)


def _fix_schema_recursive(obj: Any) -> None:
    """Recursively fix schema issues: exclusiveMinimum and null defaults."""
    if isinstance(obj, dict):
        if "exclusiveMinimum" in obj and isinstance(obj["exclusiveMinimum"], int | float):
            obj["minimum"] = obj.pop("exclusiveMinimum")
        if "default" in obj and obj["default"] is None:
            del obj["default"]
            obj["nullable"] = True
        for value in obj.values():
            _fix_schema_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _fix_schema_recursive(item)


def _clean_description(description: str) -> str:
    """Remove :param, :type, :returns, and other docstring metadata from description."""
    if not description:
        return description

    lines = description.split("\n")
    cleaned_lines = []
    skip_until_empty = False

    for line in lines:
        stripped = line.strip()
        # Skip lines that start with docstring metadata markers
        if stripped.startswith(
            (":param", ":type", ":return", ":returns", ":raises", ":exception", ":yield", ":yields", ":cvar")
        ):
            skip_until_empty = True
            continue
        # If we're skipping and hit an empty line, resume normal processing
        if skip_until_empty:
            if not stripped:
                skip_until_empty = False
            continue
        # Include the line if we're not skipping
        cleaned_lines.append(line)

    # Join and strip trailing whitespace
    result = "\n".join(cleaned_lines).strip()
    return result


def _clean_schema_descriptions(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Clean descriptions in schema definitions by removing docstring metadata."""
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]
    for schema_def in schemas.values():
        if isinstance(schema_def, dict) and "description" in schema_def and isinstance(schema_def["description"], str):
            schema_def["description"] = _clean_description(schema_def["description"])

    return openapi_schema


def _add_extra_body_params_extension(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add x-ogx-extra-body-params extension to requestBody for endpoints with ExtraBodyField parameters.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    from pydantic import TypeAdapter

    for path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method not in path_item:
                continue

            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Check if we have extra body fields for this path/method
            key = (path, method.upper())
            if key not in _extra_body_fields:
                continue

            extra_body_params = _extra_body_fields[key]

            # Ensure requestBody exists
            if "requestBody" not in operation:
                continue

            request_body = operation["requestBody"]
            if not isinstance(request_body, dict):
                continue

            # Get the schema from requestBody
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema_ref = json_content.get("schema", {})

            # Remove extra body fields from the schema if they exist as properties
            # Handle both $ref schemas and inline schemas
            if isinstance(schema_ref, dict):
                if "$ref" in schema_ref:
                    # Schema is a reference - remove from the referenced schema
                    ref_path = schema_ref["$ref"]
                    if ref_path.startswith("#/components/schemas/"):
                        schema_name = ref_path.split("/")[-1]
                        if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
                            schema_def = openapi_schema["components"]["schemas"].get(schema_name)
                            if isinstance(schema_def, dict) and "properties" in schema_def:
                                for param_name, _, _ in extra_body_params:
                                    if param_name in schema_def["properties"]:
                                        del schema_def["properties"][param_name]
                                        # Also remove from required if present
                                        if "required" in schema_def and param_name in schema_def["required"]:
                                            schema_def["required"].remove(param_name)
                elif "properties" in schema_ref:
                    # Schema is inline - remove directly from it
                    for param_name, _, _ in extra_body_params:
                        if param_name in schema_ref["properties"]:
                            del schema_ref["properties"][param_name]
                            # Also remove from required if present
                            if "required" in schema_ref and param_name in schema_ref["required"]:
                                schema_ref["required"].remove(param_name)

            # Build the extra body params schema
            extra_params_schema = {}
            for param_name, param_type, description in extra_body_params:
                try:
                    # Generate JSON schema for the parameter type
                    adapter = TypeAdapter(param_type)
                    param_schema = adapter.json_schema(ref_template="#/components/schemas/{model}")

                    # Add description if provided
                    if description:
                        param_schema["description"] = description

                    extra_params_schema[param_name] = param_schema
                except Exception:
                    # If we can't generate schema, skip this parameter
                    continue

            if extra_params_schema:
                # Add the extension to requestBody
                if "x-ogx-extra-body-params" not in request_body:
                    request_body["x-ogx-extra-body-params"] = extra_params_schema

    return openapi_schema


def _remove_query_params_from_body_endpoints(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove query parameters from POST/PUT/PATCH endpoints that have a request body.
    FastAPI sometimes infers parameters as query params even when they should be in the request body.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    body_methods = {"post", "put", "patch"}

    for _path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        for method in body_methods:
            if method not in path_item:
                continue

            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Check if this operation has a request body
            has_request_body = "requestBody" in operation and operation["requestBody"]

            if has_request_body:
                # Remove all query parameters (parameters with "in": "query")
                if "parameters" in operation:
                    # Filter out query parameters, keep path and header parameters
                    operation["parameters"] = [
                        param
                        for param in operation["parameters"]
                        if isinstance(param, dict) and param.get("in") != "query"
                    ]
                    # Remove the parameters key if it's now empty
                    if not operation["parameters"]:
                        del operation["parameters"]

    return openapi_schema


def _remove_request_bodies_from_get_endpoints(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove request bodies from GET endpoints and convert their parameters to query parameters.

    GET requests should never have request bodies - all parameters should be query parameters.
    This function removes any requestBody that FastAPI may have incorrectly added to GET endpoints
    and converts any parameters in the requestBody to query parameters.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    for _path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        # Check GET method specifically
        if "get" in path_item:
            operation = path_item["get"]
            if not isinstance(operation, dict):
                continue

            if "requestBody" in operation:
                request_body = operation["requestBody"]
                # Extract parameters from requestBody and convert to query parameters
                if isinstance(request_body, dict) and "content" in request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})

                    if "parameters" not in operation:
                        operation["parameters"] = []
                    elif not isinstance(operation["parameters"], list):
                        operation["parameters"] = []

                    # If the schema has properties, convert each to a query parameter
                    if isinstance(schema, dict) and "properties" in schema:
                        for param_name, param_schema in schema["properties"].items():
                            # Check if this parameter is already in the parameters list
                            existing_param = None
                            for existing in operation["parameters"]:
                                if isinstance(existing, dict) and existing.get("name") == param_name:
                                    existing_param = existing
                                    break

                            if not existing_param:
                                # Create a new query parameter from the requestBody property
                                required = param_name in schema.get("required", [])
                                query_param = {
                                    "name": param_name,
                                    "in": "query",
                                    "required": required,
                                    "schema": param_schema,
                                }
                                # Add description if present
                                if "description" in param_schema:
                                    query_param["description"] = param_schema["description"]
                                operation["parameters"].append(query_param)
                    elif isinstance(schema, dict):
                        # Handle direct schema (not a model with properties)
                        # Try to infer parameter name from schema title
                        param_name = schema.get("title", "").lower().replace(" ", "_")
                        if param_name:
                            # Check if this parameter is already in the parameters list
                            existing_param = None
                            for existing in operation["parameters"]:
                                if isinstance(existing, dict) and existing.get("name") == param_name:
                                    existing_param = existing
                                    break

                            if not existing_param:
                                # Create a new query parameter from the requestBody schema
                                query_param = {
                                    "name": param_name,
                                    "in": "query",
                                    "required": False,  # Default to optional for GET requests
                                    "schema": schema,
                                }
                                # Add description if present
                                if "description" in schema:
                                    query_param["description"] = schema["description"]
                                operation["parameters"].append(query_param)

                # Remove request body from GET endpoint
                del operation["requestBody"]

    return openapi_schema


def _remove_type_object_from_openai_schemas(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Remove 'type: object' from specific schemas that omit it in the OpenAI spec.

    Most OpenAI schemas (766 of 772) include ``type: object`` alongside
    ``properties``.  Only 6 omit it.  We strip the field only from those
    specific schemas to avoid hurting conformance on the majority that keep it.
    """
    # Only these schemas in the OpenAI spec omit type: object while having properties.
    # Includes both the OpenAI schema names and our equivalent schema names.
    schemas_without_type_object = {
        "ListMessagesResponse",
        "ListRunStepsResponse",
        "ListVectorStoreFilesResponse",
        "ListVectorStoresResponse",
        "Model",
        "OpenAIFile",
        "OpenAIFileObject",
        "OpenAIModel",
    }

    schemas = openapi_schema.get("components", {}).get("schemas", {})
    for schema_name, schema_def in schemas.items():
        if (
            schema_name in schemas_without_type_object
            and isinstance(schema_def, dict)
            and schema_def.get("type") == "object"
            and "properties" in schema_def
        ):
            del schema_def["type"]
    return openapi_schema


def _strip_titles_recursive(obj: Any) -> None:
    """Recursively remove 'title' fields from a schema.

    Pydantic auto-generates titles for all properties, but OpenAI's spec
    generally omits them. Removing titles helps oasdiff match schemas
    that are otherwise structurally identical.
    """
    if isinstance(obj, dict):
        obj.pop("title", None)
        for value in obj.values():
            _strip_titles_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_titles_recursive(item)


def _rename_schema_component(openapi_schema: dict[str, Any], old_name: str, new_name: str) -> None:
    """Rename a schema component and update all $ref references throughout the spec."""
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    if old_name not in schemas:
        return

    # Move the schema definition
    schemas[new_name] = schemas.pop(old_name)

    old_ref = f"#/components/schemas/{old_name}"
    new_ref = f"#/components/schemas/{new_name}"

    def _update_refs(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("$ref") == old_ref:
                obj["$ref"] = new_ref
            for value in obj.values():
                _update_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                _update_refs(item)

    _update_refs(openapi_schema)


def _inline_component_refs(openapi_schema: dict[str, Any], components_to_inline: set[str]) -> None:
    """Inline specific component $refs to match OpenAI's spec style.

    Some components in OpenAI's spec are defined inline rather than as separate
    named components. This function resolves specified $refs by replacing them
    with the actual schema content, which allows oasdiff to match the schemas.

    Handles both anyOf/oneOf variant refs and direct property refs.
    """
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    prefix = "#/components/schemas/"

    def _get_ref_name(ref: str) -> str:
        return ref[len(prefix) :] if ref.startswith(prefix) else ""

    def _inline_refs(obj: Any) -> None:
        if isinstance(obj, dict):
            # Handle anyOf/oneOf arrays that contain $refs to inline
            for key in ("anyOf", "oneOf"):
                if key in obj and isinstance(obj[key], list):
                    new_items = []
                    for item in obj[key]:
                        if isinstance(item, dict) and "$ref" in item:
                            name = _get_ref_name(item["$ref"])
                            if name in components_to_inline and name in schemas:
                                resolved = copy.deepcopy(schemas[name])
                                resolved.pop("title", None)
                                resolved.pop("description", None)
                                new_items.append(resolved)
                                continue
                        new_items.append(item)
                    obj[key] = new_items

            # Handle direct $ref properties (e.g., function: {$ref: ...})
            for key, value in list(obj.items()):
                if isinstance(value, dict) and "$ref" in value:
                    name = _get_ref_name(value["$ref"])
                    if name in components_to_inline and name in schemas:
                        resolved = copy.deepcopy(schemas[name])
                        resolved.pop("title", None)
                        # Preserve description from the referencing field if present
                        if "description" in value:
                            resolved["description"] = value["description"]
                        obj[key] = resolved

            # Recurse into all values
            for value in obj.values():
                _inline_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                _inline_refs(item)

    _inline_refs(openapi_schema)


def _fix_schema_issues(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Fix common schema issues: exclusiveMinimum, null defaults, and add titles to unions."""
    # Convert standalone const values to single-value enums (OpenAI style)
    _convert_standalone_const_to_enum(openapi_schema)

    # Convert anyOf with const values to enums across the entire schema
    _convert_anyof_const_to_enum(openapi_schema)

    # Restore defaults on single-value enums where the OpenAI spec expects them.
    # _convert_standalone_const_to_enum strips all defaults; these specific schemas
    # need them back to match the reference spec (e.g. CompactResource.object).
    _restore_const_enum_defaults(openapi_schema)

    # Align tool call schemas with OpenAI's spec BEFORE inlining (inlining copies schema content).
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        tc_schema = openapi_schema["components"]["schemas"].get("OpenAIChatCompletionToolCall")
        if tc_schema:
            props = tc_schema.get("properties", {})
            # Make id non-nullable (remove anyOf, use plain string).
            # The Pydantic model keeps id optional for streaming delta parsing,
            # but the response schema should match OpenAI's non-nullable definition.
            if "id" in props and "anyOf" in props["id"]:
                non_null = [s for s in props["id"]["anyOf"] if s.get("type") != "null"]
                if len(non_null) == 1:
                    desc = props["id"].get("description", "")
                    props["id"] = non_null[0]
                    if desc:
                        props["id"]["description"] = desc
            # Make function non-nullable (remove anyOf, use plain $ref).
            # Same rationale as id above.
            if "function" in props and "anyOf" in props["function"]:
                non_null = [s for s in props["function"]["anyOf"] if s.get("type") != "null"]
                if len(non_null) == 1:
                    desc = props["function"].get("description", "")
                    props["function"] = non_null[0]
                    if desc:
                        props["function"]["description"] = desc
            # Remove 'index' property (only used for streaming chunks, not in OpenAI response type)
            if "index" in props:
                del props["index"]
            # Strip auto-generated titles from properties (OpenAI doesn't include them)
            _strip_titles_recursive(tc_schema)
            # Set required fields to match OpenAI
            tc_schema["required"] = ["id", "type", "function"]

        ctc_schema = openapi_schema["components"]["schemas"].get("OpenAIChatCompletionCustomToolCall")
        if ctc_schema:
            _strip_titles_recursive(ctc_schema)
            # Set required fields to match OpenAI's ChatCompletionMessageCustomToolCall
            ctc_schema["required"] = ["id", "type", "custom"]

    # Inline specific component refs to match OpenAI's spec style.
    # This must run AFTER the schema fixes above since it copies the schema content.
    _inline_component_refs(
        openapi_schema,
        {
            "OpenAIChoiceLogprobs",
            "OpenAIChatCompletionToolCallFunction",
            "OpenAIChatCompletionCustomToolCallFunction",
        },
    )

    # Rename tool call components to match OpenAI's names for oasdiff matching.
    _rename_schema_component(openapi_schema, "OpenAIChatCompletionToolCall", "ChatCompletionMessageToolCall")
    _rename_schema_component(
        openapi_schema, "OpenAIChatCompletionCustomToolCall", "ChatCompletionMessageCustomToolCall"
    )

    # Add discriminator to tool_calls items anyOf (OpenAI uses propertyName: "type")
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        msg_schema = openapi_schema["components"]["schemas"].get("OpenAIChatCompletionResponseMessage")
        if msg_schema:
            tc_prop = msg_schema.get("properties", {}).get("tool_calls", {})
            items = tc_prop.get("items", {})
            if "anyOf" in items:
                items["discriminator"] = {"propertyName": "type"}

    # Fix other schema issues and add titles to unions
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        for schema_name, schema_def in openapi_schema["components"]["schemas"].items():
            _fix_schema_recursive(schema_def)
            _add_titles_to_unions(schema_def, schema_name)

    return openapi_schema

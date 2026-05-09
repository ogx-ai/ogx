# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Output-stage schema transforms: Stainless deduplication, YAML serialization,
legacy sorting for diff stability, duplicate union extraction, and validation.
"""

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError

from ._legacy_order import (
    LEGACY_OPERATION_KEYS,
    LEGACY_PATH_ORDER,
    LEGACY_RESPONSE_ORDER,
    LEGACY_SCHEMA_ORDER,
    LEGACY_SECURITY,
    LEGACY_TAG_GROUPS,
    LEGACY_TAGS,
)


def _dedupe_create_response_request_input_union_for_stainless(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate inline unions in request `input` properties for Stainless.

    The stable OpenAPI spec intentionally preserves legacy structure for oasdiff
    compatibility, but Stainless codegen treats duplicated inline unions as separate
    types and can generate name clashes.

    This transform is intended to run only on the combined (stainless) spec.
    It handles both CreateResponseRequest and CompactResponseRequest which share
    the same OpenAIResponseInput union type with duplicate OpenAIResponseMessage refs.
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    def _collect_refs(obj: Any, refs: set[str]) -> None:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if isinstance(ref, str):
                refs.add(ref)
            for value in obj.values():
                _collect_refs(value, refs)
        elif isinstance(obj, list):
            for value in obj:
                _collect_refs(value, refs)

    def _is_direct_ref_item(item: dict[str, Any]) -> bool:
        if "$ref" not in item:
            return False
        container_keys = {"oneOf", "anyOf", "items", "properties", "additionalProperties"}
        return not any(key in item for key in container_keys)

    def _dedupe_input_union(schema_name: str) -> None:
        request_schema = schemas.get(schema_name)
        if not isinstance(request_schema, dict):
            return

        properties = request_schema.get("properties")
        if not isinstance(properties, dict):
            return

        input_prop = properties.get("input")
        if not isinstance(input_prop, dict):
            return

        any_of = input_prop.get("anyOf")
        if not isinstance(any_of, list):
            return

        array_schema: dict[str, Any] | None = None
        for item in any_of:
            if isinstance(item, dict) and (item.get("type") == "array" or "items" in item):
                array_schema = item
                break

        if array_schema is None:
            return

        items_schema = array_schema.get("items")
        if not isinstance(items_schema, dict):
            return

        items_any_of = items_schema.get("anyOf")
        if not isinstance(items_any_of, list):
            return

        refs_in_nested_unions: set[str] = set()
        for item in items_any_of:
            if isinstance(item, dict) and not _is_direct_ref_item(item):
                _collect_refs(item, refs_in_nested_unions)

        if not refs_in_nested_unions:
            return

        deduplicated: list[Any] = []
        for item in items_any_of:
            if isinstance(item, dict) and _is_direct_ref_item(item) and item["$ref"] in refs_in_nested_unions:
                continue
            deduplicated.append(item)

        items_schema["anyOf"] = deduplicated

    _dedupe_input_union("CreateResponseRequest")
    _dedupe_input_union("CompactResponseRequest")

    return openapi_schema


def _convert_multiline_strings_to_literal(obj: Any) -> Any:
    """Recursively convert multi-line strings to LiteralScalarString for YAML block scalar formatting."""
    try:
        from ruamel.yaml.scalarstring import LiteralScalarString

        if isinstance(obj, str) and "\n" in obj:
            return LiteralScalarString(obj)
        elif isinstance(obj, dict):
            return {key: _convert_multiline_strings_to_literal(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_convert_multiline_strings_to_literal(item) for item in obj]
        else:
            return obj
    except ImportError:
        return obj


def _write_yaml_file(file_path: Path, schema: dict[str, Any]) -> None:
    """Write schema to YAML file using ruamel.yaml if available, otherwise standard yaml."""
    try:
        from ruamel.yaml import YAML

        yaml_writer = YAML()
        yaml_writer.default_flow_style = False
        yaml_writer.sort_keys = False
        yaml_writer.width = 4096
        yaml_writer.allow_unicode = True
        schema = _convert_multiline_strings_to_literal(schema)
        with open(file_path, "w") as f:
            yaml_writer.dump(schema, f)
    except ImportError:
        with open(file_path, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    # Post-process to remove trailing whitespace from all lines
    with open(file_path) as f:
        lines = f.readlines()

    # Strip trailing whitespace from each line, preserving newlines
    cleaned_lines = [line.rstrip() + "\n" if line.endswith("\n") else line.rstrip() for line in lines]

    with open(file_path, "w") as f:
        f.writelines(cleaned_lines)


def _apply_legacy_sorting(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Temporarily match the legacy ordering from origin/main so diffs are easier to read.
    Remove this once the generator output stabilizes and we no longer need legacy diffs.
    """

    def order_mapping(data: dict[str, Any], priority: list[str]) -> OrderedDict[str, Any]:
        ordered: OrderedDict[str, Any] = OrderedDict()
        for key in priority:
            if key in data:
                ordered[key] = data[key]
        for key, value in data.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    paths = openapi_schema.get("paths")
    if isinstance(paths, dict):
        openapi_schema["paths"] = order_mapping(paths, LEGACY_PATH_ORDER)
        for path, path_item in openapi_schema["paths"].items():
            if not isinstance(path_item, dict):
                continue
            ordered_path_item = OrderedDict()
            for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                if method in path_item:
                    ordered_path_item[method] = order_mapping(path_item[method], LEGACY_OPERATION_KEYS)
            for key, value in path_item.items():
                if key not in ordered_path_item:
                    if isinstance(value, dict) and key.lower() in {
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    }:
                        ordered_path_item[key] = order_mapping(value, LEGACY_OPERATION_KEYS)
                    else:
                        ordered_path_item[key] = value
            openapi_schema["paths"][path] = ordered_path_item

    components = openapi_schema.setdefault("components", {})
    schemas = components.get("schemas")
    if isinstance(schemas, dict):
        components["schemas"] = order_mapping(schemas, LEGACY_SCHEMA_ORDER)
    responses = components.get("responses")
    if isinstance(responses, dict):
        components["responses"] = order_mapping(responses, LEGACY_RESPONSE_ORDER)

    if LEGACY_TAGS:
        openapi_schema["tags"] = LEGACY_TAGS

    if LEGACY_TAG_GROUPS:
        openapi_schema["x-tagGroups"] = LEGACY_TAG_GROUPS

    if LEGACY_SECURITY:
        openapi_schema["security"] = LEGACY_SECURITY

    return openapi_schema


def _extract_duplicate_union_types(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Extract duplicate union types to shared schema references.

    Stainless generates type names from union types based on their context, which can cause
    duplicate names when the same union appears in different places. This function extracts
    these duplicate unions to shared schema definitions and replaces inline definitions with
    references to them.

    According to Stainless docs, when duplicate types are detected, they should be extracted
    to the same ref and declared as a model. This ensures Stainless generates consistent
    type names regardless of where the union is referenced.

    Fixes: https://www.stainless.com/docs/reference/diagnostics#Python/DuplicateDeclaration
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    # Extract the Output union type (used in OpenAIResponseObjectWithInput-Output and ListOpenAIResponseInputItem)
    output_union_schema_name = "OpenAIResponseMessageOutputUnion"
    output_union_title = None

    # Get the union type from OpenAIResponseObjectWithInput.input.items.anyOf
    if "OpenAIResponseObjectWithInput" in schemas:
        schema = schemas["OpenAIResponseObjectWithInput"]
        if isinstance(schema, dict) and "properties" in schema:
            input_prop = schema["properties"].get("input")
            if isinstance(input_prop, dict) and "items" in input_prop:
                items = input_prop["items"]
                if isinstance(items, dict) and "anyOf" in items:
                    # Extract the union schema with deep copy
                    output_union_schema = copy.deepcopy(items["anyOf"])
                    output_union_title = items.get("title", "OpenAIResponseMessageOutputUnion")

                    # Collect all refs from the oneOf to detect duplicates
                    refs_in_oneof = set()
                    for item in output_union_schema:
                        if isinstance(item, dict) and "oneOf" in item:
                            oneof = item["oneOf"]
                            if isinstance(oneof, list):
                                for variant in oneof:
                                    if isinstance(variant, dict) and "$ref" in variant:
                                        refs_in_oneof.add(variant["$ref"])
                            item["x-stainless-naming"] = "OpenAIResponseMessageOutputOneOf"

                    # Remove duplicate refs from anyOf that are already in oneOf
                    deduplicated_schema = []
                    for item in output_union_schema:
                        if isinstance(item, dict) and "$ref" in item:
                            if item["$ref"] not in refs_in_oneof:
                                deduplicated_schema.append(item)
                        else:
                            deduplicated_schema.append(item)
                    output_union_schema = deduplicated_schema

                    # Create the shared schema with x-stainless-naming to ensure consistent naming
                    if output_union_schema_name not in schemas:
                        schemas[output_union_schema_name] = {
                            "anyOf": output_union_schema,
                            "title": output_union_title,
                            "x-stainless-naming": output_union_schema_name,
                        }
                    # Replace with reference
                    input_prop["items"] = {"$ref": f"#/components/schemas/{output_union_schema_name}"}

    # Replace the same union in ListOpenAIResponseInputItem.data.items.anyOf
    if "ListOpenAIResponseInputItem" in schemas and output_union_schema_name in schemas:
        schema = schemas["ListOpenAIResponseInputItem"]
        if isinstance(schema, dict) and "properties" in schema:
            data_prop = schema["properties"].get("data")
            if isinstance(data_prop, dict) and "items" in data_prop:
                items = data_prop["items"]
                if isinstance(items, dict) and "anyOf" in items:
                    # Replace with reference
                    data_prop["items"] = {"$ref": f"#/components/schemas/{output_union_schema_name}"}

    # Replace the same union in OpenAICompactedResponse.output.items.anyOf
    if "OpenAICompactedResponse" in schemas and output_union_schema_name in schemas:
        schema = schemas["OpenAICompactedResponse"]
        if isinstance(schema, dict) and "properties" in schema:
            output_prop = schema["properties"].get("output")
            if isinstance(output_prop, dict) and "items" in output_prop:
                items = output_prop["items"]
                if isinstance(items, dict) and "anyOf" in items:
                    # Replace with reference
                    output_prop["items"] = {"$ref": f"#/components/schemas/{output_union_schema_name}"}

    # Extract the response output union type (used in OpenAIResponseObject.output and
    # OpenAIResponseObjectWithInput.output). Both schemas have identical inline oneOf unions
    # which causes Stainless to generate duplicate type names like
    # "ResponseListResponseInputOpenAIResponseMessageOutput".
    response_output_union_name = "OpenAIResponseOutputItem"

    # Extract from OpenAIResponseObject.output first to create the shared schema
    if "OpenAIResponseObject" in schemas:
        schema = schemas["OpenAIResponseObject"]
        if isinstance(schema, dict) and "properties" in schema:
            output_prop = schema["properties"].get("output")
            if isinstance(output_prop, dict) and "items" in output_prop:
                items = output_prop["items"]
                if isinstance(items, dict) and "oneOf" in items:
                    if response_output_union_name not in schemas:
                        schemas[response_output_union_name] = copy.deepcopy(items)
                        schemas[response_output_union_name]["x-stainless-naming"] = response_output_union_name
                    output_prop["items"] = {"$ref": f"#/components/schemas/{response_output_union_name}"}

    # Replace the same union in OpenAIResponseObjectWithInput.output
    if "OpenAIResponseObjectWithInput" in schemas and response_output_union_name in schemas:
        schema = schemas["OpenAIResponseObjectWithInput"]
        if isinstance(schema, dict) and "properties" in schema:
            output_prop = schema["properties"].get("output")
            if isinstance(output_prop, dict) and "items" in output_prop:
                items = output_prop["items"]
                if isinstance(items, dict) and "oneOf" in items:
                    output_prop["items"] = {"$ref": f"#/components/schemas/{response_output_union_name}"}

    # Extract the Input union type (used in _responses_Request.input.anyOf[1].items.anyOf)
    input_union_schema_name = "OpenAIResponseMessageInputUnion"

    if "_responses_Request" in schemas:
        schema = schemas["_responses_Request"]
        if isinstance(schema, dict) and "properties" in schema:
            input_prop = schema["properties"].get("input")
            if isinstance(input_prop, dict) and "anyOf" in input_prop:
                any_of = input_prop["anyOf"]
                if isinstance(any_of, list) and len(any_of) > 1:
                    # Check the second item (index 1) which should be the array type
                    second_item = any_of[1]
                    if isinstance(second_item, dict) and "items" in second_item:
                        items = second_item["items"]
                        if isinstance(items, dict) and "anyOf" in items:
                            # Extract the union schema with deep copy
                            input_union_schema = copy.deepcopy(items["anyOf"])
                            input_union_title = items.get("title", "OpenAIResponseMessageInputUnion")

                            # Collect all refs from the oneOf to detect duplicates
                            refs_in_oneof = set()
                            for item in input_union_schema:
                                if isinstance(item, dict) and "oneOf" in item:
                                    oneof = item["oneOf"]
                                    if isinstance(oneof, list):
                                        for variant in oneof:
                                            if isinstance(variant, dict) and "$ref" in variant:
                                                refs_in_oneof.add(variant["$ref"])
                                    item["x-stainless-naming"] = "OpenAIResponseMessageInputOneOf"

                            # Remove duplicate refs from anyOf that are already in oneOf
                            deduplicated_schema = []
                            for item in input_union_schema:
                                if isinstance(item, dict) and "$ref" in item:
                                    if item["$ref"] not in refs_in_oneof:
                                        deduplicated_schema.append(item)
                                else:
                                    deduplicated_schema.append(item)
                            input_union_schema = deduplicated_schema

                            # Create the shared schema with x-stainless-naming to ensure consistent naming
                            if input_union_schema_name not in schemas:
                                schemas[input_union_schema_name] = {
                                    "anyOf": input_union_schema,
                                    "title": input_union_title,
                                    "x-stainless-naming": input_union_schema_name,
                                }
                            # Replace with reference
                            second_item["items"] = {"$ref": f"#/components/schemas/{input_union_schema_name}"}

    return openapi_schema


def validate_openapi_schema(schema: dict[str, Any], schema_name: str = "OpenAPI schema") -> bool:
    """
    Validate an OpenAPI schema using openapi-spec-validator.

    Args:
        schema: The OpenAPI schema dictionary to validate
        schema_name: Name of the schema for error reporting

    Returns:
        True if valid, False otherwise

    Raises:
        OpenAPIValidationError: If validation fails
    """
    try:
        validate_spec(schema)
        print(f"{schema_name} is valid")
        return True
    except OpenAPISpecValidatorError as e:
        print(f"{schema_name} validation failed: {e}")
        return False
    except Exception as e:
        print(f"{schema_name} validation error: {e}")
        return False

#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Process OpenAPI spec to extract tag hierarchy and prepare for SDK generation.

This script operates on a Stainless-independent OpenAPI spec (the output of
merge_stainless_config.py) and performs the following transformations:

1. Extracts tag hierarchies from multi-tag endpoints (e.g., [chat, completions])
2. Reduces endpoint tags to only the leaf tag (for openapi-generator class assignment)
3. Creates dummy endpoints for non-leaf tags (so parent API classes are generated)
4. Converts oneOf-const patterns to proper enums (openapi-generator compatibility)
5. Removes fields with default values from required lists
6. Marks simple list endpoints for response unwrapping

The hierarchy data is written to a separate file for use by patch_hierarchy.py
in the post-generation step.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import ruamel.yaml as yaml
except ImportError:
    print("Error: ruamel.yaml is required. Install with: uv pip install ruamel.yaml", file=sys.stderr)
    sys.exit(1)

HTTP_METHODS = ("get", "post", "put", "delete", "patch", "options", "head", "trace")


def build_hierarchy_from_tags(tags: list[str], hierarchy: dict) -> None:
    """Build nested hierarchy dict from an ordered tag list.

    For tags ['chat', 'completions'], produces {'chat': {'completions': {}}}.
    """
    current = hierarchy
    for tag in tags:
        if tag not in current:
            current[tag] = {}
        current = current[tag]


def get_children_from_hierarchy(tag: str, hierarchy: dict) -> list[str]:
    """Return sorted direct children of a tag in the hierarchy."""
    return sorted(hierarchy.get(tag, {}).keys())


def convert_oneof_const_to_enum(schema: dict) -> dict:
    """Convert oneOf with const values to a proper enum schema.

    OpenAPI Generator doesn't handle oneOf with const values well, generating
    multiple identical validators. This converts them to proper enum schemas.
    """
    if not isinstance(schema, dict) or "oneOf" not in schema:
        return schema

    one_of = schema["oneOf"]
    if not isinstance(one_of, list):
        return schema

    if not all(isinstance(item, dict) and "const" in item for item in one_of):
        return schema

    enum_values = [item["const"] for item in one_of]
    schema_type = one_of[0].get("type", "string")

    new_schema = {"type": schema_type, "enum": enum_values}
    for key in schema:
        if key not in ("oneOf", "type", "enum"):
            new_schema[key] = schema[key]

    return new_schema


def fix_oneof_const_schemas(obj: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert oneOf-const patterns to enums throughout the spec."""

    def _fix(node: Any) -> Any:
        if isinstance(node, dict):
            if "oneOf" in node:
                node = convert_oneof_const_to_enum(node)
            return {k: _fix(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [_fix(item) for item in node]
        return node

    return _fix(obj)


def remove_defaults_from_required(spec: dict[str, Any]) -> int:
    """Remove fields with default values from required lists.

    Returns the number of schemas modified.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    modified_count = 0

    for _, schema in schemas.items():
        if not isinstance(schema, dict):
            continue
        if "required" not in schema or "properties" not in schema:
            continue

        fields_with_defaults = [
            name
            for name, field_schema in schema["properties"].items()
            if isinstance(field_schema, dict) and "default" in field_schema
        ]

        if not fields_with_defaults:
            continue

        original_len = len(schema["required"])
        schema["required"] = [f for f in schema["required"] if f not in fields_with_defaults]

        if len(schema["required"]) < original_len:
            modified_count += 1

    return modified_count


def mark_unwrappable_list_responses(spec: dict[str, Any]) -> int:
    """Add x-unwrap-list-response extension for simple (non-paginated) list endpoints.

    Returns the number of endpoints marked.
    """
    schemas = spec.get("components", {}).get("schemas", {})
    pagination_fields = {"has_more", "url", "first_id", "last_id", "next_page_token", "total"}
    count = 0

    for _, methods in spec.get("paths", {}).items():
        for method, operation in methods.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue

            response_200 = operation.get("responses", {}).get("200", {})
            json_content = response_200.get("content", {}).get("application/json", {})
            schema_ref = json_content.get("schema", {})

            ref = schema_ref.get("$ref", "")
            if not ref:
                continue

            schema_name = ref.split("/")[-1]
            if not ("List" in schema_name and schema_name.endswith("Response")):
                continue

            schema_def = schemas.get(schema_name, {})
            props = schema_def.get("properties", {})
            if "data" in props and not any(f in props for f in pagination_fields):
                operation["x-unwrap-list-response"] = True
                count += 1

    return count


def mark_streaming_operations(spec: dict[str, Any]) -> int:
    """Add x-streaming vendor extensions for operations with text/event-stream responses.

    Scans all operations for text/event-stream response content types and stamps:
    - x-streaming: true
    - x-streaming-type: <schema name> (the SSE response schema)

    The streaming type's discriminator (if any) is read at runtime from the
    generated model class, so we don't need to duplicate it here.

    Returns the number of operations marked.
    """
    count = 0

    for _, path_item in spec.get("paths", {}).items():
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue

            response_200 = operation.get("responses", {}).get("200", {})
            sse_content = response_200.get("content", {}).get("text/event-stream")
            if not sse_content:
                continue

            schema_ref = sse_content.get("schema", {}).get("$ref", "")
            if not schema_ref:
                continue

            stream_type = schema_ref.split("/")[-1]
            operation["x-streaming"] = True
            operation["x-streaming-type"] = stream_type
            count += 1

    return count


def process_openapi(input_file: str, output_file: str, hierarchy_file: str) -> None:
    """Process OpenAPI spec to extract hierarchy and prepare for code generation."""
    yaml_handler = yaml.YAML()
    yaml_handler.preserve_quotes = True
    yaml_handler.default_flow_style = False

    print(f"Loading OpenAPI spec from: {input_file}")
    with open(input_file) as f:
        spec = yaml_handler.load(f)

    # --- Phase 1: Extract tag hierarchy and reduce to leaf tags ---
    api_hierarchy = {}
    all_tags = set()
    tags_with_endpoints = set()
    endpoint_count = 0

    for _, path_item in spec.get("paths", {}).items():
        for method in HTTP_METHODS:
            if method not in path_item:
                continue

            operation = path_item[method]
            endpoint_count += 1

            tags = operation.get("tags", [])
            if not tags:
                continue

            build_hierarchy_from_tags(tags, api_hierarchy)
            all_tags.update(tags)

            leaf_tag = tags[-1]
            tags_with_endpoints.add(leaf_tag)
            operation["tags"] = [leaf_tag]

    # --- Phase 2: Annotate endpoints with child tag info ---
    for _, path_item in spec.get("paths", {}).items():
        for method in HTTP_METHODS:
            if method not in path_item:
                continue
            operation = path_item[method]
            tags = operation.get("tags", [])
            if tags:
                children = get_children_from_hierarchy(tags[0], api_hierarchy)
                if children:
                    operation["x-child-tags"] = children

    # --- Phase 3: Create dummy endpoints for non-leaf (parent-only) tags ---
    tags_without_endpoints = all_tags - tags_with_endpoints

    for tag in sorted(tags_without_endpoints):
        dummy_path = f"/dummy/{tag.lower().replace(' ', '-').replace('_', '-')}"
        children = get_children_from_hierarchy(tag, api_hierarchy)

        operation_spec = {
            "summary": f"Placeholder for {tag} resource group",
            "description": f"Placeholder endpoint so openapi-generator creates an API class for the {tag} resource group.",
            "operationId": f"dummy_{tag.replace(' ', '_').replace('-', '_')}",
            "tags": [tag],
            "responses": {"200": {"description": "Success"}},
            "x-operation-name": "dummy",
        }
        if children:
            operation_spec["x-child-tags"] = children

        spec["paths"][dummy_path] = {"get": operation_spec}

    if tags_without_endpoints:
        print(
            f"  Created {len(tags_without_endpoints)} dummy endpoints for parent tags: {sorted(tags_without_endpoints)}"
        )

    # --- Phase 4: Write hierarchy file for post-generation patching ---
    hierarchy_data = {
        "api_hierarchy": api_hierarchy,
        "all_tags": sorted(all_tags),
        "tags_with_endpoints": sorted(tags_with_endpoints),
        "tags_without_endpoints": sorted(tags_without_endpoints),
    }

    # Propagate proxy methods from enriched spec to hierarchy file, resolving
    # nesting paths to the actual tag names used in the hierarchy.
    raw_proxy_methods = spec.get("x-proxy-methods", [])
    if raw_proxy_methods:
        resolved_proxy_methods = []
        for pm in raw_proxy_methods:
            parent_path = pm["parent_nesting_path"]
            child_path = pm["child_nesting_path"]
            method_name = pm["method_name"]

            # The parent tag is the last element of the parent nesting path
            parent_tag = parent_path[-1].lower()

            # The child tag is the key in api_hierarchy under the parent.
            # Look up the child resource name in the hierarchy to find its
            # resolved tag name (which may include a parent prefix for collisions).
            child_resource = child_path[-1].lower()
            parent_children = api_hierarchy.get(parent_tag, {})
            # Find the child tag — it's either the simple name or a prefixed version
            child_tag = None
            for key in parent_children:
                if key == child_resource or key.endswith(f"_{child_resource}"):
                    child_tag = key
                    break

            if child_tag is None:
                print(f"  Warning: Could not resolve child tag for {child_resource} under {parent_tag}")
                continue

            resolved_proxy_methods.append(
                {
                    "parent_tag": parent_tag,
                    "child_tag": child_tag,
                    "method_name": method_name,
                }
            )

        if resolved_proxy_methods:
            hierarchy_data["proxy_methods"] = resolved_proxy_methods
            print(f"  Resolved {len(resolved_proxy_methods)} proxy method(s) for hierarchy file")

    with open(hierarchy_file, "w") as f:
        yaml_handler.dump(hierarchy_data, f)

    # --- Phase 5: Schema fixes for openapi-generator compatibility ---
    spec = fix_oneof_const_schemas(spec)

    modified = remove_defaults_from_required(spec)
    if modified:
        print(f"  Removed default-valued fields from required in {modified} schemas")

    unwrapped = mark_unwrappable_list_responses(spec)
    if unwrapped:
        print(f"  Marked {unwrapped} endpoints for list response unwrapping")

    streaming = mark_streaming_operations(spec)
    if streaming:
        print(f"  Marked {streaming} endpoints with streaming type metadata")

    # --- Write output ---
    with open(output_file, "w") as f:
        yaml_handler.dump(spec, f)

    # --- Summary ---
    print(f"  Processed {endpoint_count} endpoints, {len(all_tags)} tags")
    print(f"  Output: {output_file}")
    print(f"  Hierarchy: {hierarchy_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenAPI spec to extract tag hierarchy and prepare for SDK generation"
    )
    parser.add_argument(
        "--source",
        "-s",
        default="openapi.yml",
        help="Source OpenAPI YAML file (default: openapi.yml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="openapi-hierarchical.yml",
        help="Output OpenAPI YAML file (default: openapi-hierarchical.yml)",
    )
    parser.add_argument(
        "--hierarchy",
        "-H",
        default="api-hierarchy.yml",
        help="API hierarchy output file (default: api-hierarchy.yml)",
    )

    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    try:
        process_openapi(args.source, args.output, args.hierarchy)
    except Exception as e:
        print(f"Error processing OpenAPI spec: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

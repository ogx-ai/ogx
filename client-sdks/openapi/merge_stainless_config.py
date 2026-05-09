#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Merge Stainless configuration into OpenAPI spec.

This script takes:
1. client-sdks/stainless/openapi.yml - Base OpenAPI specification
2. client-sdks/stainless/config.yml - Stainless resource configuration
3. (Optional) patch file - Additional modifications to apply

And produces:
- client-sdks/openapi/openapi.yml - OpenAPI spec enriched with x-operation-name
  and tags for use with openapi-generator

Patch file format:
  operations:
    - path: "components.schemas.OpenAIResponseInput.discriminator"
      action: "set"
      value:
        propertyName: type
        mapping:
          key1: value1

    - path: "components.schemas.OpenAICompletionChoice.required"
      action: "remove_item"
      value: "finish_reason"

    - path: "components.schemas.SomeSchema.properties.field"
      action: "delete"
"""

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import ruamel.yaml as yaml
except ImportError:
    print("Error: ruamel.yaml is required. Install with: pip install ruamel.yaml", file=sys.stderr)
    sys.exit(1)


def parse_endpoint(endpoint_str: str) -> tuple[str, str]:
    """
    Parse an endpoint string like 'post /v1/chat/completions' into (method, path).

    Args:
        endpoint_str: String like "post /v1/chat/completions" or just "/v1/chat/completions"

    Returns:
        Tuple of (http_method, path)
    """
    parts = endpoint_str.strip().split(maxsplit=1)
    if len(parts) == 2:
        return parts[0].lower(), parts[1]
    else:
        # If no method specified, assume it's just a path
        return None, parts[0]


def extract_resources(stainless_config: dict[str, Any]) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    """
    Extract resource->method->endpoint mappings from Stainless config.

    Returns:
        Tuple of (endpoint_map, collision_set, proxy_methods)
        - endpoint_map: dict mapping (http_method, path) -> resource_info
        - collision_set: set of resource names that appear in multiple places
        - proxy_methods: list of proxy method entries for parent resources that share
          an endpoint with a subresource. Each entry is a dict with:
          - parent_nesting_path: the nesting path of the parent resource
          - child_nesting_path: the nesting path of the child (subresource)
          - method_name: the method name (e.g., "list")
    """
    resources = stainless_config.get("resources", {})
    endpoint_map = {}
    resource_name_counts = {}  # Count how many times each resource name appears
    proxy_methods = []

    def _record_endpoint(http_method: str, path: str, method_name: str, current_path: list[str], resource_name: str):
        """Record an endpoint mapping, detecting overwrites to capture proxy info."""
        key = (http_method, path)
        existing = endpoint_map.get(key)

        if existing is not None and existing["operation_name"] == method_name:
            # Same endpoint, same method name, different nesting paths.
            # The deeper (subresource) path keeps the endpoint; the shallower
            # (parent) path gets a proxy method that delegates to the child.
            if len(current_path) > len(existing["nesting_path"]):
                # New entry is deeper — it wins the endpoint, parent gets proxy
                proxy_methods.append(
                    {
                        "parent_nesting_path": list(existing["nesting_path"]),
                        "child_nesting_path": list(current_path),
                        "method_name": method_name,
                    }
                )
                endpoint_map[key] = {
                    "operation_name": method_name,
                    "nesting_path": current_path,
                    "resource_name": resource_name,
                }
            else:
                # Existing entry is deeper or same depth — keep existing, new gets proxy
                proxy_methods.append(
                    {
                        "parent_nesting_path": list(current_path),
                        "child_nesting_path": list(existing["nesting_path"]),
                        "method_name": method_name,
                    }
                )
        elif existing is not None:
            # Same endpoint, different method name, same or different resource.
            # Keep the first (canonical) name; record the new name as an alias.
            aliases = existing.setdefault("aliases", [])
            aliases.append(method_name)
        else:
            endpoint_map[key] = {
                "operation_name": method_name,
                "nesting_path": current_path,
                "resource_name": resource_name,
            }

    def process_resource(resource_name: str, resource_data: Any, parent_path: list[str] = None):
        """Recursively process resources and subresources."""
        if parent_path is None:
            parent_path = []

        current_path = parent_path + [resource_name] if resource_name != "$shared" else parent_path

        if not isinstance(resource_data, dict):
            return

        # Track resource name occurrences (skip $shared)
        if resource_name != "$shared":
            resource_name_counts[resource_name] = resource_name_counts.get(resource_name, 0) + 1

        # Process methods
        methods = resource_data.get("methods", {})
        for method_name, method_config in methods.items():
            if isinstance(method_config, dict):
                # Extract endpoint - could be direct or nested
                endpoint = method_config.get("endpoint")
                if endpoint:
                    http_method, path = parse_endpoint(endpoint)
                    if http_method and path:
                        _record_endpoint(http_method, path, method_name, current_path, resource_name)
            elif isinstance(method_config, str):
                # Simple string endpoint like "get /v1/tools"
                http_method, path = parse_endpoint(method_config)
                if http_method and path:
                    _record_endpoint(http_method, path, method_name, current_path, resource_name)

        # Process subresources recursively
        subresources = resource_data.get("subresources", {})
        for sub_name, sub_data in subresources.items():
            process_resource(sub_name, sub_data, current_path)

    # Process all top-level resources
    for resource_name, resource_data in resources.items():
        process_resource(resource_name, resource_data)

    # Find collisions - resource names that appear more than once
    collision_set = {name for name, count in resource_name_counts.items() if count > 1}

    return endpoint_map, collision_set, proxy_methods


def enrich_openapi_spec(
    openapi_spec: dict[str, Any],
    endpoint_map: dict[tuple[str, str], dict[str, Any]],
    collision_set: set[str],
    proxy_methods: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Enrich OpenAPI spec with x-operation-name and tags from endpoint_map.

    Args:
        openapi_spec: The base OpenAPI specification
        endpoint_map: Map of (method, path) -> resource info
        collision_set: Set of resource names that appear in multiple places in the hierarchy
        proxy_methods: List of proxy method entries for shared endpoints

    Returns:
        Enriched OpenAPI specification
    """
    paths = openapi_spec.get("paths", {})

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            if method not in path_item:
                continue

            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Normalize any existing tags to lowercase to avoid duplicates
            if "tags" in operation and isinstance(operation["tags"], list):
                operation["tags"] = [tag.lower() for tag in operation["tags"]]

            # Look up this endpoint in our map
            key = (method, path)
            resource_info = endpoint_map.get(key)

            if resource_info:
                # Add x-operation-name
                operation["x-operation-name"] = resource_info["operation_name"]

                # Add aliases for operations that have multiple method names
                if resource_info.get("aliases"):
                    operation["x-operation-aliases"] = [
                        {"alias": alias, "target": resource_info["operation_name"]}
                        for alias in resource_info["aliases"]
                    ]

                # Build tags based on the resource hierarchy from Stainless
                nesting_path = resource_info["nesting_path"]
                if nesting_path:
                    tags = []

                    # Add a tag for each level in the hierarchy
                    for i, resource_name in enumerate(nesting_path):
                        if i == 0:
                            # Top-level: always use simple name
                            tags.append(resource_name.lower())
                        else:
                            # For nested levels: check if there's a collision
                            # - If the resource name appears in multiple places, use hierarchical path
                            # - Otherwise, use just the simple resource name
                            if resource_name in collision_set:
                                # Collision: use hierarchical path up to this level
                                # e.g., for [chat, completions], completions is a collision -> chat_completions
                                hierarchical_tag = "_".join(nesting_path[: i + 1]).lower()
                                tags.append(hierarchical_tag)
                            else:
                                # No collision: use simple name
                                # e.g., for [conversations, items], items is unique -> items
                                tags.append(resource_name.lower())

                    operation["tags"] = tags

    # Store proxy methods as a top-level extension for downstream tools
    if proxy_methods:
        openapi_spec["x-proxy-methods"] = proxy_methods

    return openapi_spec


def get_nested_value(obj: Any, path: str) -> tuple[Any, Any, str]:
    """
    Navigate to a nested path in an object and return (parent, current_value, last_key).

    Args:
        obj: The root object to navigate
        path: Dot-separated path like "components.schemas.MySchema.properties"

    Returns:
        Tuple of (parent_object, current_value, last_key)
    """
    parts = path.split(".")
    current = obj

    # Navigate to parent
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            raise ValueError(f"Cannot navigate through non-dict at {part}")

    last_key = parts[-1]
    parent = current
    current_value = current.get(last_key) if isinstance(current, dict) else None

    return parent, current_value, last_key


def apply_patches(openapi_spec: dict[str, Any], patch_config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply patch operations to the OpenAPI spec.

    Args:
        openapi_spec: The OpenAPI specification to patch
        patch_config: Patch configuration with operations

    Returns:
        Patched OpenAPI specification
    """
    operations = patch_config.get("operations", [])

    for op in operations:
        path = op.get("path")
        action = op.get("action")
        value = op.get("value")

        if not path or not action:
            print(f"Warning: Skipping invalid operation: {op}")
            continue

        try:
            parent, current_value, last_key = get_nested_value(openapi_spec, path)

            if action == "set":
                # Set or overwrite a value
                if isinstance(parent, dict):
                    parent[last_key] = value
                    print(f"  Set {path}")
                else:
                    print(f"  Warning: Cannot set {path}: parent is not a dict")

            elif action == "delete":
                # Delete a key
                if isinstance(parent, dict) and last_key in parent:
                    del parent[last_key]
                    print(f"  Deleted {path}")
                else:
                    print(f"  Warning: Cannot delete {path}: key not found")

            elif action == "remove_item":
                # Remove an item from a list
                if isinstance(current_value, list):
                    if value in current_value:
                        current_value.remove(value)
                        print(f"  Removed '{value}' from {path}")
                    else:
                        print(f"  Warning: '{value}' not found in {path}")
                else:
                    print(f"  Warning: Cannot remove_item from {path}: not a list")

            elif action == "append":
                # Append to a list
                if isinstance(current_value, list):
                    if value not in current_value:
                        current_value.append(value)
                        print(f"  Appended '{value}' to {path}")
                    else:
                        print(f"  Skipped: '{value}' already in {path}")
                elif current_value is None:
                    # Create list if it doesn't exist
                    parent[last_key] = [value]
                    print(f"  Created {path} with ['{value}']")
                else:
                    print(f"  Warning: Cannot append to {path}: not a list")

            elif action == "merge":
                # Merge a dict into existing value
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                    print(f"  Merged into {path}")
                elif current_value is None:
                    parent[last_key] = value
                    print(f"  Created {path} with merged value")
                else:
                    print(f"  Warning: Cannot merge into {path}: not a dict")

            else:
                print(f"  Warning: Unknown action: {action}")

        except Exception as e:
            print(f"  Error applying operation to {path}: {e}")

    return openapi_spec


def main():
    parser = argparse.ArgumentParser(description="Merge Stainless configuration into OpenAPI spec")
    parser.add_argument(
        "--openapi",
        default="client-sdks/stainless/openapi.yml",
        help="Path to base OpenAPI specification (default: client-sdks/stainless/openapi.yml)",
    )
    parser.add_argument(
        "--stainless",
        default="client-sdks/stainless/config.yml",
        help="Path to Stainless configuration (default: client-sdks/stainless/config.yml)",
    )
    parser.add_argument("--patch", help="Optional patch file with additional modifications to apply")
    parser.add_argument(
        "--output",
        default="client-sdks/openapi/openapi.yml",
        help="Output path for enriched spec (default: client-sdks/openapi/openapi.yml)",
    )

    args = parser.parse_args()

    # Load YAML files
    yaml_loader = yaml.YAML()
    yaml_loader.preserve_quotes = True
    yaml_loader.default_flow_style = False

    openapi_path = Path(args.openapi)
    stainless_path = Path(args.stainless)
    output_path = Path(args.output)

    if not openapi_path.exists():
        print(f"Error: OpenAPI spec not found: {openapi_path}", file=sys.stderr)
        sys.exit(1)

    if not stainless_path.exists():
        print(f"Error: Stainless config not found: {stainless_path}", file=sys.stderr)
        sys.exit(1)

    with open(openapi_path) as f:
        openapi_spec = yaml_loader.load(f)

    with open(stainless_path) as f:
        stainless_config = yaml_loader.load(f)

    # Extract resource mappings
    endpoint_map, collision_set, proxy_methods = extract_resources(stainless_config)

    if proxy_methods:
        print(f"Detected {len(proxy_methods)} proxy method(s) for shared endpoints:")
        for pm in proxy_methods:
            print(
                f"  {'.'.join(pm['parent_nesting_path'])}.{pm['method_name']}() -> {'.'.join(pm['child_nesting_path'])}.{pm['method_name']}()"
            )

    # Enrich the OpenAPI spec
    enriched_spec = enrich_openapi_spec(openapi_spec, endpoint_map, collision_set, proxy_methods)

    # Apply patches if provided
    if args.patch:
        patch_path = Path(args.patch)
        if not patch_path.exists():
            print(f"Error: Patch file not found: {patch_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Applying patches from {patch_path}:")
        with open(patch_path) as f:
            patch_config = yaml_loader.load(f)

        enriched_spec = apply_patches(enriched_spec, patch_config)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml_loader.dump(enriched_spec, f)

    print(f"Generated {output_path} with {len(endpoint_map)} enriched endpoints")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Main entry point for the FastAPI OpenAPI generator.
"""

import copy
from pathlib import Path
from typing import Any

import yaml
from fastapi.openapi.utils import get_openapi

from . import app, code_samples, schema_collection, schema_filtering, schema_transforms, state


def generate_openapi_spec(output_dir: str) -> dict[str, Any]:
    """
    Generate OpenAPI specification using FastAPI's built-in method.

    Args:
        output_dir: Directory to save the generated files

    Returns:
        The generated OpenAPI specification as a dictionary
    """
    state.reset_generator_state()
    # Create the FastAPI app
    fastapi_app = app.create_ogx_app()

    # Generate the OpenAPI schema
    openapi_schema = get_openapi(
        title=fastapi_app.title,
        version=fastapi_app.version,
        description=fastapi_app.description,
        routes=fastapi_app.routes,
        servers=fastapi_app.servers,
    )

    # Set OpenAPI version to 3.1.0
    openapi_schema["openapi"] = "3.1.0"

    # Add standard error responses
    openapi_schema = schema_transforms._add_error_responses(openapi_schema)

    # Ensure all @json_schema_type decorated models are included
    openapi_schema = schema_collection._ensure_json_schema_types_included(openapi_schema)

    # Fix $ref references to point to components/schemas instead of $defs
    openapi_schema = schema_transforms._fix_ref_references(openapi_schema)

    # Fix path parameter resolution issues
    openapi_schema = schema_transforms._fix_path_parameters(openapi_schema)

    # Eliminate $defs section entirely for oasdiff compatibility
    openapi_schema = schema_transforms._eliminate_defs_section(openapi_schema)

    # Clean descriptions in schema definitions by removing docstring metadata
    openapi_schema = schema_transforms._clean_schema_descriptions(openapi_schema)
    openapi_schema = schema_transforms._normalize_empty_responses(openapi_schema)

    # Remove query parameters from POST/PUT/PATCH endpoints that have a request body
    # FastAPI sometimes infers parameters as query params even when they should be in the request body
    openapi_schema = schema_transforms._remove_query_params_from_body_endpoints(openapi_schema)

    # Add x-ogx-extra-body-params extension for ExtraBodyField parameters
    openapi_schema = schema_transforms._add_extra_body_params_extension(openapi_schema)

    # Remove request bodies from GET endpoints (GET requests should never have request bodies)
    # This must run AFTER _add_extra_body_params_extension to ensure any request bodies
    # that FastAPI incorrectly added to GET endpoints are removed
    openapi_schema = schema_transforms._remove_request_bodies_from_get_endpoints(openapi_schema)

    # Keep 'type: object' on schemas with properties (OpenAI uses it on 766/772),
    # but strip it from the 6 specific schemas where OpenAI omits it.
    openapi_schema = schema_transforms._remove_type_object_from_openai_schemas(openapi_schema)

    # Extract duplicate union types to shared schema references
    openapi_schema = schema_transforms._extract_duplicate_union_types(openapi_schema)

    # Add SDK code samples to API endpoints
    openapi_schema = code_samples._add_openai_code_samples(openapi_schema)
    openapi_schema = code_samples._add_anthropic_code_samples(openapi_schema)

    # Add Google GenAI SDK code samples to Interactions API endpoints
    openapi_schema = code_samples._add_google_code_samples(openapi_schema)

    # Split into stable (v1 only), experimental (v1alpha + v1beta), deprecated, and combined (stainless) specs
    # Each spec needs its own deep copy of the full schema to avoid cross-contamination
    stable_schema = schema_filtering._filter_schema_by_version(
        copy.deepcopy(openapi_schema), stable_only=True, exclude_deprecated=True
    )
    experimental_schema = schema_filtering._filter_schema_by_version(
        copy.deepcopy(openapi_schema), stable_only=False, exclude_deprecated=True
    )
    deprecated_schema = schema_filtering._filter_deprecated_schema(copy.deepcopy(openapi_schema))
    combined_schema = schema_filtering._filter_combined_schema(copy.deepcopy(openapi_schema))

    # Apply duplicate union extraction to combined schema (used by Stainless)
    combined_schema = schema_transforms._extract_duplicate_union_types(combined_schema)
    combined_schema = schema_transforms._dedupe_create_response_request_input_union_for_stainless(combined_schema)

    base_description = (
        "This is the specification of the OGX that provides\n"
        "                    a set of endpoints and their corresponding interfaces that are\n"
        "    tailored to\n"
        "                    best leverage Llama Models."
    )

    schema_configs = [
        (
            stable_schema,
            "OGX Specification",
            "**✅ STABLE**: Production-ready APIs with backward compatibility guarantees.",
        ),
        (
            experimental_schema,
            "OGX Specification - Experimental APIs",
            "**🧪 EXPERIMENTAL**: Pre-release APIs (v1alpha, v1beta) that may change before\n    becoming stable.",
        ),
        (
            deprecated_schema,
            "OGX Specification - Deprecated APIs",
            "**⚠️ DEPRECATED**: Legacy APIs that may be removed in future versions. Use for\n    migration reference only.",
        ),
        (
            combined_schema,
            "OGX Specification - Stable & Experimental APIs",
            "**🔗 COMBINED**: This specification includes both stable production-ready APIs\n    and experimental pre-release APIs. Use stable APIs for production deployments\n    and experimental APIs for testing new features.",
        ),
    ]

    for schema, title, description_suffix in schema_configs:
        if "info" not in schema:
            schema["info"] = {}
        schema["info"].update(
            {
                "title": title,
                "version": "v1",
                "description": f"{base_description}\n\n    {description_suffix}",
            }
        )

    schemas_to_validate = [
        (stable_schema, "Stable schema"),
        (experimental_schema, "Experimental schema"),
        (deprecated_schema, "Deprecated schema"),
        (combined_schema, "Combined (stainless) schema"),
    ]

    for schema, _ in schemas_to_validate:
        schema_transforms._fix_schema_issues(schema)
        schema_transforms._apply_legacy_sorting(schema)

    print("\nValidating generated schemas...")
    failed_schemas = [
        name for schema, name in schemas_to_validate if not schema_transforms.validate_openapi_schema(schema, name)
    ]
    if failed_schemas:
        raise ValueError(f"Invalid schemas: {', '.join(failed_schemas)}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the stable specification
    yaml_path = output_path / "ogx-spec.yaml"
    schema_transforms._write_yaml_file(yaml_path, stable_schema)
    # Post-process the YAML file to remove $defs section and fix references
    with open(yaml_path) as f:
        yaml_content = f.read()

    if "  $defs:" in yaml_content or "#/$defs/" in yaml_content:
        # Use string replacement to fix references directly
        if "#/$defs/" in yaml_content:
            yaml_content = yaml_content.replace("#/$defs/", "#/components/schemas/")

        # Parse the YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Move $defs to components/schemas if it exists
        if "$defs" in yaml_data:
            if "components" not in yaml_data:
                yaml_data["components"] = {}
            if "schemas" not in yaml_data["components"]:
                yaml_data["components"]["schemas"] = {}

            # Move all $defs to components/schemas
            for def_name, def_schema in yaml_data["$defs"].items():
                yaml_data["components"]["schemas"][def_name] = def_schema

            # Remove the $defs section
            del yaml_data["$defs"]

        # Write the modified YAML back
        schema_transforms._write_yaml_file(yaml_path, yaml_data)

    print(f"Generated YAML (stable): {yaml_path}")

    experimental_yaml_path = output_path / "experimental-ogx-spec.yaml"
    schema_transforms._write_yaml_file(experimental_yaml_path, experimental_schema)
    print(f"Generated YAML (experimental): {experimental_yaml_path}")

    deprecated_yaml_path = output_path / "deprecated-ogx-spec.yaml"
    schema_transforms._write_yaml_file(deprecated_yaml_path, deprecated_schema)
    print(f"Generated YAML (deprecated): {deprecated_yaml_path}")

    # Generate combined (stainless) spec
    stainless_yaml_path = output_path / "stainless-ogx-spec.yaml"
    schema_transforms._write_yaml_file(stainless_yaml_path, combined_schema)
    print(f"Generated YAML (stainless/combined): {stainless_yaml_path}")

    return stable_schema


def main():
    """Main entry point for the FastAPI OpenAPI generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI specification using FastAPI")
    parser.add_argument("output_dir", help="Output directory for generated files")

    args = parser.parse_args()

    print("Generating OpenAPI specification using FastAPI...")
    print(f"Output directory: {args.output_dir}")

    try:
        openapi_schema = generate_openapi_spec(output_dir=args.output_dir)

        print("\nOpenAPI specification generated successfully!")
        print(f"Schemas: {len(openapi_schema.get('components', {}).get('schemas', {}))}")
        print(f"Paths: {len(openapi_schema.get('paths', {}))}")
        operation_count = sum(
            1
            for path_info in openapi_schema.get("paths", {}).values()
            for method in ["get", "post", "put", "delete", "patch"]
            if method in path_info
        )
        print(f"Operations: {operation_count}")

    except Exception as e:
        print(f"Error generating OpenAPI specification: {e}")
        raise


if __name__ == "__main__":
    main()

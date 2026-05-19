# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema transforms to align Responses API output with the OpenResponses spec.

Pydantic generates nullable $ref fields in a flat style that differs from
OpenResponses' allOf-wrapping convention.  The functions here rewrite the
affected properties after the main OpenAPI schema has been generated.
"""

from typing import Any


def _wrap_ref_in_allof(prop: dict[str, Any]) -> None:
    """Convert a nullable anyOf property with a $ref to use allOf wrapping.

    Transforms Pydantic style:
        {anyOf: [{$ref: X, title: Y}, {type: null}], description: Z, title: Y}
    To OpenResponses style:
        {anyOf: [{allOf: [{$ref: X}, {description: Z}]}, {type: null}]}
    """
    if "anyOf" not in prop:
        return
    description = prop.pop("description", None)
    prop.pop("title", None)
    for variant in prop["anyOf"]:
        if "$ref" in variant:
            variant.pop("title", None)
            ref = variant.pop("$ref")
            allof_items: list[dict[str, Any]] = [{"$ref": ref}]
            if description:
                allof_items.append({"description": description})
            variant["allOf"] = allof_items


def _wrap_direct_ref_in_allof(prop: dict[str, Any], schemas: dict[str, Any]) -> None:
    """Convert a non-nullable direct $ref property to allOf wrapping with description.

    Transforms Pydantic style:
        {$ref: '#/components/schemas/X'}
    To OpenResponses style:
        {allOf: [{$ref: '#/components/schemas/X'}, {description: '...'}]}
    """
    if "$ref" not in prop:
        return
    ref = prop.pop("$ref")
    ref_name = ref.split("/")[-1]
    ref_schema = schemas.get(ref_name, {})
    desc = ref_schema.get("description")
    allof_items: list[dict[str, Any]] = [{"$ref": ref}]
    if desc:
        allof_items.append({"description": desc})
    prop["allOf"] = allof_items


def _make_truncation_non_nullable_allof(prop: dict[str, Any]) -> None:
    """Convert a nullable truncation $ref to a non-nullable allOf with description.

    Transforms:
        {anyOf: [{$ref: X, title: Y}, {type: null}], description: Z, title: Y}
    To:
        {allOf: [{$ref: X}, {description: Z}]}
    """
    if "anyOf" not in prop:
        return
    description = prop.pop("description", None)
    prop.pop("title", None)
    ref = None
    for variant in prop["anyOf"]:
        if "$ref" in variant:
            ref = variant["$ref"]
            break
    if ref is None:
        return
    del prop["anyOf"]
    allof_items: list[dict[str, Any]] = [{"$ref": ref}]
    if description:
        allof_items.append({"description": description})
    prop["allOf"] = allof_items


def _fix_responses_schema_conformance(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Align Responses API property schemas with OpenResponses spec structure."""
    schemas = openapi_schema.get("components", {}).get("schemas", {})

    nullable_allof_fields = {
        "CreateResponseRequest": ["reasoning", "text", "stream_options"],
        "OpenAIResponseObject": ["reasoning", "error", "incomplete_details", "usage"],
    }
    for schema_name, fields in nullable_allof_fields.items():
        props = schemas.get(schema_name, {}).get("properties", {})
        for field_name in fields:
            if field_name in props:
                _wrap_ref_in_allof(props[field_name])

    resp_props = schemas.get("OpenAIResponseObject", {}).get("properties", {})
    if "text" in resp_props:
        _wrap_direct_ref_in_allof(resp_props["text"], schemas)

    truncation_fields = {
        "CreateResponseRequest": "truncation",
        "OpenAIResponseObject": "truncation",
    }
    for schema_name, field_name in truncation_fields.items():
        props = schemas.get(schema_name, {}).get("properties", {})
        if field_name in props:
            _make_truncation_non_nullable_allof(props[field_name])

    if "metadata" in resp_props:
        desc = resp_props["metadata"].pop("description", None)
        resp_props["metadata"].clear()
        if desc:
            resp_props["metadata"]["description"] = desc

    return openapi_schema

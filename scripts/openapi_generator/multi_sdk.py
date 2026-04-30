# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Multi-SDK response schema transforms for OpenAPI generation.

Adds oneOf response schemas and SDK detection header parameters to endpoints
that return different response shapes based on which SDK is calling.
"""

import copy
from typing import Any

# Endpoints that return different response shapes based on SDK detection headers.
_MULTI_SDK_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/v1/models": {
        "list_schemas": [
            {"$ref": "#/components/schemas/OpenAIListModelsResponse", "title": "OpenAIListModelsResponse"},
            {"$ref": "#/components/schemas/AnthropicListModelsResponse", "title": "AnthropicListModelsResponse"},
            {"$ref": "#/components/schemas/GoogleListModelsResponse", "title": "GoogleListModelsResponse"},
        ],
        "description": (
            "Returns OpenAI format by default. "
            "Send `anthropic-version` header for Anthropic format, "
            "or `x-goog-api-key` header for Google format."
        ),
    },
    "/v1/models/{model_id}": {
        "list_schemas": [
            {"$ref": "#/components/schemas/Model", "title": "Model"},
            {"$ref": "#/components/schemas/AnthropicModelInfo", "title": "AnthropicModelInfo"},
            {"$ref": "#/components/schemas/GoogleModelInfo", "title": "GoogleModelInfo"},
        ],
        "description": (
            "Returns OpenAI format by default. "
            "Send `anthropic-version` header for Anthropic format, "
            "or `x-goog-api-key` header for Google format."
        ),
    },
}

_SDK_DETECTION_HEADERS = [
    {
        "name": "anthropic-version",
        "in": "header",
        "required": False,
        "schema": {"type": "string"},
        "description": (
            "When present, the response uses the Anthropic Models API format. "
            "The Anthropic SDK sends this header automatically."
        ),
    },
    {
        "name": "x-goog-api-key",
        "in": "header",
        "required": False,
        "schema": {"type": "string"},
        "description": (
            "When present, the response uses the Google Models API format. "
            "The Google AI SDK sends this header automatically."
        ),
    },
]


def _add_multi_sdk_response_schemas(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Add oneOf response schemas and SDK detection header parameters to multi-SDK endpoints.

    Endpoints like /v1/models return different response shapes depending on which SDK
    is calling (detected via headers). This transform updates the OpenAPI spec to
    document all possible response shapes using oneOf and adds the detection headers
    as optional parameters.
    """
    paths = openapi_schema.get("paths", {})

    for path, config in _MULTI_SDK_ENDPOINTS.items():
        path_item = paths.get(path)
        if not path_item or "get" not in path_item:
            continue

        operation = path_item["get"]

        # Replace single-schema response with oneOf
        response_200 = operation.get("responses", {}).get("200", {})
        json_content = response_200.get("content", {}).get("application/json", {})
        if json_content:
            json_content["schema"] = {"oneOf": config["list_schemas"]}

        # Update the response description
        response_200["description"] = config["description"]

        # Add SDK detection header parameters
        if "parameters" not in operation:
            operation["parameters"] = []

        existing_param_names = {p.get("name") for p in operation["parameters"]}
        for header in _SDK_DETECTION_HEADERS:
            if header["name"] not in existing_param_names:
                operation["parameters"].append(copy.deepcopy(header))

    return openapi_schema

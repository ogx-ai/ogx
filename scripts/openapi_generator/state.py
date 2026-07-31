# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Shared state for the OpenAPI generator module.
"""

from typing import Any

from ogx_api import Api
from ogx_api.schema_utils import clear_dynamic_schema_types, register_dynamic_schema_type

_dynamic_model_registry: dict[str, type] = {}

# Cache for protocol methods to avoid repeated lookups
_protocol_methods_cache: dict[Api, dict[str, Any]] | None = None

# Global dict to store extra body field information by endpoint
# Key: (path, method) tuple, Value: list of (param_name, param_type, description) tuples
_extra_body_fields: dict[tuple[str, str], list[tuple[str, type, str | None]]] = {}


def register_dynamic_model(name: str, model: type) -> type:
    """Register and deduplicate dynamically generated request models."""
    existing = _dynamic_model_registry.get(name)
    if existing is not None:
        register_dynamic_schema_type(existing)
        return existing
    _dynamic_model_registry[name] = model
    register_dynamic_schema_type(model)
    return model


def reset_generator_state() -> None:
    """Clear per-run caches so repeated generations stay deterministic."""
    _dynamic_model_registry.clear()
    _extra_body_fields.clear()
    clear_dynamic_schema_types()

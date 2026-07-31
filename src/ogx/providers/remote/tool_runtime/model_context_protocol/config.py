# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class MCPProviderDataValidator(BaseModel):
    """
    Validator for MCP provider-specific data passed via request headers.

    Phase 1: Support old header-based authentication for backward compatibility.
    In Phase 2, this will be deprecated in favor of the authorization parameter.
    """

    mcp_headers: dict[str, dict[str, str]] | None = None  # Map of URI -> headers dict


class MCPProviderConfig(BaseModel):
    """Configuration for the Model Context Protocol tool runtime provider."""

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {}

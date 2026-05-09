# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
from pydantic import BaseModel, Field


class BaseToolRuntimeConfig(BaseModel):
    """Base configuration for outbound HTTP clients used by tool runtimes.

    Provides sensible defaults: 30s total covers slow upstream APIs (search,
    file fetch); 10s connect catches unreachable hosts fast without penalizing
    high-latency responses.
    """

    timeout: float = Field(
        default=30.0,
        description="Overall HTTP timeout in seconds for requests to external services.",
    )
    connect_timeout: float = Field(
        default=10.0,
        description="TCP connect timeout in seconds. Shorter than the overall timeout to fail fast on unreachable hosts.",
    )

    def to_httpx_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout, connect=self.connect_timeout)

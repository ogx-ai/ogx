# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx_api import NetworkPolicy


class BuiltinContainersConfig(BaseModel):
    """Configuration for the built-in Containers provider."""

    default_network_policy: NetworkPolicy = Field(
        default_factory=NetworkPolicy,
        description=(
            "Operator-set network policy applied as the upper bound for every container. "
            "Request-supplied network policies may only narrow this, never expand it."
        ),
    )
    allowed_images: list[str] | None = Field(
        default=None,
        description=(
            "If set, restricts the images a request may select. Requests for any other "
            "image are rejected. When unset, any image is permitted."
        ),
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {}

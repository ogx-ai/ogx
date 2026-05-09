# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from typing import Any

from pydantic import BaseModel, Field

from ogx_api import json_schema_type


@json_schema_type
class NVIDIASafetyConfig(BaseModel):
    """
    Configuration for the NVIDIA Guardrail microservice endpoint.

    Attributes:
        guardrails_service_url (str): A base url for accessing the NVIDIA guardrail endpoint, e.g. http://0.0.0.0:7331
        config_id (str): The ID of the guardrails configuration to use from the configuration store
         (https://developer.nvidia.com/docs/nemo-microservices/guardrails/source/guides/configuration-store-guide.html)

    """

    guardrails_service_url: str = Field(
        default_factory=lambda: os.getenv("GUARDRAILS_SERVICE_URL", "http://0.0.0.0:7331"),
        description="The url for accessing the Guardrails service",
    )
    config_id: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_GUARDRAILS_CONFIG_ID", "self-check"),
        description="Guardrails configuration ID to use from the Guardrails configuration store",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "guardrails_service_url": "${env.GUARDRAILS_SERVICE_URL:=http://localhost:7331}",
            "config_id": "${env.NVIDIA_GUARDRAILS_CONFIG_ID:=self-check}",
        }

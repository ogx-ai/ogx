# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import Any

from pydantic import BaseModel, Field

from ogx.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from ogx_api import json_schema_type


class OCIProviderDataValidator(BaseModel):
    """Validates provider-specific request data for OCI inference."""

    oci_auth_type: str = Field(
        description="OCI authentication type (must be one of: instance_principal, config_file)",
    )
    oci_region: str = Field(
        description="OCI region (e.g., us-ashburn-1)",
    )
    oci_compartment_id: str = Field(
        description="OCI compartment ID for the Generative AI service",
    )
    oci_config_file_path: str | None = Field(
        default="~/.oci/config",
        description="OCI config file path (required if oci_auth_type is config_file)",
    )
    oci_config_profile: str | None = Field(
        default="DEFAULT",
        description="OCI config profile (required if oci_auth_type is config_file)",
    )


@json_schema_type
class OCIConfig(RemoteInferenceProviderConfig):
    """Configuration for the Oracle Cloud Infrastructure inference provider."""

    oci_auth_type: str = Field(
        description="OCI authentication type (must be one of: instance_principal, config_file)",
        default_factory=lambda: os.getenv("OCI_AUTH_TYPE", "instance_principal"),
    )
    oci_region: str = Field(
        default_factory=lambda: os.getenv("OCI_REGION", "us-ashburn-1"),
        description="OCI region (e.g., us-ashburn-1)",
    )
    oci_compartment_id: str = Field(
        default_factory=lambda: os.getenv("OCI_COMPARTMENT_OCID", ""),
        description="OCI compartment ID for the Generative AI service",
    )
    oci_config_file_path: str = Field(
        default_factory=lambda: os.getenv("OCI_CONFIG_FILE_PATH", "~/.oci/config"),
        description="OCI config file path (required if oci_auth_type is config_file)",
    )
    oci_config_profile: str = Field(
        default_factory=lambda: os.getenv("OCI_CLI_PROFILE", "DEFAULT"),
        description="OCI config profile (required if oci_auth_type is config_file)",
    )

    @classmethod
    def sample_run_config(
        cls,
        oci_auth_type: str = "${env.OCI_AUTH_TYPE:=instance_principal}",
        oci_config_file_path: str = "${env.OCI_CONFIG_FILE_PATH:=~/.oci/config}",
        oci_config_profile: str = "${env.OCI_CLI_PROFILE:=DEFAULT}",
        oci_region: str = "${env.OCI_REGION:=us-ashburn-1}",
        oci_compartment_id: str = "${env.OCI_COMPARTMENT_OCID:=}",
        **kwargs,
    ) -> dict[str, Any]:
        return {
            "oci_auth_type": oci_auth_type,
            "oci_config_file_path": oci_config_file_path,
            "oci_config_profile": oci_config_profile,
            "oci_region": oci_region,
            "oci_compartment_id": oci_compartment_id,
        }

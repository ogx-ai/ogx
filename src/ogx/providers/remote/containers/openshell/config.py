# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference


class OpenShellContainersImplConfig(BaseModel):
    """Configuration for the OpenShell containers provider."""

    gateway_endpoint: str | None = Field(
        default=None,
        description="Explicit host:port for the OpenShell gateway gRPC endpoint. Takes highest priority.",
    )
    cluster_name: str | None = Field(
        default=None,
        description="OpenShell cluster name, resolved via ~/.config/openshell/gateways/<name>/metadata.json.",
    )
    ready_timeout_seconds: float = Field(
        default=120.0,
        description="Max seconds to wait for a sandbox to reach READY phase after creation.",
    )
    grpc_timeout: float = Field(
        default=30.0,
        description="Per-RPC deadline in seconds for gRPC calls.",
    )
    tls_ca_path: str | None = Field(
        default=None,
        description="Path to CA certificate for mTLS. Only used with explicit gateway_endpoint.",
    )
    tls_cert_path: str | None = Field(
        default=None,
        description="Path to client certificate for mTLS.",
    )
    tls_key_path: str | None = Field(
        default=None,
        description="Path to client private key for mTLS.",
    )
    metadata_store: SqlStoreReference = Field(
        description="SQL store configuration for container-to-sandbox mapping metadata.",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "gateway_endpoint": "localhost:50051",
            "ready_timeout_seconds": 120.0,
            "grpc_timeout": 30.0,
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="openshell_containers_metadata",
            ).model_dump(exclude_none=True),
        }

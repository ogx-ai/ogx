# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel, Field

from ogx.core.storage.datatypes import SqlStoreReference

# Default engine sockets. Podman exposes a Docker-compatible API socket, so the
# docker-py SDK can drive both engines by pointing at the right URL.
DEFAULT_DOCKER_SOCKET = "unix:///var/run/docker.sock"
DEFAULT_PODMAN_SOCKET = "unix:///run/podman/podman.sock"


class LocalContainerRuntimeConfig(BaseModel):
    """Configuration for the local Docker/Podman container runtime."""

    engine: Literal["docker", "podman"] = Field(
        default="docker",
        description="Container engine to drive. Both are accessed through the docker-py SDK.",
    )
    socket_url: str | None = Field(
        default=None,
        description=(
            "Engine API socket URL. Defaults to the standard Docker or Podman socket for "
            "the selected engine when unset."
        ),
    )
    default_image: str = Field(
        default="python:3.12-slim",
        description="Image used for new containers when a request does not specify one.",
    )
    max_containers: int = Field(
        default=50,
        ge=1,
        description="Maximum number of concurrently active containers this runtime will create.",
    )
    enable_local: bool = Field(
        default=False,
        description="Allow non-container ('local') shell execution on the host. Disabled by default.",
    )
    expiration_poll_seconds: int = Field(
        default=60,
        ge=5,
        description="Interval between sweeps of the background task that reaps expired containers.",
    )
    metadata_store: SqlStoreReference = Field(
        description="SQL store for container and container-file metadata.",
    )

    def resolved_socket_url(self) -> str:
        """Return the effective engine socket URL, applying engine defaults."""
        if self.socket_url:
            return self.socket_url
        return DEFAULT_PODMAN_SOCKET if self.engine == "podman" else DEFAULT_DOCKER_SOCKET

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "engine": "${env.CONTAINER_ENGINE:=docker}",
            "default_image": "${env.CONTAINER_DEFAULT_IMAGE:=python:3.12-slim}",
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="container_runtime_metadata",
            ).model_dump(exclude_none=True),
        }

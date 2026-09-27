# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Internal data models for the job execution substrate.

The public, API-facing job representation is :class:`ogx_api.common.job_types.Job`
(and the per-API extensions of it). The models here are the *internal* persisted
form and the serializable description a worker process needs to rebuild a
provider impl.
"""

import time
from typing import Any

from pydantic import BaseModel, Field

from ogx.core.access_control.datatypes import AccessRule
from ogx_api.common.job_types import JobStatus

# Terminal states: a job in one of these will not change again.
TERMINAL_STATUSES = frozenset({JobStatus.completed, JobStatus.failed, JobStatus.cancelled})


class ProviderDescriptor(BaseModel):
    """Serializable description of a provider impl, used to rebuild it in a worker.

    A worker process cannot share the server's in-memory provider objects, so it
    reconstructs the real impl from this descriptor: import ``module``, build
    ``config_class`` from ``config``, and call ``method`` (``get_provider_impl``
    for inline providers, ``get_adapter_impl`` for remote ones). ``dependencies``
    carries the same description for each direct API dependency (e.g. a file
    processor depends on the ``files`` provider), keyed by API name.
    """

    api: str = Field(description="API value this provider implements, e.g. 'file_processors'.")
    provider_id: str = Field(description="Provider id as configured in the run config.")
    provider_type: str = Field(description="Provider type, e.g. 'inline::pypdf'.")
    module: str = Field(description="Python module exposing the provider entrypoint function.")
    config_class: str = Field(description="Fully-qualified Pydantic config class for the provider.")
    config: dict[str, Any] = Field(default_factory=dict, description="Resolved provider config as a plain dict.")
    method: str = Field(
        default="get_provider_impl",
        description="Entrypoint function on the module: get_provider_impl or get_adapter_impl.",
    )
    pass_policy: bool = Field(
        default=False,
        description="Whether the entrypoint accepts the access policy as a third argument.",
    )
    policy: list[AccessRule] = Field(
        default_factory=list,
        description="Configured access policy passed to policy-aware provider factories.",
    )
    dependencies: dict[str, "ProviderDescriptor"] = Field(
        default_factory=dict,
        description="Descriptors for this provider's direct API dependencies, keyed by API value.",
    )


class JobRecord(BaseModel):
    """Persisted representation of a single unit of work.

    ``payload`` holds the serialized method arguments (never file bytes — large
    payloads are passed by reference via the Files API). ``result`` and ``error``
    are populated once the job reaches a terminal state.
    """

    job_id: str
    api: str
    provider_id: str
    method: str = Field(description="Provider method the worker should invoke, e.g. 'process_file'.")
    status: JobStatus = JobStatus.scheduled
    payload: dict[str, Any] = Field(default_factory=dict)
    owner_principal: str | None = None
    owner_tenant_id: str | None = None
    authenticated_user: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 1
    lease_owner: str | None = None
    lease_expires_at: int = 0
    cleanup_owner: str | None = None
    cleanup_lease_expires_at: int | None = 0
    cleaned_at: int | None = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

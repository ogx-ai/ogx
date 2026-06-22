# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""The resolver seam: an ``execution_mode: worker`` spec must mount the registered
proxy instead of the real impl, and hand the worker a descriptor that can rebuild
the provider (and its dependencies) out of process. This is the only place where
"config says worker" gets translated into "the server holds a proxy".
"""

from typing import cast

import pytest
from pydantic import BaseModel

from ogx.core.jobs import file_processor_proxy  # noqa: F401  registers the file_processors proxy factory
from ogx.core.jobs.file_processor_proxy import FileProcessorJobProxy
from ogx.core.jobs.queue import JobQueue
from ogx.core.jobs.runtime import JobRuntime, register_job_runtime, reset_job_runtime
from ogx.core.jobs.worker import WorkerPool
from ogx.core.resolver import ProviderWithSpec, _instantiate_worker_proxy
from ogx.providers.inline.file_processor.pypdf import PyPDFFileProcessorConfig
from ogx_api import Api, InlineProviderSpec


class _DummyConfig(BaseModel):
    pass


class _Provider:
    def __init__(self, provider_id: str):
        self.provider_id = provider_id


def _provider(provider_id: str) -> ProviderWithSpec:
    return cast(ProviderWithSpec, _Provider(provider_id))


class _DepImpl:
    """Stands in for a resolved dependency, carrying the attrs the resolver reads back."""

    __provider_id__: str
    __provider_spec__: InlineProviderSpec
    __provider_config__: BaseModel


@pytest.fixture
def runtime(queue: JobQueue):
    pool = WorkerPool(jobs_backend="sql_default", jobs_table="jobs", num_workers=1)
    rt = JobRuntime(queue=queue, pool=pool)
    register_job_runtime(rt)
    yield rt
    reset_job_runtime()


def _pypdf_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::pypdf",
        execution_mode="worker",
        module="ogx.providers.inline.file_processor.pypdf",
        config_class="ogx.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
        api_dependencies=[Api.files],
    )


def _files_dep() -> _DepImpl:
    dep = _DepImpl()
    # The resolver reconstructs dependency descriptors from these attrs, which the
    # real resolution path stamps onto every impl it builds.
    dep.__provider_id__ = "localfs"
    dep.__provider_spec__ = InlineProviderSpec(
        api=Api.files,
        provider_type="inline::localfs",
        module="ogx.providers.inline.files.localfs",
        config_class="ogx.providers.inline.files.localfs.LocalfsFilesImplConfig",
    )
    dep.__provider_config__ = _DummyConfig()
    return dep


def test_worker_mode_returns_proxy_and_registers_reconstructable_descriptor(runtime: JobRuntime):
    deps = {Api.files: _files_dep()}

    impl = _instantiate_worker_proxy(_provider("pypdf"), _pypdf_spec(), PyPDFFileProcessorConfig(), deps)

    # The server mounts the proxy (not the heavy pypdf impl) wired to the shared queue.
    assert isinstance(impl, FileProcessorJobProxy)
    assert impl.job_queue is runtime.queue
    assert impl.files_api is deps[Api.files]

    # The pool received a descriptor that lets a worker rebuild pypdf AND its files dep.
    assert len(runtime.pool._descriptors) == 1
    descriptor = runtime.pool._descriptors[0]
    assert descriptor.provider_id == "pypdf"
    assert descriptor.method == "get_provider_impl"
    assert "files" in descriptor.dependencies
    assert descriptor.dependencies["files"].provider_id == "localfs"


def test_worker_mode_requires_initialized_runtime():
    reset_job_runtime()
    with pytest.raises(RuntimeError, match="job runtime"):
        _instantiate_worker_proxy(_provider("pypdf"), _pypdf_spec(), PyPDFFileProcessorConfig(), {})

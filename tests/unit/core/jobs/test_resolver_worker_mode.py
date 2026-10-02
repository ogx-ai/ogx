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
from pydantic import BaseModel, SecretStr

from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import AutoRoutedProviderSpec
from ogx.core.jobs import file_processor_proxy  # noqa: F401  registers the file_processors proxy factory
from ogx.core.jobs.file_processor_proxy import FileProcessorJobProxy
from ogx.core.jobs.queue import JobQueue
from ogx.core.jobs.runtime import JobRuntime, register_job_runtime, reset_job_runtime
from ogx.core.jobs.worker import WorkerPool
from ogx.core.resolver import (
    ProviderWithSpec,
    _descriptor_for_impl,
    _instantiate_worker_proxy,
    _register_worker_sibling_descriptors,
)
from ogx.core.storage.datatypes import SqlStoreReference
from ogx.providers.inline.file_processor.pypdf import PyPDFFileProcessorConfig
from ogx.providers.remote.files.s3.config import S3FilesImplConfig
from ogx_api import Api, InlineProviderSpec, RemoteProviderSpec


class _DummyConfig(BaseModel):
    pass


class _VlmConfig(BaseModel):
    vlm_model: str


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

    impl = _instantiate_worker_proxy(_provider("pypdf"), _pypdf_spec(), PyPDFFileProcessorConfig(), deps, [])

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
        _instantiate_worker_proxy(_provider("pypdf"), _pypdf_spec(), PyPDFFileProcessorConfig(), {}, [])


def test_worker_descriptor_preserves_secrets_and_policy() -> None:
    policy = [AccessRule(permit=Scope(actions=Action.READ))]
    spec = RemoteProviderSpec(
        api=Api.files,
        provider_type="remote::s3",
        adapter_type="s3",
        module="ogx.providers.remote.files.s3",
        config_class="ogx.providers.remote.files.s3.config.S3FilesImplConfig",
    )
    config = S3FilesImplConfig(
        bucket_name="test-bucket",
        aws_access_key_id=SecretStr("access-key"),
        aws_secret_access_key=SecretStr("secret-key"),
        metadata_store=SqlStoreReference(backend="sql_default", table_name="s3_files"),
    )

    descriptor = _descriptor_for_impl("files", "s3", spec, config, policy)

    assert descriptor.config["aws_access_key_id"].get_secret_value() == "access-key"
    assert descriptor.config["aws_secret_access_key"].get_secret_value() == "secret-key"
    assert descriptor.policy == policy


def test_worker_descriptor_skips_unreconstructable_optional_autorouter(runtime: JobRuntime) -> None:
    inference = _DepImpl()
    inference.__provider_id__ = "__autorouted__"
    inference.__provider_spec__ = AutoRoutedProviderSpec(
        api=Api.inference,
        module="ogx.core.routers",
        routing_table_api=Api.models,
        api_dependencies=[Api.models],
    )
    inference.__provider_config__ = None
    spec = InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::docling",
        execution_mode="worker",
        module="ogx.providers.inline.file_processor.docling",
        config_class="ogx.providers.inline.file_processor.docling.DoclingFileProcessorConfig",
        api_dependencies=[Api.files],
        optional_api_dependencies=[Api.inference],
    )

    _instantiate_worker_proxy(
        _provider("docling"),
        spec,
        _DummyConfig(),
        {Api.files: _files_dep(), Api.inference: inference},
        [],
    )

    descriptor = runtime.pool._descriptors[0]
    assert set(descriptor.dependencies) == {"files"}


def test_worker_descriptor_rejects_unreconstructable_configured_vlm(runtime: JobRuntime) -> None:
    inference = _DepImpl()
    inference.__provider_id__ = "__autorouted__"
    inference.__provider_spec__ = AutoRoutedProviderSpec(
        api=Api.inference,
        module="ogx.core.routers",
        routing_table_api=Api.models,
        api_dependencies=[Api.models],
    )
    inference.__provider_config__ = None
    spec = InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::docling",
        execution_mode="worker",
        module="ogx.providers.inline.file_processor.docling",
        config_class="ogx.providers.inline.file_processor.docling.DoclingFileProcessorConfig",
        api_dependencies=[Api.files],
        optional_api_dependencies=[Api.inference],
    )

    with pytest.raises(ValueError, match="required dependency 'inference' cannot be reconstructed"):
        _instantiate_worker_proxy(
            _provider("docling"),
            spec,
            _VlmConfig(vlm_model="some-model"),
            {Api.files: _files_dep(), Api.inference: inference},
            [],
        )


def test_worker_descriptor_rejects_configured_vlm_without_inference(runtime: JobRuntime) -> None:
    spec = InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::docling",
        execution_mode="worker",
        module="ogx.providers.inline.file_processor.docling",
        config_class="ogx.providers.inline.file_processor.docling.DoclingFileProcessorConfig",
        api_dependencies=[Api.files],
        optional_api_dependencies=[Api.inference],
    )

    with pytest.raises(ValueError, match="configured VLM processing requires"):
        _instantiate_worker_proxy(
            _provider("docling"),
            spec,
            _VlmConfig(vlm_model="some-model"),
            {Api.files: _files_dep()},
            [],
        )


def test_non_worker_file_processor_siblings_are_registered_for_worker_reconstruction(runtime: JobRuntime) -> None:
    runtime.pool.register(
        runtime.pool._descriptors[0]
        if runtime.pool._descriptors
        else _descriptor_for_impl("file_processors", "auto", _pypdf_spec(), PyPDFFileProcessorConfig(), [])
    )
    sibling = _DepImpl()
    sibling.__provider_id__ = "pypdf-inline"
    sibling.__provider_spec__ = InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::pypdf",
        execution_mode="inline",
        module="ogx.providers.inline.file_processor.pypdf",
        config_class="ogx.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
        api_dependencies=[Api.files],
    )
    sibling.__provider_config__ = PyPDFFileProcessorConfig()
    sibling.__provider_deps__ = {Api.files: _files_dep()}

    _register_worker_sibling_descriptors(
        {"file_processors": {"pypdf-inline": sibling}},
        runtime.pool,
        [],
    )

    descriptors = {(descriptor.api, descriptor.provider_id): descriptor for descriptor in runtime.pool._descriptors}
    sibling_descriptor = descriptors[("file_processors", "pypdf-inline")]
    assert sibling_descriptor.dependencies["files"].provider_id == "localfs"

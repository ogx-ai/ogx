# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Exercises the worker-side reconstruction of a real provider from a descriptor.

This is the one piece of the worker that cannot be covered with stubs: rebuilding
an actual provider impl (with config, dependencies, policy, and initialize()) from
the serializable ProviderDescriptor the server hands off.
"""

from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.jobs import worker
from ogx.core.jobs.models import ProviderDescriptor
from ogx.core.jobs.worker import _build_impl
from ogx.core.storage.datatypes import SqliteSqlStoreConfig
from ogx.core.storage.sqlstore.sqlstore import register_sqlstore_backends


async def test_build_impl_reconstructs_provider_with_dependencies(tmp_path):
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=str(tmp_path / "meta.db"))})

    # A file processor descriptor that depends on a real localfs files provider —
    # the same shape the resolver produces for worker-mode file_processors.
    files_descriptor = ProviderDescriptor(
        api="files",
        provider_id="localfs",
        provider_type="inline::localfs",
        module="ogx.providers.inline.files.localfs",
        config_class="ogx.providers.inline.files.localfs.LocalfsFilesImplConfig",
        config={
            "storage_dir": str(tmp_path / "files"),
            "metadata_store": {"backend": "sql_default", "table_name": "files_metadata"},
        },
        method="get_provider_impl",
        pass_policy=True,
    )

    descriptor = ProviderDescriptor(
        api="file_processors",
        provider_id="pypdf",
        provider_type="inline::pypdf",
        module="ogx.providers.inline.file_processor.pypdf",
        config_class="ogx.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
        config={},
        method="get_provider_impl",
        pass_policy=False,
        dependencies={"files": files_descriptor},
    )

    policy = [AccessRule(permit=Scope(actions=Action.READ))]
    files_descriptor.policy = policy
    impl = await _build_impl(descriptor)

    # The rebuilt processor is a working impl wired to its (also rebuilt) files dependency.
    assert hasattr(impl, "process_file")
    assert impl.files_api is not None
    assert hasattr(impl.files_api, "openai_upload_file")
    assert impl.files_api.policy == policy


def test_worker_wires_same_api_sibling_providers() -> None:
    class _Impl:
        def __init__(self):
            self.siblings = None

        def set_sibling_providers(self, siblings):
            self.siblings = siblings

    auto = _Impl()
    pypdf = object()
    markitdown = object()
    impls = {
        ("file_processors", "auto"): auto,
        ("file_processors", "pypdf"): pypdf,
        ("file_processors", "markitdown"): markitdown,
    }

    worker._wire_sibling_providers(impls)

    assert auto.siblings == {"pypdf": pypdf, "markitdown": markitdown}

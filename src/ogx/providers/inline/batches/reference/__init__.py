# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx.core.datatypes import AccessRule, Api
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx_api import Files, Inference, Models

from .batches import ReferenceBatchesImpl
from .config import ReferenceBatchesImplConfig

__all__ = ["ReferenceBatchesImpl", "ReferenceBatchesImplConfig"]


async def get_provider_impl(config: ReferenceBatchesImplConfig, deps: dict[Api, Any], policy: list[AccessRule]):
    base_sql_store = sqlstore_impl(config.sqlstore)
    sql_store = AuthorizedSqlStore(base_sql_store, policy)
    inference_api: Inference | None = deps.get(Api.inference)
    files_api: Files | None = deps.get(Api.files)
    models_api: Models | None = deps.get(Api.models)

    if inference_api is None:
        raise ValueError("Inference API is required but not provided in dependencies")
    if files_api is None:
        raise ValueError("Files API is required but not provided in dependencies")
    if models_api is None:
        raise ValueError("Models API is required but not provided in dependencies")

    impl = ReferenceBatchesImpl(config, inference_api, files_api, models_api, sql_store)
    await impl.initialize()
    return impl

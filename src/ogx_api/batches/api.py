# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from openai.types import Batch as BatchObject

from .models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    ListBatchesResponse,
    RetrieveBatchRequest,
)


@runtime_checkable
class Batches(Protocol):
    """
    The Batches API enables efficient processing of multiple requests in a single operation,
    particularly useful for processing large datasets, batch evaluation workflows, and
    cost-effective inference at scale.

    The API is designed to allow use of openai client libraries for seamless integration.

    This API provides the following extensions:
     - idempotent batch creation

    Note: This API is currently under active development and may undergo changes.
    """

    async def create_batch(
        self,
        request: CreateBatchRequest,
    ) -> BatchObject: ...

    async def retrieve_batch(
        self,
        request: RetrieveBatchRequest,
    ) -> BatchObject: ...

    async def cancel_batch(
        self,
        request: CancelBatchRequest,
    ) -> BatchObject: ...

    async def list_batches(
        self,
        request: ListBatchesRequest,
    ) -> ListBatchesResponse: ...

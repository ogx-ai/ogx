# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for idempotency functionality in the reference batches provider.

This module tests the optional idempotency feature that allows clients to provide
an idempotency key (idempotency_key) to ensure that repeated requests with the same key
and parameters return the same batch, while requests with the same key but different
parameters result in a conflict error.

Test Categories:
1. Core Idempotency: Same parameters with same key return same batch
2. Parameter Independence: Different parameters without keys create different batches
3. Conflict Detection: Same key with different parameters raises ConflictError

Tests by Category:

1. Core Idempotency:
   - test_idempotent_batch_creation_same_params
   - test_idempotent_batch_creation_metadata_order_independence

2. Parameter Independence:
   - test_non_idempotent_behavior_without_key
   - test_different_idempotency_keys_create_different_batches

3. Conflict Detection:
   - test_same_idempotency_key_different_params_conflict (parametrized: input_file_id, metadata values, metadata None vs {})

Key Behaviors Tested:
- Idempotent batch creation when idempotency_key provided with identical parameters
- Metadata order independence for consistent batch ID generation
- Non-idempotent behavior when no idempotency_key provided (random UUIDs)
- Conflict detection for parameter mismatches with same idempotency key
- Deterministic ID generation based solely on idempotency key
- Proper error handling with detailed conflict messages including key and error codes
- Protection against idempotency key reuse with different request parameters
"""

import asyncio

import pytest

from ogx_api import ConflictError
from ogx_api.batches.models import CreateBatchRequest, RetrieveBatchRequest


class TestReferenceBatchesIdempotency:
    """Test suite for idempotency functionality in the reference implementation."""

    async def test_idempotent_batch_creation_same_params(self, provider, sample_batch_data):
        """Test that creating batches with identical parameters returns the same batch when idempotency_key is provided."""

        del sample_batch_data["metadata"]

        batch1 = await provider.create_batch(
            CreateBatchRequest(
                **sample_batch_data,
                metadata={"test": "value1", "other": "value2"},
                idempotency_key="unique-token-1",
            )
        )

        # sleep for 1 second to allow created_at timestamps to be different
        await asyncio.sleep(1)

        batch2 = await provider.create_batch(
            CreateBatchRequest(
                **sample_batch_data,
                metadata={"other": "value2", "test": "value1"},  # Different order
                idempotency_key="unique-token-1",
            )
        )

        assert batch1.id == batch2.id
        assert batch1.input_file_id == batch2.input_file_id
        assert batch1.metadata == batch2.metadata
        assert batch1.created_at == batch2.created_at

    async def test_different_idempotency_keys_create_different_batches(self, provider, sample_batch_data):
        """Test that different idempotency keys create different batches even with same params."""
        batch1 = await provider.create_batch(CreateBatchRequest(**sample_batch_data, idempotency_key="token-A"))

        batch2 = await provider.create_batch(CreateBatchRequest(**sample_batch_data, idempotency_key="token-B"))

        assert batch1.id != batch2.id

    async def test_non_idempotent_behavior_without_key(self, provider, sample_batch_data):
        """Test that batches without idempotency key create unique batches even with identical parameters."""
        batch1 = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        batch2 = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        assert batch1.id != batch2.id
        assert batch1.input_file_id == batch2.input_file_id
        assert batch1.endpoint == batch2.endpoint
        assert batch1.completion_window == batch2.completion_window
        assert batch1.metadata == batch2.metadata

    @pytest.mark.parametrize(
        "param_name,first_value,second_value",
        [
            ("input_file_id", "file_001", "file_002"),
            ("metadata", {"test": "value1"}, {"test": "value2"}),
            ("metadata", None, {}),
        ],
    )
    async def test_same_idempotency_key_different_params_conflict(
        self, provider, sample_batch_data, param_name, first_value, second_value
    ):
        """Test that same idempotency_key with different parameters raises conflict error."""
        sample_batch_data["idempotency_key"] = "same-token"

        sample_batch_data[param_name] = first_value

        batch1 = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        with pytest.raises(ConflictError, match="Idempotency key.*was previously used with different parameters"):
            sample_batch_data[param_name] = second_value
            await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        retrieved_batch = await provider.retrieve_batch(RetrieveBatchRequest(batch_id=batch1.id))
        assert retrieved_batch.id == batch1.id
        assert getattr(retrieved_batch, param_name) == first_value

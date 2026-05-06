# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for batch idempotency functionality using the OpenAI client library.

This module tests the idempotency feature in the batches API using the OpenAI-compatible
client interface. These tests verify that the idempotency key (idempotency_key) works correctly
in a real client-server environment.

Test Categories:
1. Successful Idempotency: Same key returns same batch with identical parameters
   - test_idempotent_batch_creation_successful: Verifies that requests with the same
     idempotency key return identical batches, even with different metadata order

2. Conflict Detection: Same key with conflicting parameters raises HTTP 409 errors
   - test_idempotency_conflict_with_different_params: Verifies that reusing an idempotency key
     with truly conflicting parameters (both file ID and metadata values) raises ConflictError
"""

import time

import pytest
from openai import ConflictError


class TestBatchesIdempotencyIntegration:
    """Integration tests for batch idempotency using OpenAI client."""

    def test_idempotent_batch_creation_successful(self, openai_client):
        """Test that identical requests with same idempotency key return the same batch."""
        batch1 = openai_client.batches.create(
            input_file_id="bogus-id",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "test_type": "idempotency_success",
                "purpose": "integration_test",
            },
            extra_body={"idempotency_key": "test-idempotency-token-1"},
        )

        # sleep to ensure different timestamps
        time.sleep(1)

        batch2 = openai_client.batches.create(
            input_file_id="bogus-id",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "purpose": "integration_test",
                "test_type": "idempotency_success",
            },  # Different order
            extra_body={"idempotency_key": "test-idempotency-token-1"},
        )

        assert batch1.id == batch2.id
        assert batch1.input_file_id == batch2.input_file_id
        assert batch1.endpoint == batch2.endpoint
        assert batch1.completion_window == batch2.completion_window
        assert batch1.metadata == batch2.metadata
        assert batch1.created_at == batch2.created_at

    def test_idempotency_conflict_with_different_params(self, openai_client):
        """Test that using same idempotency key with different params raises conflict error."""
        batch1 = openai_client.batches.create(
            input_file_id="bogus-id-1",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"test_type": "conflict_test_1"},
            extra_body={"idempotency_key": "conflict-token"},
        )

        with pytest.raises(ConflictError) as exc_info:
            openai_client.batches.create(
                input_file_id="bogus-id-2",  # Different file ID
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"test_type": "conflict_test_2"},  # Different metadata
                extra_body={"idempotency_key": "conflict-token"},  # Same token
            )

        assert exc_info.value.status_code == 409
        assert "conflict" in str(exc_info.value).lower()

        retrieved_batch = openai_client.batches.retrieve(batch1.id)
        assert retrieved_batch.id == batch1.id
        assert retrieved_batch.input_file_id == "bogus-id-1"

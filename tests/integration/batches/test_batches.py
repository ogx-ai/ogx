# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for the OGX batch processing functionality.

This module contains comprehensive integration tests for the batch processing API,
using the OpenAI-compatible client interface for consistency.

Test Categories:
    1. Core Batch Operations:
        - test_batch_creation_and_retrieval: Comprehensive batch creation, structure validation, and retrieval
        - test_batch_listing: Basic batch listing functionality
        - test_batch_immediate_cancellation: Batch cancellation workflow
        # TODO: cancel during processing

    2. End-to-End Processing:
        - test_batch_e2e_chat_completions: Full chat completions workflow with output and error validation

Note: Error conditions and edge cases are primarily tested in test_batches_errors.py
for better organization and separation of concerns.

CLEANUP WARNING: These tests currently create batches that are not automatically
cleaned up after test completion. This may lead to resource accumulation over
multiple test runs. Only test_batch_immediate_cancellation properly cancels its batch.
The test_batch_e2e_chat_completions test does clean up its output and error files.
"""

import json


class TestBatchesIntegration:
    """Integration tests for the batches API."""

    def test_batch_creation_and_retrieval(self, openai_client, batch_helper, text_model_id):
        """Test comprehensive batch creation and retrieval scenarios."""
        test_metadata = {
            "test_type": "comprehensive",
            "purpose": "creation_and_retrieval_test",
            "version": "1.0",
            "tags": "test,batch",
        }

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests, "batch_creation_test") as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=test_metadata,
            )

            assert batch.endpoint == "/v1/chat/completions"
            assert batch.input_file_id == uploaded_file.id
            assert batch.completion_window == "24h"
            assert batch.metadata == test_metadata

            retrieved_batch = openai_client.batches.retrieve(batch.id)

            assert retrieved_batch.id == batch.id
            assert retrieved_batch.object == batch.object
            assert retrieved_batch.endpoint == batch.endpoint
            assert retrieved_batch.input_file_id == batch.input_file_id
            assert retrieved_batch.completion_window == batch.completion_window
            assert retrieved_batch.metadata == batch.metadata

    def test_batch_listing(self, openai_client, batch_helper, text_model_id):
        """
        Test batch listing.

        This test creates multiple batches and verifies that they can be listed.
        It also deletes the input files before execution, which means the batches
        will appear as failed due to missing input files. This is expected and
        a good thing, because it means no inference is performed.
        """
        batch_ids = []

        for i in range(2):
            batch_requests = [
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": text_model_id,
                        "messages": [{"role": "user", "content": f"Hello {i}"}],
                        "max_tokens": 10,
                    },
                }
            ]

            with batch_helper.create_file(batch_requests, f"batch_input_{i}") as uploaded_file:
                batch = openai_client.batches.create(
                    input_file_id=uploaded_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                batch_ids.append(batch.id)

        batch_list = openai_client.batches.list()

        assert isinstance(batch_list.data, list)

        listed_batch_ids = {b.id for b in batch_list.data}
        for batch_id in batch_ids:
            assert batch_id in listed_batch_ids

    def test_batch_immediate_cancellation(self, openai_client, batch_helper, text_model_id):
        """Test immediate batch cancellation."""
        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            # hopefully cancel the batch before it completes
            cancelling_batch = openai_client.batches.cancel(batch.id)
            assert cancelling_batch.status in ["cancelling", "cancelled"]
            assert isinstance(cancelling_batch.cancelling_at, int), (
                f"cancelling_at should be int, got {type(cancelling_batch.cancelling_at)}"
            )

            final_batch = batch_helper.wait_for(
                batch.id,
                max_wait_time=3 * 60,  # often takes 10-11 minutes, give it 3 min
                expected_statuses={"cancelled"},
                timeout_action="skip",
            )

        assert final_batch.status == "cancelled"
        assert isinstance(final_batch.cancelled_at, int), (
            f"cancelled_at should be int, got {type(final_batch.cancelled_at)}"
        )

    def test_batch_e2e_chat_completions(self, openai_client, batch_helper, text_model_id):
        """Test end-to-end batch processing for chat completions with both successful and failed operations."""
        batch_requests = [
            {
                "custom_id": "success-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 20,
                },
            },
            {
                "custom_id": "error-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": text_model_id,
                    "messages": [{"rolez": "user", "contentz": "This should fail"}],  # Invalid keys to trigger error
                    # note: ollama does not validate max_tokens values or the "role" key, so they won't trigger an error
                },
            },
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"test": "e2e_success_and_errors_test"},
            )

            final_batch = batch_helper.wait_for(
                batch.id,
                max_wait_time=3 * 60,  # often takes 2-3 minutes
                expected_statuses={"completed"},
                timeout_action="skip",
            )

        # Expecting a completed batch with both successful and failed requests
        #  Batch(id='batch_xxx',
        #        completion_window='24h',
        #        created_at=...,
        #        endpoint='/v1/chat/completions',
        #        input_file_id='file-xxx',
        #        object='batch',
        #        status='completed',
        #        output_file_id='file-xxx',
        #        error_file_id='file-xxx',
        #        request_counts=BatchRequestCounts(completed=1, failed=1, total=2))

        assert final_batch.status == "completed"
        assert final_batch.request_counts is not None
        assert final_batch.request_counts.total == 2
        assert final_batch.request_counts.completed == 1
        assert final_batch.request_counts.failed == 1

        assert final_batch.output_file_id is not None, "Output file should exist for successful requests"

        output_content = openai_client.files.content(final_batch.output_file_id)
        if isinstance(output_content, str):
            output_text = output_content
        else:
            output_text = output_content.content.decode("utf-8")

        output_lines = output_text.strip().split("\n")

        for line in output_lines:
            result = json.loads(line)

            assert "id" in result
            assert "custom_id" in result
            assert result["custom_id"] == "success-1"

            assert "response" in result

            assert result["response"]["status_code"] == 200
            assert "body" in result["response"]
            assert "choices" in result["response"]["body"]

        assert final_batch.error_file_id is not None, "Error file should exist for failed requests"

        error_content = openai_client.files.content(final_batch.error_file_id)
        if isinstance(error_content, str):
            error_text = error_content
        else:
            error_text = error_content.content.decode("utf-8")

        error_lines = error_text.strip().split("\n")

        for line in error_lines:
            result = json.loads(line)

            assert "id" in result
            assert "custom_id" in result
            assert result["custom_id"] == "error-1"
            assert "error" in result
            error = result["error"]
            assert error is not None
            assert "code" in error or "message" in error, "Error should have code or message"

        deleted_output_file = openai_client.files.delete(final_batch.output_file_id)
        assert deleted_output_file.deleted, f"Output file {final_batch.output_file_id} was not deleted successfully"

        deleted_error_file = openai_client.files.delete(final_batch.error_file_id)
        assert deleted_error_file.deleted, f"Error file {final_batch.error_file_id} was not deleted successfully"

    def test_batch_e2e_completions(self, openai_client, batch_helper, text_model_id):
        """Run an end-to-end batch with a single successful text completion request."""
        request_body = {"model": text_model_id, "prompt": "Say completions", "max_tokens": 20}

        batch_requests = [
            {
                "custom_id": "success-1",
                "method": "POST",
                "url": "/v1/completions",
                "body": request_body,
            }
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/completions",
                completion_window="24h",
                metadata={"test": "e2e_completions_success"},
            )

            final_batch = batch_helper.wait_for(
                batch.id,
                max_wait_time=3 * 60,
                expected_statuses={"completed"},
                timeout_action="skip",
            )

        assert final_batch.status == "completed"
        assert final_batch.request_counts is not None
        assert final_batch.request_counts.total == 1
        assert final_batch.request_counts.completed == 1
        assert final_batch.output_file_id is not None

        output_content = openai_client.files.content(final_batch.output_file_id)
        if isinstance(output_content, str):
            output_text = output_content
        else:
            output_text = output_content.content.decode("utf-8")

        output_lines = output_text.strip().split("\n")
        assert len(output_lines) == 1

        result = json.loads(output_lines[0])
        assert result["custom_id"] == "success-1"
        assert "response" in result
        assert result["response"]["status_code"] == 200

        deleted_output_file = openai_client.files.delete(final_batch.output_file_id)
        assert deleted_output_file.deleted

        if final_batch.error_file_id is not None:
            deleted_error_file = openai_client.files.delete(final_batch.error_file_id)
            assert deleted_error_file.deleted

    def test_batch_e2e_embeddings(self, openai_client, batch_helper, embedding_model_id):
        """Run an end-to-end batch with embeddings requests including both string and list inputs."""
        batch_requests = [
            {
                "custom_id": "success-1",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": embedding_model_id, "input": "Hello world", "encoding_format": "float"},
            },
            {
                "custom_id": "success-2",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": embedding_model_id,
                    "input": ["How are you?", "Good morning", "Have a great day"],
                    "encoding_format": "float",
                },
            },
        ]

        with batch_helper.create_file(batch_requests) as uploaded_file:
            batch = openai_client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={"test": "e2e_embeddings_success"},
            )

            final_batch = batch_helper.wait_for(
                batch.id,
                max_wait_time=3 * 60,
                expected_statuses={"completed"},
                timeout_action="skip",
            )

        assert final_batch.status == "completed"
        assert final_batch.request_counts is not None
        assert final_batch.request_counts.total == 2
        assert final_batch.request_counts.completed == 2
        assert final_batch.output_file_id is not None

        output_content = openai_client.files.content(final_batch.output_file_id)
        if isinstance(output_content, str):
            output_text = output_content
        else:
            output_text = output_content.content.decode("utf-8")

        output_lines = output_text.strip().split("\n")
        assert len(output_lines) == 2

        # Check first result (string input)
        result1 = json.loads(output_lines[0])
        assert result1["custom_id"] in ["success-1", "success-2"]
        assert "response" in result1
        assert result1["response"]["status_code"] == 200

        # Verify the response body contains embeddings data
        response_body1 = json.loads(result1["response"]["body"])
        assert response_body1["object"] == "list"
        assert "data" in response_body1
        assert len(response_body1["data"]) == 1
        assert "embedding" in response_body1["data"][0]
        assert "index" in response_body1["data"][0]
        assert response_body1["data"][0]["index"] == 0

        # Check second result (list input)
        result2 = json.loads(output_lines[1])
        assert result2["custom_id"] in ["success-1", "success-2"]
        assert "response" in result2
        assert result2["response"]["status_code"] == 200

        # Verify the response body contains embeddings data for list input
        response_body2 = json.loads(result2["response"]["body"])
        assert response_body2["object"] == "list"
        assert "data" in response_body2
        assert len(response_body2["data"]) == 3  # Three strings in the list
        for i, embedding_data in enumerate(response_body2["data"]):
            assert "embedding" in embedding_data
            assert "index" in embedding_data
            assert embedding_data["index"] == i

        deleted_output_file = openai_client.files.delete(final_batch.output_file_id)
        assert deleted_output_file.deleted

        if final_batch.error_file_id is not None:
            deleted_error_file = openai_client.files.delete(final_batch.error_file_id)
            assert deleted_error_file.deleted

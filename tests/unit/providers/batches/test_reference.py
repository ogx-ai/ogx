# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test suite for the reference implementation of the Batches API.

The tests are categorized and outlined below, keep this updated:

- Batch creation with various parameters and validation:
  * test_create_and_retrieve_batch_success (positive)
  * test_create_batch_without_metadata (positive)
  * test_create_batch_completion_window (negative)
  * test_create_batch_invalid_endpoints (negative)
  * test_create_batch_invalid_metadata (negative)

- Batch retrieval and error handling for non-existent batches:
  * test_retrieve_batch_not_found (negative)

- Batch cancellation with proper status transitions:
  * test_cancel_batch_success (positive)
  * test_cancel_batch_invalid_statuses (negative)
  * test_cancel_batch_not_found (negative)

- Batch listing with pagination and filtering:
  * test_list_batches_empty (positive)
  * test_list_batches_single_batch (positive)
  * test_list_batches_multiple_batches (positive)
  * test_list_batches_with_limit (positive)
  * test_list_batches_with_pagination (positive)
  * test_list_batches_invalid_after (negative)

- Data persistence in the underlying key-value store:
  * test_kvstore_persistence (positive)

- Batch processing concurrency control:
  * test_max_concurrent_batches (positive)

- Input validation testing (direct _validate_input method tests):
  * test_validate_input_file_not_found (negative)
  * test_validate_input_file_exists_empty_content (positive)
  * test_validate_input_file_mixed_valid_invalid_json (mixed)
  * test_validate_input_invalid_model (negative)
  * test_validate_input_url_mismatch (negative)
  * test_validate_input_multiple_errors_per_request (negative)
  * test_validate_input_invalid_request_format (negative)
  * test_validate_input_missing_parameters_chat_completions (parametrized negative - custom_id, method, url, body, model, messages missing validation for chat/completions)
  * test_validate_input_missing_parameters_completions (parametrized negative - custom_id, method, url, body, model, prompt missing validation for completions)
  * test_validate_input_invalid_parameter_types (parametrized negative - custom_id, url, method, body, model, messages type validation)

The tests use temporary SQLite databases for isolation and mock external
dependencies like inference, files, and models APIs.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ogx_api import BatchObject, ConflictError, ResourceNotFoundError
from ogx_api.batches.models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    RetrieveBatchRequest,
)


class TestReferenceBatchesImpl:
    """Test the reference implementation of the Batches API."""

    def _validate_batch_type(self, batch, expected_metadata=None):
        """
        Helper function to validate batch object structure and field types.

        Note: This validates the direct BatchObject from the provider, not the
              client library response which has a different structure.

        Args:
            batch: The BatchObject instance to validate.
            expected_metadata: Optional expected metadata dictionary to validate against.
        """
        assert isinstance(batch.id, str)
        assert isinstance(batch.completion_window, str)
        assert isinstance(batch.created_at, int)
        assert isinstance(batch.endpoint, str)
        assert isinstance(batch.input_file_id, str)
        assert batch.object == "batch"
        assert batch.status in [
            "validating",
            "failed",
            "in_progress",
            "finalizing",
            "completed",
            "expired",
            "cancelling",
            "cancelled",
        ]

        if expected_metadata is not None:
            assert batch.metadata == expected_metadata

        timestamp_fields = [
            "cancelled_at",
            "cancelling_at",
            "completed_at",
            "expired_at",
            "expires_at",
            "failed_at",
            "finalizing_at",
            "in_progress_at",
        ]
        for field in timestamp_fields:
            field_value = getattr(batch, field, None)
            if field_value is not None:
                assert isinstance(field_value, int), f"{field} should be int or None, got {type(field_value)}"

        file_id_fields = ["error_file_id", "output_file_id"]
        for field in file_id_fields:
            field_value = getattr(batch, field, None)
            if field_value is not None:
                assert isinstance(field_value, str), f"{field} should be str or None, got {type(field_value)}"

        if hasattr(batch, "request_counts") and batch.request_counts is not None:
            assert isinstance(batch.request_counts.completed, int), (
                f"request_counts.completed should be int, got {type(batch.request_counts.completed)}"
            )
            assert isinstance(batch.request_counts.failed, int), (
                f"request_counts.failed should be int, got {type(batch.request_counts.failed)}"
            )
            assert isinstance(batch.request_counts.total, int), (
                f"request_counts.total should be int, got {type(batch.request_counts.total)}"
            )

        if hasattr(batch, "errors") and batch.errors is not None:
            assert isinstance(batch.errors, dict), f"errors should be object or dict, got {type(batch.errors)}"

            if hasattr(batch.errors, "data") and batch.errors.data is not None:
                assert isinstance(batch.errors.data, list), (
                    f"errors.data should be list or None, got {type(batch.errors.data)}"
                )

                for i, error_item in enumerate(batch.errors.data):
                    assert isinstance(error_item, dict), (
                        f"errors.data[{i}] should be object or dict, got {type(error_item)}"
                    )

                    if hasattr(error_item, "code") and error_item.code is not None:
                        assert isinstance(error_item.code, str), (
                            f"errors.data[{i}].code should be str or None, got {type(error_item.code)}"
                        )

                    if hasattr(error_item, "line") and error_item.line is not None:
                        assert isinstance(error_item.line, int), (
                            f"errors.data[{i}].line should be int or None, got {type(error_item.line)}"
                        )

                    if hasattr(error_item, "message") and error_item.message is not None:
                        assert isinstance(error_item.message, str), (
                            f"errors.data[{i}].message should be str or None, got {type(error_item.message)}"
                        )

                    if hasattr(error_item, "param") and error_item.param is not None:
                        assert isinstance(error_item.param, str), (
                            f"errors.data[{i}].param should be str or None, got {type(error_item.param)}"
                        )

            if hasattr(batch.errors, "object") and batch.errors.object is not None:
                assert isinstance(batch.errors.object, str), (
                    f"errors.object should be str or None, got {type(batch.errors.object)}"
                )
                assert batch.errors.object == "list", f"errors.object should be 'list', got {batch.errors.object}"

    async def test_create_and_retrieve_batch_success(self, provider, sample_batch_data):
        """Test successful batch creation and retrieval."""
        created_batch = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        self._validate_batch_type(created_batch, expected_metadata=sample_batch_data["metadata"])

        assert created_batch.id.startswith("batch_")
        assert len(created_batch.id) > 13
        assert created_batch.object == "batch"
        assert created_batch.endpoint == sample_batch_data["endpoint"]
        assert created_batch.input_file_id == sample_batch_data["input_file_id"]
        assert created_batch.completion_window == sample_batch_data["completion_window"]
        assert created_batch.status == "validating"
        assert created_batch.metadata == sample_batch_data["metadata"]
        assert isinstance(created_batch.created_at, int)
        assert created_batch.created_at > 0

        retrieved_batch = await provider.retrieve_batch(RetrieveBatchRequest(batch_id=created_batch.id))

        self._validate_batch_type(retrieved_batch, expected_metadata=sample_batch_data["metadata"])

        assert retrieved_batch.id == created_batch.id
        assert retrieved_batch.input_file_id == created_batch.input_file_id
        assert retrieved_batch.endpoint == created_batch.endpoint
        assert retrieved_batch.status == created_batch.status
        assert retrieved_batch.metadata == created_batch.metadata

    async def test_create_batch_without_metadata(self, provider):
        """Test batch creation without optional metadata."""
        batch = await provider.create_batch(
            CreateBatchRequest(input_file_id="file_123", endpoint="/v1/chat/completions", completion_window="24h")
        )

        assert batch.metadata is None

    async def test_create_batch_completion_window(self, provider):
        """Test batch creation with invalid completion window."""
        with pytest.raises(ValidationError, match="completion_window"):
            CreateBatchRequest(input_file_id="file_123", endpoint="/v1/chat/completions", completion_window="now")

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/v1/invalid/endpoint",
            "",
        ],
    )
    async def test_create_batch_invalid_endpoints(self, provider, endpoint):
        """Test batch creation with various invalid endpoints."""
        with pytest.raises(ValueError, match="Invalid endpoint"):
            await provider.create_batch(
                CreateBatchRequest(input_file_id="file_123", endpoint=endpoint, completion_window="24h")
            )

    async def test_create_batch_invalid_metadata(self, provider):
        """Test that batch creation fails with invalid metadata."""
        with pytest.raises(ValueError, match="should be a valid string"):
            await provider.create_batch(
                CreateBatchRequest(
                    input_file_id="file_123",
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={123: "invalid_key"},  # Non-string key
                )
            )

        with pytest.raises(ValueError, match="should be a valid string"):
            await provider.create_batch(
                CreateBatchRequest(
                    input_file_id="file_123",
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={"valid_key": 456},  # Non-string value
                )
            )

    async def test_retrieve_batch_not_found(self, provider):
        """Test error when retrieving non-existent batch."""
        with pytest.raises(ResourceNotFoundError, match=r"Batch 'nonexistent_batch' not found"):
            await provider.retrieve_batch(RetrieveBatchRequest(batch_id="nonexistent_batch"))

    async def test_cancel_batch_success(self, provider, sample_batch_data):
        """Test successful batch cancellation."""
        created_batch = await provider.create_batch(CreateBatchRequest(**sample_batch_data))
        assert created_batch.status == "validating"

        cancelled_batch = await provider.cancel_batch(CancelBatchRequest(batch_id=created_batch.id))

        assert cancelled_batch.id == created_batch.id
        assert cancelled_batch.status in ["cancelling", "cancelled"]
        assert isinstance(cancelled_batch.cancelling_at, int)
        assert cancelled_batch.cancelling_at >= created_batch.created_at

    @pytest.mark.parametrize("status", ["failed", "expired", "completed"])
    async def test_cancel_batch_invalid_statuses(self, provider, sample_batch_data, status):
        """Test error when cancelling batch in final states."""
        provider.process_batches = False
        created_batch = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        # directly update status in kvstore
        await provider._update_batch(created_batch.id, status=status)

        with pytest.raises(ConflictError, match=f"Cannot cancel batch '{created_batch.id}' with status '{status}'"):
            await provider.cancel_batch(CancelBatchRequest(batch_id=created_batch.id))

    async def test_cancel_batch_not_found(self, provider):
        """Test error when cancelling non-existent batch."""
        with pytest.raises(ResourceNotFoundError, match=r"Batch 'nonexistent_batch' not found"):
            await provider.cancel_batch(CancelBatchRequest(batch_id="nonexistent_batch"))

    async def test_list_batches_empty(self, provider):
        """Test listing batches when none exist."""
        response = await provider.list_batches(ListBatchesRequest())

        assert response.object == "list"
        assert response.data == []
        assert response.first_id is None
        assert response.last_id is None
        assert response.has_more is False

    async def test_list_batches_single_batch(self, provider, sample_batch_data):
        """Test listing batches with single batch."""
        created_batch = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        response = await provider.list_batches(ListBatchesRequest())

        assert len(response.data) == 1
        self._validate_batch_type(response.data[0], expected_metadata=sample_batch_data["metadata"])
        assert response.data[0].id == created_batch.id
        assert response.first_id == created_batch.id
        assert response.last_id == created_batch.id
        assert response.has_more is False

    async def test_list_batches_multiple_batches(self, provider):
        """Test listing multiple batches."""
        batches = [
            await provider.create_batch(
                CreateBatchRequest(input_file_id=f"file_{i}", endpoint="/v1/chat/completions", completion_window="24h")
            )
            for i in range(3)
        ]

        response = await provider.list_batches(ListBatchesRequest())

        assert len(response.data) == 3

        batch_ids = {batch.id for batch in response.data}
        expected_ids = {batch.id for batch in batches}
        assert batch_ids == expected_ids
        assert response.has_more is False

        assert response.first_id in expected_ids
        assert response.last_id in expected_ids

    async def test_list_batches_with_limit(self, provider):
        """Test listing batches with limit parameter."""
        batches = [
            await provider.create_batch(
                CreateBatchRequest(input_file_id=f"file_{i}", endpoint="/v1/chat/completions", completion_window="24h")
            )
            for i in range(3)
        ]

        response = await provider.list_batches(ListBatchesRequest(limit=2))

        assert len(response.data) == 2
        assert response.has_more is True
        assert response.first_id == response.data[0].id
        assert response.last_id == response.data[1].id
        batch_ids = {batch.id for batch in response.data}
        expected_ids = {batch.id for batch in batches}
        assert batch_ids.issubset(expected_ids)

    async def test_list_batches_with_pagination(self, provider):
        """Test listing batches with pagination using 'after' parameter."""
        for i in range(3):
            await provider.create_batch(
                CreateBatchRequest(input_file_id=f"file_{i}", endpoint="/v1/chat/completions", completion_window="24h")
            )

        # Get first page
        first_page = await provider.list_batches(ListBatchesRequest(limit=1))
        assert len(first_page.data) == 1
        assert first_page.has_more is True

        # Get second page using 'after'
        second_page = await provider.list_batches(ListBatchesRequest(limit=1, after=first_page.data[0].id))
        assert len(second_page.data) == 1
        assert second_page.data[0].id != first_page.data[0].id

        # Verify we got the next batch in order
        all_batches = await provider.list_batches(ListBatchesRequest())
        expected_second_batch_id = all_batches.data[1].id
        assert second_page.data[0].id == expected_second_batch_id

    async def test_list_batches_invalid_after(self, provider, sample_batch_data):
        """Test listing batches with invalid 'after' parameter."""
        await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        response = await provider.list_batches(ListBatchesRequest(after="nonexistent_batch"))

        # Should return all batches (no filtering when 'after' batch not found)
        assert len(response.data) == 1

    async def test_kvstore_persistence(self, provider, sample_batch_data):
        """Test that batches are properly persisted in kvstore."""
        batch = await provider.create_batch(CreateBatchRequest(**sample_batch_data))

        stored_data = await provider.kvstore.get(f"batch:{batch.id}")
        assert stored_data is not None

        stored_batch_dict = json.loads(stored_data)
        assert stored_batch_dict["id"] == batch.id
        assert stored_batch_dict["input_file_id"] == sample_batch_data["input_file_id"]

    async def test_validate_input_file_not_found(self, provider):
        """Test _validate_input when input file does not exist."""
        provider.files_api.openai_retrieve_file = AsyncMock(side_effect=Exception("File not found"))

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="nonexistent_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0
        assert errors[0].code == "invalid_request"
        assert errors[0].message == "Cannot find file nonexistent_file."
        assert errors[0].param == "input_file_id"
        assert errors[0].line is None

    async def test_validate_input_file_exists_empty_content(self, provider):
        """Test _validate_input when file exists but is empty."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        mock_response.body = b""
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="empty_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 0
        assert len(requests) == 0

    async def test_validate_input_file_mixed_valid_invalid_json(self, provider):
        """Test _validate_input when file contains valid and invalid JSON lines."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        # Line 1: valid JSON with proper body args, Line 2: invalid JSON
        mock_response.body = b'{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}}\n{invalid json'
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="mixed_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        # Should have 1 JSON parsing error from line 2, and 1 valid request from line 1
        assert len(errors) == 1
        assert len(requests) == 1

        assert errors[0].code == "invalid_json_line"
        assert errors[0].line == 2
        assert errors[0].message == "This line is not parseable as valid JSON."

        assert requests[0].custom_id == "req-1"
        assert requests[0].method == "POST"
        assert requests[0].url == "/v1/chat/completions"
        assert requests[0].body["model"] == "test-model"
        assert requests[0].body["messages"] == [{"role": "user", "content": "Hello"}]

    async def test_validate_input_invalid_model(self, provider):
        """Test _validate_input when file contains request with non-existent model."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        mock_response.body = b'{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "nonexistent-model", "messages": [{"role": "user", "content": "Hello"}]}}'
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        provider.models_api.get_model = AsyncMock(side_effect=Exception("Model not found"))

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="invalid_model_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == "model_not_found"
        assert errors[0].line == 1
        assert errors[0].message == "Model 'nonexistent-model' does not exist or is not supported"
        assert errors[0].param == "body.model"

    @pytest.mark.parametrize(
        "param_name,param_path,error_code,error_message",
        [
            ("custom_id", "custom_id", "missing_required_parameter", "Missing required parameter: custom_id"),
            ("method", "method", "missing_required_parameter", "Missing required parameter: method"),
            ("url", "url", "missing_required_parameter", "Missing required parameter: url"),
            ("body", "body", "missing_required_parameter", "Missing required parameter: body"),
            ("model", "body.model", "invalid_request", "Model parameter is required"),
            ("messages", "body.messages", "invalid_request", "Messages parameter is required"),
        ],
    )
    async def test_validate_input_missing_parameters_chat_completions(
        self, provider, param_name, param_path, error_code, error_message
    ):
        """Test _validate_input when file contains request with missing required parameters for chat completions."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()

        base_request = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]},
        }

        # Remove the specific parameter being tested
        if "." in param_path:
            top_level, nested_param = param_path.split(".", 1)
            del base_request[top_level][nested_param]
        else:
            del base_request[param_name]

        mock_response.body = json.dumps(base_request).encode()
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id=f"missing_{param_name}_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == error_code
        assert errors[0].line == 1
        assert errors[0].message == error_message
        assert errors[0].param == param_path

    @pytest.mark.parametrize(
        "param_name,param_path,error_code,error_message",
        [
            ("custom_id", "custom_id", "missing_required_parameter", "Missing required parameter: custom_id"),
            ("method", "method", "missing_required_parameter", "Missing required parameter: method"),
            ("url", "url", "missing_required_parameter", "Missing required parameter: url"),
            ("body", "body", "missing_required_parameter", "Missing required parameter: body"),
            ("model", "body.model", "invalid_request", "Model parameter is required"),
            ("prompt", "body.prompt", "invalid_request", "Prompt parameter is required"),
        ],
    )
    async def test_validate_input_missing_parameters_completions(
        self, provider, param_name, param_path, error_code, error_message
    ):
        """Test _validate_input when file contains request with missing required parameters for text completions."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()

        base_request = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/completions",
            "body": {"model": "test-model", "prompt": "Hello"},
        }

        # Remove the specific parameter being tested
        if "." in param_path:
            top_level, nested_param = param_path.split(".", 1)
            del base_request[top_level][nested_param]
        else:
            del base_request[param_name]

        mock_response.body = json.dumps(base_request).encode()
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/completions",
            input_file_id=f"missing_{param_name}_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == error_code
        assert errors[0].line == 1
        assert errors[0].message == error_message
        assert errors[0].param == param_path

    async def test_validate_input_url_mismatch(self, provider):
        """Test _validate_input when file contains request with URL that doesn't match batch endpoint."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        mock_response.body = b'{"custom_id": "req-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}}'
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",  # This doesn't match the URL in the request
            input_file_id="url_mismatch_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == "invalid_url"
        assert errors[0].line == 1
        assert errors[0].message == "URL provided for this request does not match the batch endpoint"
        assert errors[0].param == "url"

    async def test_validate_input_multiple_errors_per_request(self, provider):
        """Test _validate_input when a single request has multiple validation errors."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        # Request missing custom_id, has invalid URL, and missing model in body
        mock_response.body = (
            b'{"method": "POST", "url": "/v1/embeddings", "body": {"messages": [{"role": "user", "content": "Hello"}]}}'
        )
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",  # Doesn't match /v1/embeddings in request
            input_file_id="multiple_errors_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) >= 2  # At least missing custom_id and URL mismatch
        assert len(requests) == 0

        for error in errors:
            assert error.line == 1

        error_codes = {error.code for error in errors}
        assert "missing_required_parameter" in error_codes  # missing custom_id
        assert "invalid_url" in error_codes  # URL mismatch

    async def test_validate_input_invalid_request_format(self, provider):
        """Test _validate_input when file contains non-object JSON (array, string, number)."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()
        mock_response.body = b'["not", "a", "request", "object"]'
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id="invalid_format_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == "invalid_request"
        assert errors[0].line == 1
        assert errors[0].message == "Each line must be a JSON dictionary object"

    @pytest.mark.parametrize(
        "param_name,param_path,invalid_value,error_message",
        [
            ("custom_id", "custom_id", 12345, "Custom_id must be a string"),
            ("url", "url", 123, "URL must be a string"),
            ("method", "method", ["POST"], "Method must be a string"),
            ("body", "body", ["not", "valid"], "Body must be a JSON dictionary object"),
            ("model", "body.model", 123, "Model must be a string"),
            ("messages", "body.messages", "invalid messages format", "Messages must be an array"),
        ],
    )
    async def test_validate_input_invalid_parameter_types(
        self, provider, param_name, param_path, invalid_value, error_message
    ):
        """Test _validate_input when file contains request with parameters that have invalid types."""
        provider.files_api.openai_retrieve_file = AsyncMock()
        mock_response = MagicMock()

        base_request = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]},
        }

        # Override the specific parameter with invalid value
        if "." in param_path:
            top_level, nested_param = param_path.split(".", 1)
            base_request[top_level][nested_param] = invalid_value
        else:
            base_request[param_name] = invalid_value

        mock_response.body = json.dumps(base_request).encode()
        provider.files_api.openai_retrieve_file_content = AsyncMock(return_value=mock_response)

        batch = BatchObject(
            id="batch_test",
            object="batch",
            endpoint="/v1/chat/completions",
            input_file_id=f"invalid_{param_name}_type_file",
            completion_window="24h",
            status="validating",
            created_at=1234567890,
        )

        errors, requests = await provider._validate_input(batch)

        assert len(errors) == 1
        assert len(requests) == 0

        assert errors[0].code == "invalid_request"
        assert errors[0].line == 1
        assert errors[0].message == error_message
        assert errors[0].param == param_path

    async def test_max_concurrent_batches(self, provider):
        """Test max_concurrent_batches configuration and concurrency control."""
        import asyncio

        provider._batch_semaphore = asyncio.Semaphore(2)

        provider.process_batches = True  # enable because we're testing background processing

        active_batches = 0

        async def add_and_wait(batch_id: str):
            nonlocal active_batches
            active_batches += 1
            await asyncio.sleep(float("inf"))

        # the first thing done in _process_batch is to acquire the semaphore, then call _process_batch_impl,
        # so we can replace _process_batch_impl with our mock to control concurrency
        provider._process_batch_impl = add_and_wait

        for _ in range(3):
            await provider.create_batch(
                CreateBatchRequest(input_file_id="file_id", endpoint="/v1/chat/completions", completion_window="24h")
            )

        await asyncio.sleep(0.042)  # let tasks start

        assert active_batches == 2, f"Expected 2 active batches, got {active_batches}"

    async def test_create_batch_embeddings_endpoint(self, provider):
        """Test that batch creation succeeds with embeddings endpoint."""
        batch = await provider.create_batch(
            CreateBatchRequest(
                input_file_id="file_123",
                endpoint="/v1/embeddings",
                completion_window="24h",
            )
        )
        assert batch.endpoint == "/v1/embeddings"

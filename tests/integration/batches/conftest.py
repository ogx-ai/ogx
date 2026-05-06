# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared pytest fixtures for batch tests."""

import json
import time
import warnings
from contextlib import contextmanager
from io import BytesIO

import pytest

from ogx_api import OpenAIFilePurpose


class BatchHelper:
    """Helper class for creating and managing batch input files."""

    def __init__(self, client):
        """Initialize with either a batch_client or openai_client."""
        self.client = client

    @contextmanager
    def create_file(self, content: str | list[dict], filename_prefix="batch_input"):
        """Context manager for creating and cleaning up batch input files.

        Args:
            content: Either a list of batch request dictionaries or raw string content
            filename_prefix: Prefix for the generated filename (or full filename if content is string)

        Yields:
            The uploaded file object
        """
        if isinstance(content, str):
            # Handle raw string content (e.g., malformed JSONL, empty files)
            file_content = content.encode("utf-8")
        else:
            # Handle list of batch request dictionaries
            jsonl_content = "\n".join(json.dumps(req) for req in content)
            file_content = jsonl_content.encode("utf-8")

        filename = filename_prefix if filename_prefix.endswith(".jsonl") else f"{filename_prefix}.jsonl"

        with BytesIO(file_content) as file_buffer:
            file_buffer.name = filename
            uploaded_file = self.client.files.create(file=file_buffer, purpose=OpenAIFilePurpose.BATCH)

        try:
            yield uploaded_file
        finally:
            try:
                self.client.files.delete(uploaded_file.id)
            except Exception:
                warnings.warn(
                    f"Failed to cleanup file {uploaded_file.id}: {uploaded_file.filename}",
                    stacklevel=2,
                )

    def wait_for(
        self,
        batch_id: str,
        max_wait_time: int = 60,
        sleep_interval: int | None = None,
        expected_statuses: set[str] | None = None,
        timeout_action: str = "fail",
    ):
        """Wait for a batch to reach a terminal status.

        Uses exponential backoff polling strategy for efficient waiting:
        - Starts with short intervals (0.1s) for fast batches (e.g., replay mode)
        - Doubles interval each iteration up to a maximum
        - Adapts automatically to both fast and slow batch processing

        Args:
            batch_id: The batch ID to monitor
            max_wait_time: Maximum time to wait in seconds (default: 60 seconds)
            sleep_interval: If provided, uses fixed interval instead of exponential backoff
            expected_statuses: Set of expected terminal statuses (default: {"completed"})
            timeout_action: Action on timeout - "fail" (pytest.fail) or "skip" (pytest.skip)

        Returns:
            The final batch object

        Raises:
            pytest.Failed: If batch reaches an unexpected status or timeout_action is "fail"
            pytest.Skipped: If timeout_action is "skip" on timeout or unexpected status
        """
        if expected_statuses is None:
            expected_statuses = {"completed"}

        terminal_statuses = {"completed", "failed", "cancelled", "expired"}
        unexpected_statuses = terminal_statuses - expected_statuses

        start_time = time.time()

        # Use exponential backoff if no explicit sleep_interval provided
        if sleep_interval is None:
            current_interval = 0.1  # Start with 100ms
            max_interval = 10.0  # Cap at 10 seconds
        else:
            current_interval = sleep_interval
            max_interval = sleep_interval

        while time.time() - start_time < max_wait_time:
            current_batch = self.client.batches.retrieve(batch_id)

            if current_batch.status in expected_statuses:
                return current_batch
            elif current_batch.status in unexpected_statuses:
                error_msg = f"Batch reached unexpected status: {current_batch.status}"
                if timeout_action == "skip":
                    pytest.skip(error_msg)
                else:
                    pytest.fail(error_msg)

            time.sleep(current_interval)

            # Exponential backoff: double the interval each time, up to max
            if sleep_interval is None:
                current_interval = min(current_interval * 2, max_interval)

        timeout_msg = f"Batch did not reach expected status {expected_statuses} within {max_wait_time} seconds"
        if timeout_action == "skip":
            pytest.skip(timeout_msg)
        else:
            pytest.fail(timeout_msg)


@pytest.fixture
def batch_helper(openai_client):
    """Fixture that provides a BatchHelper instance for OpenAI client."""
    return BatchHelper(openai_client)

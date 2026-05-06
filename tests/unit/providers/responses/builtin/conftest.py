# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from ogx.providers.inline.responses.builtin.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from ogx.providers.utils.responses.responses_store import (
    ResponsesStore,
)
from ogx_api import Connectors
from ogx_api.tools import ToolGroups, ToolRuntime


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_tool_groups_api():
    tool_groups_api = AsyncMock(spec=ToolGroups)
    return tool_groups_api


@pytest.fixture
def mock_tool_runtime_api():
    tool_runtime_api = AsyncMock(spec=ToolRuntime)
    return tool_runtime_api


@pytest.fixture
def mock_responses_store():
    responses_store = AsyncMock(spec=ResponsesStore)
    return responses_store


@pytest.fixture
def mock_vector_io_api():
    vector_io_api = AsyncMock()
    return vector_io_api


@pytest.fixture
def mock_conversations_api():
    """Mock conversations API for testing."""
    mock_api = AsyncMock()
    return mock_api


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
    return safety_api


@pytest.fixture
def mock_prompts_api():
    prompts_api = AsyncMock()
    return prompts_api


@pytest.fixture
def mock_files_api():
    """Mock files API for testing."""
    files_api = AsyncMock()
    return files_api


@pytest.fixture
def mock_connectors_api():
    connectors_api = AsyncMock(spec=Connectors)
    return connectors_api


@pytest.fixture
def openai_responses_impl(
    mock_inference_api,
    mock_tool_groups_api,
    mock_tool_runtime_api,
    mock_responses_store,
    mock_vector_io_api,
    mock_safety_api,
    mock_conversations_api,
    mock_prompts_api,
    mock_files_api,
    mock_connectors_api,
):
    return OpenAIResponsesImpl(
        inference_api=mock_inference_api,
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
        responses_store=mock_responses_store,
        vector_io_api=mock_vector_io_api,
        safety_api=mock_safety_api,
        conversations_api=mock_conversations_api,
        prompts_api=mock_prompts_api,
        files_api=mock_files_api,
        connectors_api=mock_connectors_api,
    )

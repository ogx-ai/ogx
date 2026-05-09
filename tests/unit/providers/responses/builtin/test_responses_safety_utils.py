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
from ogx.providers.inline.responses.builtin.responses.utils import (
    extract_guardrail_ids,
    run_guardrails,
)
from ogx_api.responses import ResponseGuardrailSpec
from ogx_api.safety import ModerationObject, ModerationObjectResults


@pytest.fixture
def mock_apis():
    """Create mock APIs for testing."""
    return {
        "inference_api": AsyncMock(),
        "tool_groups_api": AsyncMock(),
        "tool_runtime_api": AsyncMock(),
        "responses_store": AsyncMock(),
        "vector_io_api": AsyncMock(),
        "conversations_api": AsyncMock(),
        "safety_api": AsyncMock(),
        "prompts_api": AsyncMock(),
        "files_api": AsyncMock(),
        "connectors_api": AsyncMock(),
    }


@pytest.fixture
def responses_impl(mock_apis):
    """Create OpenAIResponsesImpl instance with mocked dependencies."""
    return OpenAIResponsesImpl(**mock_apis)


def test_extract_guardrail_ids_from_strings(responses_impl):
    """Test extraction from simple string guardrail IDs."""
    guardrails = ["llama-guard", "content-filter", "nsfw-detector"]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_guardrail_ids_from_objects(responses_impl):
    """Test extraction from ResponseGuardrailSpec objects."""
    guardrails = [
        ResponseGuardrailSpec(type="llama-guard"),
        ResponseGuardrailSpec(type="content-filter"),
    ]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter"]


def test_extract_guardrail_ids_mixed_formats(responses_impl):
    """Test extraction from mixed string and object formats."""
    guardrails = [
        "llama-guard",
        ResponseGuardrailSpec(type="content-filter"),
        "nsfw-detector",
    ]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_guardrail_ids_none_input(responses_impl):
    """Test extraction with None input."""
    result = extract_guardrail_ids(None)
    assert result == []


def test_extract_guardrail_ids_empty_list(responses_impl):
    """Test extraction with empty list."""
    result = extract_guardrail_ids([])
    assert result == []


def test_extract_guardrail_ids_unknown_format(responses_impl):
    """Test extraction with unknown guardrail format raises ValueError."""
    # Create an object that's neither string nor ResponseGuardrailSpec
    unknown_object = {"invalid": "format"}  # Plain dict, not ResponseGuardrailSpec
    guardrails = ["valid-guardrail", unknown_object, "another-guardrail"]
    with pytest.raises(ValueError, match="Unknown guardrail format.*expected str or ResponseGuardrailSpec"):
        extract_guardrail_ids(guardrails)


@pytest.fixture
def mock_safety_api():
    """Create mock safety API for guardrails testing."""
    safety_api = AsyncMock()
    # Mock the routing table and shields list for guardrails lookup
    safety_api.routing_table = AsyncMock()
    shield = AsyncMock()
    shield.identifier = "llama-guard"
    shield.provider_resource_id = "llama-guard-model"
    safety_api.routing_table.list_shields.return_value = AsyncMock(data=[shield])
    return safety_api


async def test_run_guardrails_no_violation(mock_safety_api):
    """Test guardrails validation with no violations."""
    text = "Hello world"
    guardrail_ids = ["llama-guard"]

    # Mock moderation to return non-flagged content
    unflagged_result = ModerationObjectResults(flagged=False, categories={"violence": False})
    mock_moderation_object = ModerationObject(id="test-mod-id", model="llama-guard-model", results=[unflagged_result])
    mock_safety_api.run_moderation.return_value = mock_moderation_object

    result = await run_guardrails(mock_safety_api, text, guardrail_ids)

    assert result is None
    # Verify run_moderation was called with the correct request object
    mock_safety_api.run_moderation.assert_called_once()
    call_args = mock_safety_api.run_moderation.call_args
    request = call_args[0][0]  # First positional argument is the RunModerationRequest
    assert request.model == "llama-guard-model"
    assert request.input == text


async def test_run_guardrails_with_violation(mock_safety_api):
    """Test guardrails validation with safety violation."""
    text = "Harmful content"
    guardrail_ids = ["llama-guard"]

    # Mock moderation to return flagged content
    flagged_result = ModerationObjectResults(
        flagged=True,
        categories={"violence": True},
        user_message="Content flagged by moderation",
        metadata={"violation_type": ["S1"]},
    )
    mock_moderation_object = ModerationObject(id="test-mod-id", model="llama-guard-model", results=[flagged_result])
    mock_safety_api.run_moderation.return_value = mock_moderation_object

    result = await run_guardrails(mock_safety_api, text, guardrail_ids)

    assert result == "Content flagged by moderation (flagged for: violence) (violation type: S1)"


async def test_run_guardrails_empty_inputs(mock_safety_api):
    """Test guardrails validation with empty inputs."""
    # Test empty guardrail_ids
    result = await run_guardrails(mock_safety_api, "test", [])
    assert result is None

    # Test empty text
    result = await run_guardrails(mock_safety_api, "", ["llama-guard"])
    assert result is None

    # Test both empty
    result = await run_guardrails(mock_safety_api, "", [])
    assert result is None

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Validate that our Interactions API response shapes match Google's real API.

These tests compare our Pydantic model structure against fixture files captured
from the real Google Interactions API. If Google changes their response format,
re-run scripts/capture_google_interactions_fixtures.py to update the fixtures,
then fix any failing tests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ogx.providers.inline.interactions.config import InteractionsConfig
from ogx.providers.inline.interactions.impl import BuiltinInteractionsImpl

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict | list:
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(
            f"Fixture {name} not found. Run: "
            "GEMINI_API_KEY=<key> uv run scripts/capture_google_interactions_fixtures.py"
        )
    return json.loads(path.read_text())


class TestNonStreamingShape:
    """Verify our non-streaming response has the same top-level fields as Google's."""

    def test_top_level_fields_present(self):
        google = _load_fixture("google_non_streaming.json")
        expected_fields = {"id", "status", "outputs", "usage", "created", "updated", "model", "role", "object"}
        actual_fields = set(google.keys())
        missing = expected_fields - actual_fields
        assert not missing, f"Expected fields missing from Google fixture: {missing}"

    def test_our_response_has_required_google_fields(self):
        _load_fixture("google_non_streaming.json")

        impl = BuiltinInteractionsImpl(config=InteractionsConfig(), inference_api=MagicMock())
        openai_resp = MagicMock()
        openai_resp.choices = [MagicMock()]
        openai_resp.choices[0].message = MagicMock()
        openai_resp.choices[0].message.content = "4"
        openai_resp.choices[0].finish_reason = "stop"
        openai_resp.usage = MagicMock()
        openai_resp.usage.prompt_tokens = 10
        openai_resp.usage.completion_tokens = 1

        result = impl._openai_to_google(openai_resp, "test-model")
        our_fields = set(result.model_dump(exclude_none=True).keys())

        required_fields = {"id", "status", "outputs", "usage", "model", "role", "object"}
        missing = required_fields - our_fields
        assert not missing, f"Our response missing fields that Google returns: {missing}"

    def test_usage_field_names_match(self):
        google = _load_fixture("google_non_streaming.json")
        google_usage = google.get("usage", {})

        assert "total_input_tokens" in google_usage, "Google usage should have total_input_tokens"
        assert "total_output_tokens" in google_usage, "Google usage should have total_output_tokens"
        assert "total_tokens" in google_usage, "Google usage should have total_tokens"

    def test_output_structure_matches(self):
        google = _load_fixture("google_non_streaming.json")
        text_outputs = [o for o in google.get("outputs", []) if o.get("type") == "text"]
        assert len(text_outputs) > 0, "Google fixture should have text outputs"
        assert "text" in text_outputs[0], "Text output should have 'text' field"

    def test_status_value(self):
        google = _load_fixture("google_non_streaming.json")
        assert google["status"] == "completed"

    def test_object_value(self):
        google = _load_fixture("google_non_streaming.json")
        assert google["object"] == "interaction"

    def test_role_value(self):
        google = _load_fixture("google_non_streaming.json")
        assert google["role"] == "model"


class TestStreamingShape:
    """Verify our streaming events have the same structure as Google's."""

    def test_event_sequence(self):
        events = _load_fixture("google_streaming.json")
        event_types = [e["event_type"] for e in events]

        assert "InteractionStartEvent" in event_types, f"Missing InteractionStartEvent in {event_types}"
        assert "InteractionCompleteEvent" in event_types, f"Missing InteractionCompleteEvent in {event_types}"
        assert "ContentStart" in event_types, f"Missing ContentStart in {event_types}"
        assert "ContentDelta" in event_types, f"Missing ContentDelta in {event_types}"
        assert "ContentStop" in event_types, f"Missing ContentStop in {event_types}"

    def test_interaction_start_wraps_in_interaction_object(self):
        events = _load_fixture("google_streaming.json")
        start = next(e for e in events if e["event_type"] == "InteractionStartEvent")
        data = start["data"]

        assert "interaction" in data, "interaction.start should wrap data in 'interaction' object"
        assert "id" in data["interaction"]
        assert "status" in data["interaction"]
        assert data["interaction"]["object"] == "interaction"

    def test_interaction_complete_wraps_in_interaction_object(self):
        events = _load_fixture("google_streaming.json")
        complete = next(e for e in events if e["event_type"] == "InteractionCompleteEvent")
        data = complete["data"]

        assert "interaction" in data, "interaction.complete should wrap data in 'interaction' object"
        assert data["interaction"]["status"] == "completed"
        assert "usage" in data["interaction"]
        assert data["interaction"]["object"] == "interaction"

    def test_content_start_wraps_type_in_content_object(self):
        events = _load_fixture("google_streaming.json")
        text_starts = [
            e
            for e in events
            if e["event_type"] == "ContentStart" and e["data"].get("content", {}).get("type") == "text"
        ]
        assert len(text_starts) > 0, "Should have at least one text content.start event"
        assert "content" in text_starts[0]["data"], "content.start should wrap type in 'content' object"

    def test_content_delta_text_structure(self):
        events = _load_fixture("google_streaming.json")
        text_deltas = [
            e for e in events if e["event_type"] == "ContentDelta" and e["data"].get("delta", {}).get("type") == "text"
        ]
        assert len(text_deltas) > 0, "Should have text content.delta events"
        assert "text" in text_deltas[0]["data"]["delta"], "Text delta should have 'text' field"

    def test_streaming_usage_field_names(self):
        events = _load_fixture("google_streaming.json")
        complete = next(e for e in events if e["event_type"] == "InteractionCompleteEvent")
        usage = complete["data"]["interaction"]["usage"]

        assert "total_input_tokens" in usage, "Streaming usage should have total_input_tokens"
        assert "total_output_tokens" in usage, "Streaming usage should have total_output_tokens"
        assert "total_tokens" in usage, "Streaming usage should have total_tokens"

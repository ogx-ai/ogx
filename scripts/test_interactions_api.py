#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# /// script
# dependencies = [
#   "google-genai",
# ]
# ///

"""Test plan script for the Google Interactions API front-end.

This script validates the Interactions API endpoint against a running
OGX server using the official Google GenAI SDK, proving that
ADK/Gemini ecosystem clients can call OGX natively.

Usage:
    # Start a OGX server first:
    OLLAMA_URL=http://localhost:11434/v1 uv run --extra starter ogx stack run starter --port 8321

    # Then run this script:
    uv run python scripts/test_interactions_api.py --base-url http://localhost:8321 --model ollama/llama3.2:3b

    # Or with OpenAI:
    uv run python scripts/test_interactions_api.py --base-url http://localhost:8321 --model openai/gpt-4o-mini
"""

import argparse
import sys

from google import genai
from google.genai import types


def _create_client(base_url: str) -> genai.Client:
    """Create a Google GenAI client pointed at the OGX server."""
    return genai.Client(
        api_key="no-key-required",
        http_options=types.HttpOptions(
            base_url=base_url,
            api_version="v1alpha",
        ),
    )


def run_non_streaming_basic(client: genai.Client, model: str) -> None:
    """Test 1: Basic non-streaming interaction."""
    print("Test 1: Non-streaming basic interaction...")

    interaction = client.interactions.create(
        model=model,
        input="What is 2+2? Reply with just the number.",
    )

    assert len(interaction.id) > 0, f"ID should not be empty, got: {interaction.id}"
    assert interaction.status == "completed", f"Status should be 'completed', got: {interaction.status}"
    assert len(interaction.outputs) > 0, "Expected at least one output"
    assert interaction.outputs[0].type == "text", f"Output type should be 'text', got: {interaction.outputs[0].type}"
    assert len(interaction.outputs[0].text) > 0, "Output text should not be empty"
    assert interaction.usage.total_input_tokens > 0, "Expected input_tokens > 0"
    assert interaction.usage.total_output_tokens > 0, "Expected output_tokens > 0"
    assert (
        interaction.usage.total_tokens == interaction.usage.total_input_tokens + interaction.usage.total_output_tokens
    )

    print(f"  Response: {interaction.outputs[0].text[:80]}")
    print(
        f"  Usage: input={interaction.usage.total_input_tokens}, output={interaction.usage.total_output_tokens}, total={interaction.usage.total_tokens}"
    )
    print("  PASSED")


def run_non_streaming_system_instruction(client: genai.Client, model: str) -> None:
    """Test 2: Non-streaming with system instruction."""
    print("Test 2: Non-streaming with system instruction...")

    interaction = client.interactions.create(
        model=model,
        input="What are you?",
        system_instruction="You are a pirate. Always respond in pirate speak. Keep it short.",
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0
    assert len(interaction.outputs[0].text) > 0

    print(f"  Response: {interaction.outputs[0].text[:100]}")
    print("  PASSED")


def run_non_streaming_multi_turn(client: genai.Client, model: str) -> None:
    """Test 3: Multi-turn conversation with 'model' role."""
    print("Test 3: Multi-turn conversation...")

    interaction = client.interactions.create(
        model=model,
        input=[
            {"role": "user", "content": [{"type": "text", "text": "My name is Alice."}]},
            {"role": "model", "content": [{"type": "text", "text": "Hello Alice! Nice to meet you."}]},
            {"role": "user", "content": [{"type": "text", "text": "What is my name?"}]},
        ],
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0

    text = interaction.outputs[0].text.lower()
    assert "alice" in text, f"Expected 'alice' in response, got: {interaction.outputs[0].text}"

    print(f"  Response: {interaction.outputs[0].text[:100]}")
    print("  PASSED")


def run_non_streaming_generation_config(client: genai.Client, model: str) -> None:
    """Test 4: Non-streaming with generation_config parameters."""
    print("Test 4: Generation config (temperature, max_output_tokens)...")

    interaction = client.interactions.create(
        model=model,
        input="Say hello.",
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 32,
        },
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0

    print(f"  Response: {interaction.outputs[0].text[:80]}")
    print("  PASSED")


def run_tool_calling(client: genai.Client, model: str) -> None:
    """Test 5: Tool calling - function declaration, function_call output, function_response input."""
    print("Test 5: Tool calling round-trip...")

    # Step 1: Send request with tools, expect a function_call output
    interaction = client.interactions.create(
        model=model,
        input="What is the weather in New York City? Use the get_weather tool.",
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "The city name."},
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
    )

    assert interaction.status == "completed"
    assert len(interaction.outputs) > 0

    # The model should have produced a function_call output
    function_calls = [o for o in interaction.outputs if o.type == "function_call"]
    print(f"  Outputs: {[o.type for o in interaction.outputs]}")

    if not function_calls:
        # Some models may just answer directly without calling the tool
        print("  Note: Model did not call the tool (answered directly). Skipping round-trip.")
        print("  PASSED (partial - no function_call)")
        return

    fc = function_calls[0]
    print(f"  Function call: {fc.name}({fc.args})")
    assert fc.name == "get_weather", f"Expected function name 'get_weather', got: {fc.name}"

    # Step 2: Send the function result back and get a final text response
    interaction2 = client.interactions.create(
        model=model,
        input=[
            {"role": "user", "content": [{"type": "text", "text": "What is the weather in New York City?"}]},
            {
                "role": "model",
                "content": [
                    {
                        "type": "function_call",
                        "id": fc.id,
                        "name": fc.name,
                        "args": fc.args,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "function_result",
                        "id": fc.id,
                        "name": fc.name,
                        "result": {"temperature": 72, "condition": "Sunny"},
                    }
                ],
            },
        ],
        tools=[
            {
                "function_declarations": [
                    {
                        "name": "get_weather",
                        "description": "Get the current weather for a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "The city name."},
                            },
                            "required": ["location"],
                        },
                    }
                ]
            }
        ],
    )

    assert interaction2.status == "completed"
    assert len(interaction2.outputs) > 0

    text_outputs = [o for o in interaction2.outputs if o.type == "text"]
    assert len(text_outputs) > 0, "Expected a text response after providing function result"
    print(f"  Final response: {text_outputs[0].text[:100]}")
    print("  PASSED")


def run_streaming_basic(client: genai.Client, model: str) -> None:
    """Test 6: Streaming interaction with SSE events."""
    print("Test 6: Streaming basic interaction...")

    stream = client.interactions.create(
        model=model,
        input="Count from 1 to 5, separated by commas.",
        stream=True,
    )

    event_types = []
    text_parts = []
    interaction_id = None
    complete_event = None

    for event in stream:
        event_type = type(event).__name__
        event_types.append(event_type)

        if hasattr(event, "interaction") and event.interaction and hasattr(event.interaction, "id"):
            interaction_id = event.interaction.id

        if hasattr(event, "delta") and event.delta and hasattr(event.delta, "text"):
            text_parts.append(event.delta.text)

        if (
            hasattr(event, "interaction")
            and event.interaction
            and getattr(event.interaction, "status", None) == "completed"
        ):
            complete_event = event

    full_text = "".join(text_parts)
    assert len(full_text) > 0, "Streaming should produce text"
    assert interaction_id is not None, "Should have received an interaction ID"
    assert len(interaction_id) > 0, f"ID should not be empty, got: {interaction_id}"

    print(f"  Events: {event_types}")
    print(f"  Full text: {full_text[:80]}")
    if complete_event and hasattr(complete_event, "interaction") and complete_event.interaction:
        interaction_obj = complete_event.interaction
        usage = getattr(interaction_obj, "usage", None) or (
            interaction_obj.get("usage") if isinstance(interaction_obj, dict) else None
        )
        if usage:
            in_tok = getattr(usage, "total_input_tokens", None) or (
                usage.get("total_input_tokens", 0) if isinstance(usage, dict) else 0
            )
            out_tok = getattr(usage, "total_output_tokens", None) or (
                usage.get("total_output_tokens", 0) if isinstance(usage, dict) else 0
            )
            print(f"  Usage: input={in_tok}, output={out_tok}")
    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test the Google Interactions API endpoint using the Google GenAI SDK")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8321",
        help="Base URL of the OGX server (default: http://localhost:8321)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model ID to use for tests (default: openai/gpt-4o-mini)",
    )
    args = parser.parse_args()

    client = _create_client(args.base_url)

    print(f"Testing Google Interactions API at {args.base_url}")
    print("Using Google GenAI SDK (google-genai)")
    print(f"Model: {args.model}")
    print("=" * 60)

    tests = [
        run_non_streaming_basic,
        run_non_streaming_system_instruction,
        run_non_streaming_multi_turn,
        run_non_streaming_generation_config,
        run_tool_calling,
        run_streaming_basic,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn(client, args.model)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

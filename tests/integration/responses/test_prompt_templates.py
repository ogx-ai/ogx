# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


@pytest.fixture
def responses_client_with_prompts(responses_client):
    """Skip clients without Prompts API support."""
    if not hasattr(responses_client, "prompts"):
        client_type = type(responses_client).__name__
        client_module = type(responses_client).__module__
        pytest.skip(f"Client does not support Prompts API: {client_module}.{client_type}")
    return responses_client


def text_var(value: str) -> dict:
    """Helper to create a text variable value for prompt substitution."""
    return {"type": "input_text", "text": value}


def test_basic_prompt_template(responses_client_with_prompts, text_model_id):
    """Test creating a prompt and using it in a response."""

    prompt = responses_client_with_prompts.prompts.create(prompt="You are a {{ role }} assistant.", variables=["role"])

    assert prompt.prompt_id.startswith("pmpt_")
    assert prompt.version == 1
    assert prompt.variables == ["role"]

    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "What is your role?"}],
        prompt={"id": prompt.prompt_id, "variables": {"role": text_var("helpful")}},
        stream=False,
    )

    assert response.prompt is not None
    assert response.prompt.id == prompt.prompt_id
    assert "role" in response.prompt.variables.keys()
    assert len(response.output_text) > 0


def test_multi_variable_prompt_template(responses_client_with_prompts, text_model_id):
    """Test a prompt template with multiple variables."""

    prompt_text = "You are a {{ role }} assistant specializing in {{ domain }}.  Your responsibilities include {{ responsibility1 }} and {{ responsibility2 }}.  Always maintain a {{ tone }} tone."

    prompt = responses_client_with_prompts.prompts.create(
        prompt=prompt_text, variables=["role", "domain", "responsibility1", "responsibility2", "tone"]
    )

    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "What do you do?"}],
        prompt={
            "id": prompt.prompt_id,
            "variables": {
                "role": text_var("technical support"),
                "domain": text_var("software engineering"),
                "responsibility1": text_var("debugging code"),
                "responsibility2": text_var("answering questions"),
                "tone": text_var("professional"),
            },
        },
        stream=False,
    )

    assert response.prompt.id == prompt.prompt_id
    assert len(response.output_text) > 0


def test_prompt_template_no_variables(responses_client_with_prompts, text_model_id):
    """Test using a prompt without any variables."""

    prompt = responses_client_with_prompts.prompts.create(prompt="You are a helpful assistant.", variables=[])

    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Hello"}],
        prompt={"id": prompt.prompt_id},
        stream=False,
    )

    assert response.prompt.id == prompt.prompt_id
    assert len(response.output_text) > 0


def test_multi_version_prompt_template(responses_client_with_prompts, text_model_id):
    """Test prompt evolution through versions with verifiable behavior changes."""

    # Version 1: Formal assistant with specific response prefix
    prompt_v1 = responses_client_with_prompts.prompts.create(
        prompt="You are a formal assistant. Always begin your response with the exact phrase 'Formal response:' followed by your answer in a professional tone.",
        variables=[],
    )
    assert prompt_v1.version == 1

    # Use v1 and verify it uses the formal prefix
    response_v1 = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Tell me about Python."}],
        prompt={"id": prompt_v1.prompt_id},
        stream=False,
    )
    assert response_v1.prompt.id == prompt_v1.prompt_id
    assert "Formal response:" in response_v1.output_text

    # Version 2: Casual assistant with different response prefix
    prompt_v2 = responses_client_with_prompts.prompts.update(
        prompt_v1.prompt_id,
        version=1,
        prompt="You are a casual, friendly assistant. Always begin your response with the exact phrase 'Casual response:' followed by your answer in a conversational, informal tone.",
        variables=[],
    )
    assert prompt_v2.version == 2
    assert prompt_v2.prompt_id == prompt_v1.prompt_id

    # Use default version (should be v2) and verify it uses the casual prefix
    response_v2_default = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Tell me about Python."}],
        prompt={"id": prompt_v1.prompt_id},
        stream=False,
    )
    assert response_v2_default.prompt.id == prompt_v1.prompt_id
    assert "Casual response:" in response_v2_default.output_text

    # Explicitly use v1 and verify it still uses the formal prefix
    response_v1_explicit = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Tell me about Python."}],
        prompt={"id": prompt_v1.prompt_id, "version": "1"},
        stream=False,
    )
    assert response_v1_explicit.prompt.id == prompt_v1.prompt_id
    assert "Formal response:" in response_v1_explicit.output_text


def test_prompt_template_with_streaming(responses_client_with_prompts, text_model_id):
    """Test using a prompt with streaming responses."""

    prompt = responses_client_with_prompts.prompts.create(prompt="You are a {{ role }} assistant.", variables=["role"])

    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Hello"}],
        prompt={"id": prompt.prompt_id, "variables": {"role": text_var("friendly")}},
        stream=True,
    )

    events = []
    for chunk in response:
        events.append(chunk)
        if chunk.type == "response.created":
            assert chunk.response.status == "in_progress"
            assert chunk.response.prompt.id == prompt.prompt_id
        elif chunk.type == "response.completed":
            assert chunk.response.status == "completed"
            assert chunk.response.prompt.id == prompt.prompt_id
            assert len(chunk.response.output_text) > 0

    event_types = {e.type for e in events}
    assert "response.created" in event_types
    assert "response.completed" in event_types

    assert "role" in events[-1].response.prompt.variables.keys()
    assert len(events[-1].response.output_text) > 0


def test_prompt_template_with_multi_turn(responses_client_with_prompts, text_model_id):
    """Test using a prompt across multiple turns in a conversation."""

    prompt = responses_client_with_prompts.prompts.create(
        prompt="You are a {{ specialty }} tutor.", variables=["specialty"]
    )

    # First turn
    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "What is your specialty?"}],
        prompt={"id": prompt.prompt_id, "variables": {"specialty": text_var("mathematics")}},
        stream=False,
    )
    assert response.prompt.id == prompt.prompt_id
    assert len(response.output_text) > 0
    assert "math" in response.output_text.lower()

    # Second turn
    response = responses_client_with_prompts.responses.create(
        model=text_model_id,
        input=[{"role": "user", "content": "Can you help me with algebra?"}],
        previous_response_id=response.id,
        stream=False,
    )
    assert len(response.output_text) > 0


def test_nonexistent_prompt_id(responses_client_with_prompts, text_model_id):
    """Test that response fails when prompt doesn't exist."""

    with pytest.raises(Exception) as e:
        responses_client_with_prompts.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "Hello"}],
            prompt={"id": "pmpt_nonexistent_12345"},
            stream=False,
        )

    assert e.value.status_code == 400
    assert "not found" in str(e.value).lower()


def test_invalid_prompt_version(responses_client_with_prompts, text_model_id):
    """Test that response fails when prompt version doesn't exist."""

    prompt = responses_client_with_prompts.prompts.create(prompt="You are helpful.", variables=[])

    with pytest.raises(Exception) as e:
        responses_client_with_prompts.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": "Hello"}],
            prompt={"id": prompt.prompt_id, "version": "99"},
            stream=False,
        )

    assert e.value.status_code == 400
    assert "not found" in str(e.value).lower()

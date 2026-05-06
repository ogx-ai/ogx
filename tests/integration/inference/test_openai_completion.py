# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import time

import pytest
from pydantic import BaseModel

from ..helpers import assert_text_contains
from ..test_cases.test_case import TestCase


def provider_from_model(client_with_models, model_id):
    models = {m.id: m for m in client_with_models.models.list()}
    models.update(
        {m.custom_metadata["provider_resource_id"]: m for m in client_with_models.models.list() if m.custom_metadata}
    )
    provider_id = models[model_id].custom_metadata["provider_id"]
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_model_doesnt_support_openai_completion(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "inline::sentence-transformers",
        "remote::vllm",
        "remote::bedrock",
        "remote::databricks",
        # Technically Nvidia does support OpenAI completions, but none of their hosted models
        # support both completions and chat completions endpoint and all the Llama models are
        # just chat completions
        "remote::nvidia",
        "remote::runpod",
        "remote::sambanova",
        "remote::vertexai",
        # {"error":{"message":"Unknown request URL: GET /openai/v1/completions. Please check the URL for typos,
        # or see the docs at https://console.groq.com/docs/","type":"invalid_request_error","code":"unknown_url"}}
        "remote::groq",
        "remote::llama-cpp-server",
        "remote::oci",
        "remote::gemini",  # https://generativelanguage.googleapis.com/v1beta/openai/completions -> 404
        "remote::anthropic",  # at least claude-3-{5,7}-{haiku,sonnet}-* / claude-{sonnet,opus}-4-* are not supported
        "remote::azure",  # {'error': {'code': 'OperationNotSupported', 'message': 'The completion operation
        #  does not work with the specified model, gpt-5-mini. Please choose different model and try
        #  again. You can learn more about which models can be used with each operation here:
        #  https://go.microsoft.com/fwlink/?linkid=2197993.'}}"}
        "remote::llama-openai-compat",
        "remote::watsonx",  # WatsonX only has /v1/chat/completions, no /v1/completions
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI completions.")


def skip_if_doesnt_support_completions_logprobs(client_with_models, model_id):
    provider_type = provider_from_model(client_with_models, model_id).provider_type
    if provider_type in (
        "remote::ollama",  # logprobs is ignored
        "remote::watsonx",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider_type} doesn't support /v1/completions logprobs.")


def skip_if_model_doesnt_support_suffix(client_with_models, model_id):
    # To test `fim` ( fill in the middle ) completion, we need to use a model that supports suffix.
    # Use this to specifically test this API functionality.

    # pytest -sv --stack-config="inference=starter" \
    # tests/integration/inference/test_openai_completion.py \
    # --text-model qwen2.5-coder:1.5b \
    # -k test_openai_completion_non_streaming_suffix

    if model_id != "qwen2.5-coder:1.5b":
        pytest.skip(f"Suffix is not supported for the model: {model_id}.")

    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::ollama":
        pytest.skip(f"Provider {provider.provider_type} doesn't support suffix.")


def skip_if_doesnt_support_n(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "remote::sambanova",
        "remote::ollama",
        # https://console.groq.com/docs/openai#currently-unsupported-openai-features
        # -> Error code: 400 - {'error': {'message': "'n' : number must be at most 1", 'type': 'invalid_request_error'}}
        "remote::groq",
        # Error code: 400 - [{'error': {'code': 400, 'message': 'Only one candidate can be specified in the
        # current model', 'status': 'INVALID_ARGUMENT'}}]
        "remote::gemini",
        # https://docs.anthropic.com/en/api/openai-sdk#simple-fields
        "remote::anthropic",
        "remote::vertexai",
        #  Error code: 400 - [{'error': {'code': 400, 'message': 'Unable to submit request because candidateCount must be 1 but
        #  the entered value was 2. Update the candidateCount value and try again.', 'status': 'INVALID_ARGUMENT'}
        "remote::together",  # `n` > 1 is not supported when streaming tokens. Please disable `stream`
        # Error code 400 - {'message': '"n" > 1 is not currently supported', 'type': 'invalid_request_error', 'param': 'n', 'code': 'wrong_api_format'}
        "remote::cerebras",
        "remote::databricks",  # Bad request: parameter "n" must be equal to 1 for streaming mode
        "remote::watsonx",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support n param.")


def skip_if_model_doesnt_support_openai_chat_completion(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "inline::sentence-transformers",
        "remote::vllm",
        "remote::databricks",
        "remote::cerebras",
        "remote::runpod",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI chat completions.")


def skip_if_provider_isnt_vllm(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::vllm":
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support vllm extra_body parameters.")


def skip_if_provider_isnt_openai(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::openai":
        pytest.skip(
            f"Model {model_id} hosted by {provider.provider_type} doesn't support chat completion calls with base64 encoded files."
        )


def skip_if_provider_doesnt_support_tool_calling(client_with_models, model_id):
    """Skip tests for providers that don't support tool calling in their OpenAI-compatible API."""
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "remote::bedrock",  # Bedrock's OpenAI endpoint doesn't support tool calling
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support tool calling.")


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_openai_completion_non_streaming(ogx_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = ogx_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.text) > 10


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:suffix",
    ],
)
def test_openai_completion_non_streaming_suffix(ogx_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    skip_if_model_doesnt_support_suffix(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    response = ogx_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        stream=False,
        suffix=tc["suffix"],
        max_tokens=10,
    )

    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.text) > 5
    assert_text_contains(choice.text, "france")


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_openai_completion_streaming(ogx_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = ogx_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=True,
        max_tokens=50,
    )
    streamed_content = [chunk.choices[0].text or "" for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    assert len(content_str) > 10


def test_openai_completion_guided_choice(ogx_client, client_with_models, text_model_id):
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    prompt = "I am feeling really sad today."
    response = ogx_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
        extra_body={"structured_outputs": {"choice": ["joy", "sadness"]}},
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert choice.text in ["joy", "sadness"]


# Run the chat-completion tests with both the OpenAI client and the OGX client


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:non_streaming_01",
        "inference:chat_completion:non_streaming_02",
    ],
)
def test_openai_chat_completion_non_streaming(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        stream=False,
    )
    message_content = response.choices[0].message.content
    assert len(message_content) > 0
    assert_text_contains(message_content, expected)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:streaming_01",
        "inference:chat_completion:streaming_02",
    ],
)
def test_openai_chat_completion_streaming(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history
    )
    streamed_content = []
    for chunk in response:
        # On some providers like Azure, the choices are empty on the first chunk, so we need to check for that
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            streamed_content.append(chunk.choices[0].delta.content)
    assert len(streamed_content) > 0
    assert_text_contains("".join(streamed_content), expected)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:streaming_01",
        "inference:chat_completion:streaming_02",
    ],
)
def test_openai_chat_completion_streaming_with_n(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    skip_if_doesnt_support_n(client_with_models, text_model_id)

    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history,
        n=2,
    )
    streamed_content = {}
    for chunk in response:
        for choice in chunk.choices:
            if choice.delta.content:
                streamed_content[choice.index] = streamed_content.get(choice.index, "") + choice.delta.content
    assert len(streamed_content) == 2
    for i, content in streamed_content.items():
        assert_text_contains(content, expected, msg=f"Choice {i}: Expected '{expected}' in '{content}'")


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_inference_store(compat_client, client_with_models, text_model_id, stream):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    client = compat_client
    # make a chat completion
    message = "Hello, world!"
    response = client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=stream,
    )
    if stream:
        # accumulate the streamed content
        content = ""
        response_id = None
        for chunk in response:
            if response_id is None and chunk.id:
                response_id = chunk.id
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    else:
        response_id = response.id
        content = response.choices[0].message.content

    tries = 0
    while tries < 10:
        responses = client.chat.completions.list(limit=1000)
        if response_id in [r.id for r in responses.data]:
            break
        else:
            tries += 1
            time.sleep(0.1)
    assert tries < 10, f"Response {response_id} not found after 1 second"

    retrieved_response = client.chat.completions.retrieve(response_id)
    assert retrieved_response.id == response_id
    assert retrieved_response.choices[0].message.content == content, retrieved_response

    input_content = (
        getattr(retrieved_response.input_messages[0], "content", None)
        or retrieved_response.input_messages[0]["content"]
    )
    assert input_content == message, retrieved_response


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_inference_store_tool_calls(compat_client, client_with_models, text_model_id, stream):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    skip_if_provider_doesnt_support_tool_calling(client_with_models, text_model_id)
    client = compat_client
    # make a chat completion
    message = "What's the weather in Tokyo? Use the get_weather function to get the weather."
    response = client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city to get the weather for"},
                        },
                    },
                },
            }
        ],
    )
    if stream:
        # accumulate the streamed content
        content = ""
        response_id = None
        for chunk in response:
            if response_id is None and chunk.id:
                response_id = chunk.id
            if chunk.choices and len(chunk.choices) > 0:
                if delta := chunk.choices[0].delta:
                    if delta.content:
                        content += delta.content
    else:
        response_id = response.id
        content = response.choices[0].message.content

    # wait for the response to be stored
    tries = 0
    while tries < 10:
        responses = client.chat.completions.list(limit=1000)
        if response_id in [r.id for r in responses.data]:
            break
        else:
            tries += 1
            time.sleep(0.1)

    assert tries < 10, f"Response {response_id} not found after 1 second"

    responses = client.chat.completions.list(limit=1000)
    assert response_id in [r.id for r in responses.data]

    retrieved_response = client.chat.completions.retrieve(response_id)
    assert retrieved_response.id == response_id
    input_content = (
        getattr(retrieved_response.input_messages[0], "content", None)
        or retrieved_response.input_messages[0]["content"]
    )
    assert input_content == message, retrieved_response
    tool_calls = retrieved_response.choices[0].message.tool_calls
    # sometimes model doesn't output tool calls, but we still want to test that the tool was called
    if tool_calls:
        # because we test with small models, just check that we retrieved
        # a tool call with a name and arguments string, but ignore contents
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name
        assert tool_calls[0].function.arguments
    else:
        # failed tool call parses show up as a message with content, so ensure
        # that the retrieve response content matches the original request
        assert retrieved_response.choices[0].message.content == content


def test_openai_chat_completion_non_streaming_with_file(openai_client, client_with_models, text_model_id):
    skip_if_provider_isnt_openai(client_with_models, text_model_id)

    # Hardcoded base64-encoded PDF with "Hello World" text
    pdf_base64 = "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCjIgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFszIDAgUl0KL0NvdW50IDEKPD4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCAyIDAgUgovTWVkaWFCb3ggWzAgMCA2MTIgNzkyXQovQ29udGVudHMgNCAwIFIKL1Jlc291cmNlcyA8PAovRm9udCA8PAovRjEgPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+Cj4+Cj4+Cj4+CmVuZG9iago0IDAgb2JqCjw8Ci9MZW5ndGggNDQKPj4Kc3RyZWFtCkJUCi9GMSAxMiBUZgoxMDAgNzUwIFRkCihIZWxsbyBXb3JsZCkgVGoKRVQKZW5kc3RyZWFtCmVuZG9iagp4cmVmCjAgNQowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAzMTUgMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSA1Ci9Sb290IDEgMCBSCj4+CnN0YXJ0eHJlZgo0MDkKJSVFT0Y="

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": "Describe what you see in this PDF file.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "my-temp-hello-world-pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_base64}",
                        },
                    }
                ],
            },
        ],
        stream=False,
    )
    assert_text_contains(response.choices[0].message.content, "hello world")


def skip_if_model_doesnt_support_reasoning(model_id):
    """Skip if the model is not known to emit reasoning/thinking tokens."""
    if "gpt-oss" not in model_id.lower():
        pytest.skip(f"Model {model_id} doesn't emit reasoning tokens; skipping reasoning passthrough test.")


def test_openai_chat_completion_reasoning_passthrough(openai_client, client_with_models, text_model_id):
    """Verify that reasoning tokens survive a round-trip through OGX.

    Turn 1: send a prompt, assert that the response carries reasoning content
            in model_extra (transparent passthrough from the provider).
    Turn 2: echo the assistant message, including its reasoning field, back
            as history, send a follow-up, and assert that the stack accepted
            the message and returned a coherent answer.  This exercises the
            OpenAIAssistantMessageParam extra="allow" fix: without it the
            reasoning field would be silently dropped before reaching the
            provider, degrading multi-turn quality.
    """
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    skip_if_model_doesnt_support_reasoning(text_model_id)

    # Turn 1
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Think step by step."},
    ]
    resp1 = openai_client.chat.completions.create(model=text_model_id, messages=messages)
    msg1 = resp1.choices[0].message

    # Reasoning content arrives as a non-spec field; it lives in model_extra
    # because the OpenAI SDK uses extra="allow" on its response types.
    reasoning = (msg1.model_extra or {}).get("reasoning") or (msg1.model_extra or {}).get("reasoning_content")
    assert reasoning, (
        f"Expected reasoning content in turn-1 response from {text_model_id}. model_extra={msg1.model_extra}"
    )

    # Turn 2
    # model_dump() includes model_extra, so 'reasoning' is present in the dict
    # that gets sent back to the stack.  The stack must not drop it.
    messages.append(msg1.model_dump())
    messages.append({"role": "user", "content": "Now multiply that result by 3."})

    resp2 = openai_client.chat.completions.create(model=text_model_id, messages=messages)
    msg2 = resp2.choices[0].message

    assert msg2.content, "Expected a non-empty response in turn 2"


def skip_if_doesnt_support_completions_stop_sequence(client_with_models, model_id):
    provider_type = provider_from_model(client_with_models, model_id).provider_type
    if provider_type in ("remote::watsonx",):  # openai.BadRequestError: Error code: 400
        pytest.skip(f"Model {model_id} hosted by {provider_type} doesn't support /v1/completions stop sequence.")


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:stop_sequence",
    ],
)
def test_openai_completion_stop_sequence(client_with_models, openai_client, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    skip_if_doesnt_support_completions_stop_sequence(client_with_models, text_model_id)

    tc = TestCase(test_case)

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        stop="1963",
        stream=False,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert "1963" not in choice.text

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        stop=["blathering", "1963"],
        stream=False,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert "1963" not in choice.text


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:log_probs",
    ],
)
def test_openai_completion_logprobs(client_with_models, openai_client, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    skip_if_doesnt_support_completions_logprobs(client_with_models, text_model_id)

    tc = TestCase(test_case)

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        logprobs=5,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert choice.text, "Response text should not be empty"
    assert choice.logprobs, "Logprobs should not be empty"
    logprobs = choice.logprobs
    assert logprobs.token_logprobs, "Response tokens should not be empty"
    assert len(logprobs.tokens) == len(logprobs.token_logprobs)
    assert len(logprobs.token_logprobs) == len(logprobs.top_logprobs)
    for i, (token, prob) in enumerate(zip(logprobs.tokens, logprobs.token_logprobs, strict=True)):
        assert logprobs.top_logprobs[i][token] == prob
        assert len(logprobs.top_logprobs[i]) == 5


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:log_probs",
    ],
)
def test_openai_completion_logprobs_streaming(client_with_models, openai_client, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    skip_if_doesnt_support_completions_logprobs(client_with_models, text_model_id)

    tc = TestCase(test_case)

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        logprobs=3,
        stream=True,
        max_tokens=5,
    )
    for chunk in response:
        choice = chunk.choices[0]
        choice = response.choices[0]
        if choice.text:  # if there's a token, we expect logprobs
            assert choice.logprobs, "Logprobs should not be empty"
            logprobs = choice.logprobs
            assert logprobs.token_logprobs, "Response tokens should not be empty"
            assert len(logprobs.tokens) == len(logprobs.token_logprobs)
            assert len(logprobs.token_logprobs) == len(logprobs.top_logprobs)
            for i, (token, prob) in enumerate(zip(logprobs.tokens, logprobs.token_logprobs, strict=True)):
                assert logprobs.top_logprobs[i][token] == prob
                assert len(logprobs.top_logprobs[i]) == 3
        else:  # no token, no logprobs
            assert not choice.logprobs, "Logprobs should be empty"


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_openai_chat_completion_with_tools(openai_client, text_model_id, test_case):
    tc = TestCase(test_case)

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_choice="auto",
        stream=False,
    )
    assert len(response.choices) == 1
    assert len(response.choices[0].message.tool_calls) == 1
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == tc["tools"][0]["function"]["name"]
    assert "location" in tool_call.function.arguments
    assert tc["expected"]["location"] in tool_call.function.arguments


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_openai_chat_completion_with_tools_and_streaming(openai_client, text_model_id, test_case):
    tc = TestCase(test_case)

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_choice="auto",
        stream=True,
    )
    # Accumulate tool calls from streaming chunks
    tool_calls = []
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.tool_calls:
            for i, tc_delta in enumerate(chunk.choices[0].delta.tool_calls):
                while len(tool_calls) <= i:
                    tool_calls.append({"function": {"name": "", "arguments": ""}})
                if tc_delta.function and tc_delta.function.name:
                    tool_calls[i]["function"]["name"] = tc_delta.function.name
                if tc_delta.function and tc_delta.function.arguments:
                    tool_calls[i]["function"]["arguments"] += tc_delta.function.arguments
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == tc["tools"][0]["function"]["name"]
    assert "location" in tool_call["function"]["arguments"]
    assert tc["expected"]["location"] in tool_call["function"]["arguments"]


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:tool_calling",
    ],
)
def test_openai_chat_completion_with_tool_choice_none(openai_client, text_model_id, test_case):
    tc = TestCase(test_case)

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=tc["messages"],
        tools=tc["tools"],
        tool_choice="none",
        stream=False,
    )
    assert len(response.choices) == 1
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is None or len(tool_calls) == 0


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:structured_output",
    ],
)
def test_openai_chat_completion_structured_output(openai_client, text_model_id, test_case):
    # Note: Skip condition may need adjustment for OpenAI client
    class AnswerFormat(BaseModel):
        first_name: str
        last_name: str
        year_of_birth: int

    tc = TestCase(test_case)

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=tc["messages"],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "AnswerFormat",
                "schema": AnswerFormat.model_json_schema(),
            },
        },
        stream=False,
    )
    print(response.choices[0].message.content)
    answer = AnswerFormat.model_validate_json(response.choices[0].message.content)
    expected = tc["expected"]
    assert expected["first_name"].lower() in answer.first_name.lower()
    assert expected["last_name"].lower() in answer.last_name.lower()
    assert answer.year_of_birth == expected["year_of_birth"]

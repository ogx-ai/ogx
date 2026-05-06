# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import struct

import pytest
from openai import OpenAI

from ogx.core.library_client import OGXAsLibraryClient

ASYMMETRIC_EMBEDDING_MODELS_BY_PROVIDER = {
    "remote::nvidia": [
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "snowflake/arctic-embed-l",
    ],
}


def decode_base64_to_floats(base64_string: str) -> list[float]:
    """Helper function to decode base64 string to list of float32 values."""
    embedding_bytes = base64.b64decode(base64_string)
    float_count = len(embedding_bytes) // 4  # 4 bytes per float32
    embedding_floats = struct.unpack(f"{float_count}f", embedding_bytes)
    return list(embedding_floats)


def provider_from_model(client_with_models, model_id):
    models = {m.id: m for m in client_with_models.models.list()}
    models.update(
        {m.custom_metadata["provider_resource_id"]: m for m in client_with_models.models.list() if m.custom_metadata}
    )
    provider_id = models[model_id].custom_metadata["provider_id"]
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def is_asymmetric_model(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    provider_type = provider.provider_type

    if provider_type not in ASYMMETRIC_EMBEDDING_MODELS_BY_PROVIDER:
        return False

    return model_id in ASYMMETRIC_EMBEDDING_MODELS_BY_PROVIDER[provider_type]


def get_extra_body_for_model(client_with_models, model_id, input_type="query"):
    if not is_asymmetric_model(client_with_models, model_id):
        return None

    provider = provider_from_model(client_with_models, model_id)

    if provider.provider_type == "remote::nvidia":
        return {"input_type": input_type}

    return None


def skip_if_model_doesnt_support_user_param(client, model_id):
    provider = provider_from_model(client, model_id)
    if provider.provider_type in (
        "remote::together",  # service returns 400
        "remote::fireworks",  # service returns 400 malformed input
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} does not support user param.")


def skip_if_model_doesnt_support_encoding_format_base64(client, model_id):
    provider = provider_from_model(client, model_id)

    should_skip = provider.provider_type in (
        "remote::databricks",  # param silently ignored, always returns floats
        "remote::fireworks",  # param silently ignored, always returns list of floats
        "remote::ollama",  # param silently ignored, always returns list of floats
    ) or (
        provider.provider_type == "remote::nvidia"
        and model_id
        in [
            "nvidia/nv-embedqa-e5-v5",
            "nvidia/nv-embedqa-mistral-7b-v2",
            "snowflake/arctic-embed-l",
        ]
    )

    if should_skip:
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} does not support encoding_format='base64'.")


def skip_if_model_doesnt_support_variable_dimensions(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)

    should_skip = (
        provider.provider_type
        in (
            "remote::together",  # returns 400
            "inline::sentence-transformers",
            # Error code: 400 - {'error_code': 'BAD_REQUEST', 'message': 'Bad request: json: unknown field "dimensions"\n'}
            "remote::databricks",
            "remote::watsonx",  # watsonx does not support the dimensions parameter
        )
        or (provider.provider_type == "remote::openai" and "text-embedding-3" not in model_id)
        or (
            provider.provider_type == "remote::nvidia"
            and model_id
            in [
                "nvidia/nv-embedqa-e5-v5",
                "nvidia/nv-embedqa-mistral-7b-v2",
                "snowflake/arctic-embed-l",
            ]
        )
    )

    if should_skip:
        pytest.skip(
            f"Model {model_id} hosted by {provider.provider_type} does not support variable output embedding dimensions."
        )


@pytest.fixture(params=["openai_client", "ogx_client"])
def compat_client(request, client_with_models):
    if request.param == "openai_client" and isinstance(client_with_models, OGXAsLibraryClient):
        pytest.skip("OpenAI client tests not supported with library client")
    return request.getfixturevalue(request.param)


def skip_if_model_doesnt_support_openai_embeddings(client, model_id):
    provider = provider_from_model(client, model_id)
    if provider.provider_type in (
        "inline::builtin",
        "remote::bedrock",
        "remote::cerebras",
        "remote::runpod",
        "remote::sambanova",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI embeddings.")


@pytest.fixture
def openai_client(client_with_models):
    base_url = f"{client_with_models.base_url}/v1"
    return OpenAI(base_url=base_url, api_key="fake")


def test_openai_embeddings_single_string(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with a single string input."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    input_text = "Hello, world!"

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text,
        encoding_format="float",
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    assert response.object == "list"

    assert response.model == embedding_model_id
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    assert isinstance(response.data[0].embedding, list)
    assert len(response.data[0].embedding) > 0
    assert all(isinstance(x, float) for x in response.data[0].embedding)


def test_openai_embeddings_multiple_strings(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with multiple string inputs."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    input_texts = ["Hello, world!", "How are you today?", "This is a test."]

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_texts,
        encoding_format="float",
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    assert response.object == "list"

    assert response.model == embedding_model_id
    assert len(response.data) == len(input_texts)

    for i, embedding_data in enumerate(response.data):
        assert embedding_data.object == "embedding"
        assert embedding_data.index == i
        assert isinstance(embedding_data.embedding, list)
        assert len(embedding_data.embedding) > 0
        assert all(isinstance(x, float) for x in embedding_data.embedding)


def test_openai_embeddings_with_encoding_format_float(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with float encoding format."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    input_text = "Test encoding format"

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text,
        encoding_format="float",
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    assert response.object == "list"
    assert len(response.data) == 1
    assert isinstance(response.data[0].embedding, list)
    assert all(isinstance(x, float) for x in response.data[0].embedding)


def test_openai_embeddings_with_dimensions(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with custom dimensions parameter."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)
    skip_if_model_doesnt_support_variable_dimensions(client_with_models, embedding_model_id)

    input_text = "Test dimensions parameter"
    dimensions = 16

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text,
        dimensions=dimensions,
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    assert response.object == "list"
    assert len(response.data) == 1
    # Note: Not all models support custom dimensions, so we don't assert the exact dimension
    assert isinstance(response.data[0].embedding, list)
    assert len(response.data[0].embedding) > 0


def test_openai_embeddings_with_user_parameter(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with user parameter."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)
    skip_if_model_doesnt_support_user_param(client_with_models, embedding_model_id)

    input_text = "Test user parameter"
    user_id = "test-user-123"

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text,
        user=user_id,
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    assert response.object == "list"
    assert len(response.data) == 1
    assert isinstance(response.data[0].embedding, list)
    assert len(response.data[0].embedding) > 0


def test_openai_embeddings_empty_list_error(compat_client, client_with_models, embedding_model_id):
    """Test that empty list input raises an appropriate error."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    with pytest.raises(Exception):  # noqa: B017
        compat_client.embeddings.create(
            model=embedding_model_id,
            input=[],
            extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
        )


def test_openai_embeddings_invalid_model_error(compat_client, client_with_models, embedding_model_id):
    """Test that invalid model ID raises an appropriate error."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    with pytest.raises(Exception):  # noqa: B017
        compat_client.embeddings.create(
            model="invalid-model-id",
            input="Test text",
            extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
        )


def test_openai_embeddings_different_inputs_different_outputs(compat_client, client_with_models, embedding_model_id):
    """Test that different inputs produce different embeddings."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)

    input_text1 = "This is the first text"
    input_text2 = "This is completely different content"

    extra_body = get_extra_body_for_model(client_with_models, embedding_model_id)
    response1 = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text1,
        encoding_format="float",
        extra_body=extra_body,
    )

    response2 = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text2,
        encoding_format="float",
        extra_body=extra_body,
    )

    embedding1 = response1.data[0].embedding
    embedding2 = response2.data[0].embedding

    assert len(embedding1) == len(embedding2)
    # Embeddings should be different for different inputs
    assert embedding1 != embedding2


def test_openai_embeddings_with_encoding_format_base64(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with base64 encoding format."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)
    skip_if_model_doesnt_support_encoding_format_base64(client_with_models, embedding_model_id)
    skip_if_model_doesnt_support_variable_dimensions(client_with_models, embedding_model_id)

    input_text = "Test base64 encoding format"
    dimensions = 12

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_text,
        encoding_format="base64",
        dimensions=dimensions,
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )

    # Validate response structure
    assert response.object == "list"
    assert len(response.data) == 1

    # With base64 encoding, embedding should be a string, not a list
    embedding_data = response.data[0]
    assert embedding_data.object == "embedding"
    assert embedding_data.index == 0
    assert isinstance(embedding_data.embedding, str)

    # Verify it's valid base64 and decode to floats
    embedding_floats = decode_base64_to_floats(embedding_data.embedding)

    # Verify we got valid floats
    assert len(embedding_floats) == dimensions, f"Got embedding length {len(embedding_floats)}, expected {dimensions}"
    assert all(isinstance(x, float) for x in embedding_floats)


def test_openai_embeddings_base64_batch_processing(compat_client, client_with_models, embedding_model_id):
    """Test OpenAI embeddings endpoint with base64 encoding for batch processing."""
    skip_if_model_doesnt_support_openai_embeddings(client_with_models, embedding_model_id)
    skip_if_model_doesnt_support_encoding_format_base64(client_with_models, embedding_model_id)

    input_texts = ["First text for base64", "Second text for base64", "Third text for base64"]

    response = compat_client.embeddings.create(
        model=embedding_model_id,
        input=input_texts,
        encoding_format="base64",
        extra_body=get_extra_body_for_model(client_with_models, embedding_model_id),
    )
    # Validate response structure
    assert response.object == "list"

    assert response.model == embedding_model_id
    assert len(response.data) == len(input_texts)

    # Validate each embedding in the batch
    embedding_dimensions = []
    for i, embedding_data in enumerate(response.data):
        assert embedding_data.object == "embedding"
        assert embedding_data.index == i

        # With base64 encoding, embedding should be a string, not a list
        assert isinstance(embedding_data.embedding, str)
        embedding_floats = decode_base64_to_floats(embedding_data.embedding)
        assert len(embedding_floats) > 0
        assert all(isinstance(x, float) for x in embedding_floats)
        embedding_dimensions.append(len(embedding_floats))

    # All embeddings should have the same dimensionality
    assert all(dim == embedding_dimensions[0] for dim in embedding_dimensions)

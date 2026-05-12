# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for SDK compatibility on /v1/models."""


def test_models_list_openai_sdk_response_shape(openai_client, text_model_id):
    """OpenAI SDK models.list() returns OpenAI model objects."""
    response = openai_client.models.list()
    assert len(response.data) > 0

    model = next((m for m in response.data if m.id == text_model_id), response.data[0])
    assert model.object == "model"
    assert isinstance(model.id, str) and model.id
    assert model.owned_by == "ogx"


def test_models_list_anthropic_sdk_response_shape(anthropic_client, text_model_id):
    """Anthropic SDK models.list() returns Anthropic model info objects."""
    response = anthropic_client.models.list(limit=100)
    assert len(response.data) > 0

    model = next((m for m in response.data if m.id == text_model_id), response.data[0])
    assert model.type == "model"
    assert model.display_name == model.id
    assert model.created_at is not None

    assert response.first_id == response.data[0].id
    assert response.last_id == response.data[-1].id


def test_models_list_google_sdk_response_shape(google_genai_client, text_model_id):
    """Google GenAI SDK models.list() returns Google model objects."""
    models = list(google_genai_client.models.list())
    assert len(models) > 0

    model = next((m for m in models if m.name == f"models/{text_model_id}"), models[0])
    assert model.name.startswith("models/")
    normalized_name = model.name.split("models/", 1)[1]
    assert normalized_name
    if model.display_name is not None:
        assert isinstance(model.display_name, str) and model.display_name


def test_models_get_google_sdk_response_shape(google_genai_client, text_model_id):
    """Google GenAI SDK models.get() resolves /v1/models/{model_id} correctly."""
    model = google_genai_client.models.get(model=f"models/{text_model_id}")
    assert model.name == f"models/{text_model_id}"
    if model.display_name is not None:
        assert model.display_name == text_model_id

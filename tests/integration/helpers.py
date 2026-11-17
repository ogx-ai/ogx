# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unicodedata

import pytest
from ogx_client import OgxClient, ProviderInfo

from ogx.providers.utils.inference.models_dev_registry import supports_reasoning


def normalize_text(text: str) -> str:
    """Normalize Unicode text for comparison by stripping diacritical marks and non-ASCII characters.

    Models often return typographically correct but comparison-hostile text:
    narrow no-break spaces between numbers and units (``100\\u202f°C``),
    macrons on Latin words (``sōl``), etc.  NFD-decomposing then encoding
    to ASCII strips all such variation so that simple ``in`` checks work.
    """
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def assert_text_contains(text: str, expected: str, msg: str | None = None):
    """Assert that *expected* appears in *text* after Unicode normalisation."""
    normalized = normalize_text(text)
    normalized_expected = normalize_text(expected)
    assert normalized_expected in normalized, msg or f"Expected '{expected}' in text: {text}"


def provider_from_model(client_with_models: OgxClient, model_id: str) -> ProviderInfo:
    """Resolve the provider serving *model_id*.

    A model is matched by its registered ID or by its ``provider_resource_id``
    alias; the alias wins when the two disagree.
    """
    listed_models = client_with_models.models.list().data
    models = {m.id: m for m in listed_models}
    models.update(
        {
            m.custom_metadata["provider_resource_id"]: m
            for m in listed_models
            if m.custom_metadata and "provider_resource_id" in m.custom_metadata
        }
    )

    model = models.get(model_id)
    if model is None:
        pytest.fail(
            f"Failed to resolve provider for model '{model_id}' — not found among {len(listed_models)} listed models"
        )

    provider_id = (model.custom_metadata or {}).get("provider_id")
    if provider_id is None:
        pytest.fail(f"Failed to resolve provider for model '{model_id}' — no 'provider_id' in its custom_metadata")

    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    if provider_id not in providers:
        pytest.fail(f"Failed to resolve provider for model '{model_id}' — provider '{provider_id}' is not registered")
    return providers[provider_id]


def is_ollama_reasoning_model(client: OgxClient, model_id: str) -> bool:
    """Whether *model_id* is served by Ollama and reasons by default (e.g. qwen3.5:0.8b).

    The legacy /v1/completions endpoint has no reasoning control at all, and
    Ollama thinking models answer it with reasoning that exhausts the budget
    and empty content. The completion tests raise the token budget for these
    models (ollama_reasoning_completion_max_tokens); this check keeps them
    running unchanged against non-reasoning Ollama models.
    """
    provider_type = provider_from_model(client, model_id).provider_type
    return provider_type == "remote::ollama" and supports_reasoning(model_id)

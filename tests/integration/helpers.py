# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unicodedata

import pytest
from ogx_client import OgxClient, ProviderInfo


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

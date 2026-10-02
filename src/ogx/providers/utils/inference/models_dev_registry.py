# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared model classification for remote adapters whose /v1/models response has no
model task/type field, so embedding and rerank models can't be told apart from the
identifier alone. Providers that serve arbitrary upstream model IDs (vLLM, llama.cpp
servers, etc.) call classify_model() to classify by the models.dev registry first,
falling back to a name heuristic for models it doesn't know about.

Only embedding classification consults models.dev -- its registry has no rerank
entries, so rerank classification is a name heuristic only, with no metadata
enrichment.
"""

from functools import cache

import models_dev as _models_dev

from ogx.log import get_logger
from ogx_api import Model, ModelType

log = get_logger(name=__name__, category="inference::models_dev_registry")


def _is_embedding_model(model_id: str, model: _models_dev.Model) -> bool:
    """True if models.dev or the identifier itself marks this as an embedding model."""
    return (model.family is not None and "embed" in model.family) or "embed" in model_id.lower()


@cache
def _models_dev_index() -> dict[str, _models_dev.Model]:
    """Index of models.dev embedding models by model ID, across all providers.

    Sorted so the huggingface provider entry is processed last: adapters that serve
    Hugging Face model IDs directly (vLLM, llama.cpp GGUF repos, etc.) find it the
    most authoritative source for those IDs, so it wins on conflicting entries.
    """
    index: dict[str, _models_dev.Model] = {}
    for provider in sorted(_models_dev.providers(), key=lambda p: p.id == "huggingface"):
        for model_id, model in provider.models.items():
            if _is_embedding_model(model_id, model):
                index[model_id] = model
    return index


def _lookup_models_dev(identifier: str) -> _models_dev.Model | None:
    """Look up ``identifier`` in the models.dev embedding-model index, if present."""
    return _models_dev_index().get(identifier)


def _embedding_metadata(model: _models_dev.Model) -> dict[str, int]:
    """Build the Model.metadata dict (embedding_dimension, context_length) for a
    matched models.dev embedding model entry."""
    metadata: dict[str, int] = {}
    if model.limit.output:
        metadata["embedding_dimension"] = model.limit.output
    if model.limit.context:
        metadata["context_length"] = model.limit.context
    return metadata


def classify_model(identifier: str, provider_id: str) -> Model | None:
    """Classify ``identifier`` as an embedding or rerank model.

    Embedding classification checks the models.dev registry first, enriching
    ``Model.metadata`` with ``embedding_dimension``/``context_length`` when found,
    and falls back to a name heuristic (``"embed"`` in the identifier) for models
    models.dev doesn't know about. Rerank classification is a pure name heuristic
    (``"rerank"`` in the identifier) -- models.dev has no rerank entries to consult.

    Returns ``None`` if ``identifier`` matches neither category; callers should
    fall back to their own default classification (typically
    ``super().construct_model_from_identifier()``).
    """
    md = _lookup_models_dev(identifier)
    is_embedding = md is not None or "embed" in identifier.lower()

    if is_embedding:
        metadata: dict[str, int] = {}
        if md is not None:
            metadata = _embedding_metadata(md)
            log.debug(
                "Classified embedding model via models.dev",
                identifier=identifier,
                provider_id=provider_id,
                family=md.family,
                metadata=metadata,
            )
        else:
            log.debug(
                "Classified embedding model via name heuristic (not in models.dev)",
                identifier=identifier,
                provider_id=provider_id,
            )
        return Model(
            provider_id=provider_id,
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.embedding,
            metadata=metadata,
        )

    if "rerank" in identifier.lower():
        return Model(
            provider_id=provider_id,
            provider_resource_id=identifier,
            identifier=identifier,
            model_type=ModelType.rerank,
        )

    return None

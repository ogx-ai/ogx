# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the shared models.dev classification used by remote adapters whose
/v1/models response has no model task/type field (vLLM, llama.cpp servers).
Exercised once here against the real models.dev data; adapter-specific test
files only need to check that they delegate to classify_model correctly."""

from ogx.providers.utils.inference.models_dev_registry import classify_model
from ogx_api import ModelType


class TestClassifyModel:
    def test_family_check_classifies_embedding_without_embed_in_identifier(self):
        # intfloat/multilingual-e5-large-instruct has no "embed" in its identifier
        # but its models.dev family is "text-embedding", so it must be classified
        # as an embedding model with metadata populated from models.dev.
        model = classify_model("intfloat/multilingual-e5-large-instruct", "test-provider")

        assert model is not None
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension") == 512

    def test_known_embedding_model_populates_metadata_from_models_dev(self):
        # text-embedding-3-large is in models_dev (openai provider) with
        # limit.output=3072 (embedding dimension) and limit.context=8191.
        model = classify_model("text-embedding-3-large", "test-provider")

        assert model is not None
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension") == 3072
        assert model.metadata.get("context_length") == 8191

    def test_unknown_embedding_model_falls_back_to_name_heuristic(self):
        model = classify_model("acme/custom-embed-v1", "test-provider")

        assert model is not None
        assert model.model_type == ModelType.embedding
        assert model.metadata == {}

    def test_rerank_model_classified_by_name_heuristic_only(self):
        # models.dev's index only has embedding entries, so rerank classification
        # never consults it -- this must succeed purely from the identifier.
        model = classify_model("Qwen/Qwen3-Reranker-0.6B", "test-provider")

        assert model is not None
        assert model.model_type == ModelType.rerank
        assert model.metadata == {}

    def test_huggingface_provider_wins_over_other_providers(self):
        # Qwen/Qwen3-Embedding-8B exists in multiple providers. evroc records
        # output=40960 (the context window, not the embedding dimension).
        # huggingface correctly records output=4096. The index must prefer
        # huggingface so callers see the right embedding_dimension.
        model = classify_model("Qwen/Qwen3-Embedding-8B", "test-provider")

        assert model is not None
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension") == 4096

    def test_non_matching_identifier_returns_none(self):
        # A plain chat model: no "embed"/"rerank" in the name and not in the
        # models.dev embedding index. Callers fall back to their own default
        # classification (typically super().construct_model_from_identifier()).
        model = classify_model("qwen3-0.6b", "test-provider")

        assert model is None

    def test_returns_provider_id_and_identifier_on_returned_model(self):
        model = classify_model("some/embed-model", "my-provider")

        assert model is not None
        assert model.provider_id == "my-provider"
        assert model.identifier == "some/embed-model"
        assert model.provider_resource_id == "some/embed-model"

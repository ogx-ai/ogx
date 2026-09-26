# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for the shared models.dev classification used by remote adapters whose
/v1/models response has no model task/type field (vLLM, llama.cpp servers).
Exercised once here against the real models.dev data; adapter-specific test
files only need to check that they delegate to classify_model correctly."""

import pytest

from ogx.providers.utils.inference.models_dev_registry import classify_model, supports_reasoning
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


class TestSupportsReasoning:
    """supports_reasoning() gates Anthropic ``thinking`` on translated (non-native) providers."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-oss:20b",  # Ollama tag
            "openai/gpt-oss-20b",
            "Qwen/Qwen3-0.6B",  # not in models.dev: name heuristic
            "qwen3:0.6b",
            "deepseek-r1:1.5b",
            "deepseek-ai/DeepSeek-R1",
            "o3-mini",
            "o4-mini",
            "QwQ-32B",
            "Qwen3-Next-80B-A3B-Thinking",
        ],
    )
    def test_reasoning_models(self, model_id):
        assert supports_reasoning(model_id)

    @pytest.mark.parametrize(
        "model_id",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "llama3.2:3b-instruct-fp16",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Reranker-0.6B",
            "Qwen3-Next-80B-A3B-Instruct-FP8",
            "gpt-5-chat-latest",
            "qwen2.5:1.5b",
            "mistral:latest",
        ],
    )
    def test_non_reasoning_models(self, model_id):
        assert not supports_reasoning(model_id)

    def test_unknown_model_is_not_reasoning(self):
        assert not supports_reasoning("my-private-finetune-v2")

    def test_is_case_insensitive(self):
        assert supports_reasoning("GPT-OSS:20B")

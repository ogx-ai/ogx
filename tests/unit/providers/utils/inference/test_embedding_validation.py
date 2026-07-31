# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock

import pytest

from ogx.providers.inline.inference.sentence_transformers.config import SentenceTransformersInferenceConfig
from ogx.providers.inline.inference.sentence_transformers.sentence_transformers import (
    EMBEDDING_MODELS,
    SentenceTransformersInferenceImpl,
)
from ogx_api import OpenAIEmbeddingsRequestWithExtraBody, validate_embeddings_input_is_text


class TestEmbeddingValidation:
    """Test the validate_embeddings_input_is_text function."""

    def test_valid_string_input(self):
        """Test that string input is accepted."""
        params = OpenAIEmbeddingsRequestWithExtraBody(input="hello world", model="test-model")
        # Should not raise
        validate_embeddings_input_is_text(params)

    def test_valid_list_of_strings_input(self):
        """Test that list of strings is accepted."""
        params = OpenAIEmbeddingsRequestWithExtraBody(input=["hello", "world"], model="test-model")
        # Should not raise
        validate_embeddings_input_is_text(params)

    def test_invalid_list_of_ints_input(self):
        """Test that list of ints (token array) is rejected."""
        params = OpenAIEmbeddingsRequestWithExtraBody(input=[1, 2, 3], model="test-model")
        with pytest.raises(ValueError) as exc_info:
            validate_embeddings_input_is_text(params)

        error_msg = str(exc_info.value)
        assert "test-model" in error_msg
        assert "does not support token arrays" in error_msg

    def test_invalid_list_of_list_of_ints_input(self):
        """Test that list of list of ints (batch token array) is rejected."""
        params = OpenAIEmbeddingsRequestWithExtraBody(input=[[1, 2, 3], [4, 5, 6]], model="test-model")
        with pytest.raises(ValueError) as exc_info:
            validate_embeddings_input_is_text(params)

        error_msg = str(exc_info.value)
        assert "test-model" in error_msg
        assert "does not support token arrays" in error_msg

    def test_error_message_includes_model_name(self):
        """Test that error message includes the model name."""
        model_names = ["meta-llama/Llama-3.1-8B", "nomic-ai/nomic-embed-text-v1.5", "text-embedding-3-small"]

        for model in model_names:
            params = OpenAIEmbeddingsRequestWithExtraBody(input=[1, 2, 3], model=model)
            with pytest.raises(ValueError) as exc_info:
                validate_embeddings_input_is_text(params)

            error_msg = str(exc_info.value)
            assert model in error_msg


@pytest.fixture()
def clear_embedding_cache():
    EMBEDDING_MODELS.clear()
    yield
    EMBEDDING_MODELS.clear()


@pytest.fixture()
def mock_torch_st():
    """Inject fake torch and sentence_transformers modules so _load_model() doesn't need real deps."""
    import sys
    from types import ModuleType

    calls = []

    def fake_constructor(model_name, trust_remote_code=False):
        m = MagicMock(name=f"ST({model_name}, trust={trust_remote_code})")
        calls.append({"model": model_name, "trust_remote_code": trust_remote_code, "instance": m})
        return m

    fake_st_module = ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = fake_constructor

    fake_torch = ModuleType("torch")
    fake_torch.set_num_threads = MagicMock()

    original_st = sys.modules.get("sentence_transformers")
    original_torch = sys.modules.get("torch")

    sys.modules["sentence_transformers"] = fake_st_module
    sys.modules["torch"] = fake_torch

    yield calls

    if original_st is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = original_st

    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch


class TestEmbeddingCacheTrustRemoteCode:
    async def test_different_trust_remote_code_values_get_separate_cache_entries(
        self, clear_embedding_cache, mock_torch_st
    ):
        provider_trusted = SentenceTransformersInferenceImpl(
            SentenceTransformersInferenceConfig(trust_remote_code=True)
        )
        provider_untrusted = SentenceTransformersInferenceImpl(
            SentenceTransformersInferenceConfig(trust_remote_code=False)
        )

        model_a = await provider_trusted._load_sentence_transformer_model("test-model", trust_remote_code=True)
        model_b = await provider_untrusted._load_sentence_transformer_model("test-model", trust_remote_code=False)

        assert len(mock_torch_st) == 2
        assert mock_torch_st[0]["trust_remote_code"] is True
        assert mock_torch_st[1]["trust_remote_code"] is False
        assert model_a is not model_b

    async def test_same_trust_remote_code_uses_cache(self, clear_embedding_cache, mock_torch_st):
        provider = SentenceTransformersInferenceImpl(SentenceTransformersInferenceConfig(trust_remote_code=True))

        model_a = await provider._load_sentence_transformer_model("test-model", trust_remote_code=True)
        model_b = await provider._load_sentence_transformer_model("test-model", trust_remote_code=True)

        assert len(mock_torch_st) == 1
        assert model_a is model_b

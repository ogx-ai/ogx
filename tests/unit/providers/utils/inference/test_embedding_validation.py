# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

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

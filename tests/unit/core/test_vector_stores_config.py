# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx.core.datatypes import (
    ChunkRetrievalParams,
    QualifiedModel,
    RerankerModel,
    RewriteQueryParams,
    VectorStoresConfig,
)


class TestVectorStoresConfigValidation:
    """Test validation of VectorStoresConfig prompt templates."""

    def test_default_config_is_valid(self):
        """Test that default configuration passes all validation."""
        config = VectorStoresConfig()

        # Verify all sub-configs exist with valid templates
        assert config.file_search_params.header_template is not None
        assert config.context_prompt_params.chunk_annotation_template is not None
        assert config.annotation_prompt_params.chunk_annotation_template is not None

        # Verify required placeholders are present
        assert "{num_chunks}" in config.file_search_params.header_template
        assert "file_search" in config.file_search_params.header_template.lower()
        assert "{chunk.content}" in config.context_prompt_params.chunk_annotation_template
        assert "{query}" in config.context_prompt_params.context_template

    def test_default_search_mode_defaults_to_vector(self):
        """Test that default_search_mode defaults to 'vector' for backward compatibility."""
        config = VectorStoresConfig()
        assert config.chunk_retrieval_params.default_search_mode == "vector"

    def test_default_search_mode_can_be_set_to_hybrid(self):
        """Test that default_search_mode can be configured to 'hybrid'."""
        config = VectorStoresConfig(
            chunk_retrieval_params=ChunkRetrievalParams(default_search_mode="hybrid"),
        )
        assert config.chunk_retrieval_params.default_search_mode == "hybrid"

    def test_default_search_mode_can_be_set_to_keyword(self):
        """Test that default_search_mode can be configured to 'keyword'."""
        params = ChunkRetrievalParams(default_search_mode="keyword")
        assert params.default_search_mode == "keyword"

    def test_template_validation_errors(self):
        """Test that templates fail validation for common errors."""
        from ogx.core.datatypes import AnnotationPromptParams, ContextPromptParams, FileSearchParams

        # Empty templates fail
        with pytest.raises(ValidationError, match="must not be empty"):
            FileSearchParams(header_template="")

        # Missing required placeholders fail
        with pytest.raises(ValidationError, match="must contain {num_chunks} placeholder"):
            FileSearchParams(header_template="search found results")

        with pytest.raises(ValidationError, match="must contain 'file_search' keyword"):
            FileSearchParams(header_template="search found {num_chunks} results")

        with pytest.raises(ValidationError, match="must contain {chunk.content} placeholder"):
            ContextPromptParams(chunk_annotation_template="Result {index}: some content")

        with pytest.raises(ValidationError, match="must contain {query} placeholder"):
            ContextPromptParams(context_template="Retrieved results. Use as context.")

        with pytest.raises(ValidationError, match="must contain {file_id} placeholder"):
            AnnotationPromptParams(chunk_annotation_template="[{index}] {chunk_text}")

    def test_rewrite_query_params_validation(self):
        """Test RewriteQueryParams validation."""
        model = QualifiedModel(provider_id="test", model_id="test-model")

        # Valid config works
        valid_params = RewriteQueryParams(
            model=model, prompt="Expand this query: {query}", max_tokens=100, temperature=0.5
        )
        assert valid_params.prompt == "Expand this query: {query}"

        # Invalid configurations fail
        with pytest.raises(ValidationError, match="prompt must contain \\{query\\} placeholder"):
            RewriteQueryParams(model=model, prompt="No placeholder here")

        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            RewriteQueryParams(model=model, max_tokens=0)

        with pytest.raises(ValidationError, match="temperature must be between 0.0 and 2.0"):
            RewriteQueryParams(model=model, temperature=3.0)

    def test_custom_configuration(self):
        """Test complete custom configuration."""
        from ogx.core.datatypes import AnnotationPromptParams, ContextPromptParams, FileSearchParams

        config = VectorStoresConfig(
            default_provider_id="test-provider",
            default_embedding_model=QualifiedModel(provider_id="test", model_id="embedding-model"),
            default_reranker_model=RerankerModel(provider_id="test", model_id="reranker-model"),
            file_search_params=FileSearchParams(
                header_template="Custom file_search found {num_chunks} items:\nSTART\n", footer_template="END\n"
            ),
            context_prompt_params=ContextPromptParams(
                chunk_annotation_template="Item {index}: {chunk.content} | Meta: {metadata}\n",
                context_template='Results for "{query}": Use carefully.\n',
            ),
            annotation_prompt_params=AnnotationPromptParams(
                enable_annotations=False,
                annotation_instruction_template=" Custom citation format.",
                chunk_citation_template="[{index}] {metadata_text} --> {file_id}\n{chunk_text}\n",
            ),
        )

        assert config.default_provider_id == "test-provider"
        assert "Custom file_search" in config.file_search_params.header_template
        assert config.annotation_prompt_params.enable_annotations is False


class TestOptionalArchitecture:
    """Test optional sub-config architecture and constants fallback."""

    def test_guaranteed_defaults_behavior(self):
        """Test that sub-configs are always instantiated with defaults."""
        # Sub-configs are always instantiated due to default_factory
        config = VectorStoresConfig()
        assert config.file_search_params is not None
        assert config.context_prompt_params is not None
        assert config.annotation_prompt_params is not None
        assert "{num_chunks}" in config.file_search_params.header_template

    def test_guaranteed_defaults_have_expected_values(self):
        """Test that guaranteed defaults have expected hardcoded values."""
        # Create config with guaranteed defaults
        config = VectorStoresConfig()

        # Verify defaults have expected values
        header_template = config.file_search_params.header_template
        context_template = config.context_prompt_params.context_template

        assert header_template == "file_search tool found {num_chunks} chunks:\nBEGIN of file_search tool results.\n"
        assert (
            context_template
            == 'The above results were retrieved to help answer the user\'s query: "{query}". Use them as supporting information only in answering this query. {annotation_instruction}\n'
        )

        # Verify templates can be formatted successfully
        formatted_header = header_template.format(num_chunks=3)
        assert "3" in formatted_header
        assert "file_search" in formatted_header.lower()

        formatted_context = context_template.format(
            query="test query", annotation_instruction=" Cite sources properly."
        )
        assert "test query" in formatted_context

    def test_end_to_end_template_usage(self):
        """Test that guaranteed defaults lead to working template output."""
        # Create config with guaranteed defaults
        config = VectorStoresConfig()

        header_template = config.file_search_params.header_template
        chunk_template = config.context_prompt_params.chunk_annotation_template

        # Generate realistic output
        test_chunks = [
            {"content": "Paris is the capital of France.", "metadata": {"doc": "geo.pdf"}},
            {"content": "London is the capital of England.", "metadata": {"doc": "cities.txt"}},
        ]

        header_output = header_template.format(num_chunks=len(test_chunks))
        chunk_outputs = []
        for i, chunk_data in enumerate(test_chunks):

            class MockChunk:
                content = chunk_data["content"]

            chunk_output = chunk_template.format(index=i + 1, chunk=MockChunk(), metadata=chunk_data["metadata"])
            chunk_outputs.append(chunk_output)

        complete_output = header_output + "".join(chunk_outputs)

        # Verify output is substantial and contains expected content
        assert len(complete_output) > 100
        assert "file_search" in complete_output.lower()
        assert "Paris is the capital" in complete_output
        assert "London is the capital" in complete_output

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import ValidationError

from ogx.providers.inline.responses.builtin.config import CompactionConfig
from ogx.providers.inline.responses.builtin.responses.openai_responses import OpenAIResponsesImpl
from ogx_api.common.errors import InvalidParameterError
from ogx_api.openai_responses import (
    OpenAIResponseCompaction,
    OpenAIResponseMessage,
)


class TestCompactionConfigDefaults:
    def test_model_tokenizer_mappings_has_defaults(self):
        config = CompactionConfig()
        assert "llama" in config.model_tokenizer_mappings
        assert config.model_tokenizer_mappings["llama"] == "cl100k_base"

    def test_model_tokenizer_mappings_customizable(self):
        config = CompactionConfig(model_tokenizer_mappings={"mymodel": "o200k_base"})
        assert config.model_tokenizer_mappings == {"mymodel": "o200k_base"}
        assert "llama" not in config.model_tokenizer_mappings

    def test_valid_tokenizer_encoding_accepted(self):
        config = CompactionConfig(tokenizer_encoding="cl100k_base")
        assert config.tokenizer_encoding == "cl100k_base"

    def test_invalid_tokenizer_encoding_rejected_at_config(self):
        with pytest.raises(ValidationError, match="tokenizer_encoding"):
            CompactionConfig(tokenizer_encoding="not_a_real_encoding")

    def test_invalid_model_tokenizer_mapping_rejected_at_config(self):
        with pytest.raises(ValidationError, match="model_tokenizer_mappings"):
            CompactionConfig(model_tokenizer_mappings={"mymodel": "not_a_real_encoding"})


class TestResolveEncoding:
    """Tests for _resolve_encoding() — the 5-step tokenizer resolution chain."""

    def _make_impl(self, **config_kwargs) -> OpenAIResponsesImpl:
        """Create a minimal OpenAIResponsesImpl with just compaction_config set."""
        config = CompactionConfig(**config_kwargs)
        impl = object.__new__(OpenAIResponsesImpl)
        impl.compaction_config = config
        return impl

    def test_step1_per_request_override(self):
        impl = self._make_impl()
        encoding = impl._resolve_encoding("some-model", extra_body={"tokenizer_encoding": "cl100k_base"})
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_step1_per_request_invalid_raises(self):
        impl = self._make_impl()
        with pytest.raises(InvalidParameterError, match="tokenizer_encoding"):
            impl._resolve_encoding("some-model", extra_body={"tokenizer_encoding": "not_a_real_encoding"})

    def test_step1_per_request_beats_admin_default(self):
        impl = self._make_impl(tokenizer_encoding="o200k_base")
        encoding = impl._resolve_encoding("some-model", extra_body={"tokenizer_encoding": "cl100k_base"})
        assert encoding.name == "cl100k_base"

    def test_step2_admin_default(self):
        impl = self._make_impl(tokenizer_encoding="o200k_base")
        encoding = impl._resolve_encoding("some-model")
        assert encoding is not None
        assert encoding.name == "o200k_base"

    def test_step3_tiktoken_builtin_openai_model(self):
        impl = self._make_impl()
        encoding = impl._resolve_encoding("gpt-4o")
        assert encoding is not None

    def test_step3_tiktoken_strips_provider_prefix(self):
        impl = self._make_impl()
        encoding = impl._resolve_encoding("openai/gpt-4o")
        assert encoding is not None

    def test_step4_model_family_mapping_llama(self):
        impl = self._make_impl()
        encoding = impl._resolve_encoding("ollama/llama3.2:3b-instruct-fp16")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_step4_model_family_mapping_case_insensitive(self):
        impl = self._make_impl()
        encoding = impl._resolve_encoding("Llama-3.1-70B")
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_step4_model_family_custom_mapping(self):
        impl = self._make_impl(model_tokenizer_mappings={"mymodel": "o200k_base"})
        encoding = impl._resolve_encoding("mymodel-v2")
        assert encoding is not None
        assert encoding.name == "o200k_base"

    def test_step5_character_fallback_returns_none(self):
        impl = self._make_impl(model_tokenizer_mappings={})
        encoding = impl._resolve_encoding("totally-unknown-model-xyz")
        assert encoding is None


class TestExtractTextSegments:
    """Tests for _extract_text_segments() — shared helper for text extraction."""

    def test_extracts_message_content(self):
        items = [OpenAIResponseMessage(role="user", content="hello world")]
        segments = OpenAIResponsesImpl._extract_text_segments(items)
        assert segments == ["hello world"]

    def test_extracts_compaction_content(self):
        items = [OpenAIResponseCompaction(type="compaction", encrypted_content="summary")]
        segments = OpenAIResponsesImpl._extract_text_segments(items)
        assert segments == ["summary"]

    def test_extracts_mixed_items(self):
        items = [
            OpenAIResponseMessage(role="user", content="hello"),
            OpenAIResponseCompaction(type="compaction", encrypted_content="summary"),
        ]
        segments = OpenAIResponsesImpl._extract_text_segments(items)
        assert segments == ["hello", "summary"]

    def test_empty_list_returns_empty(self):
        assert OpenAIResponsesImpl._extract_text_segments([]) == []


class TestCountTokens:
    """Tests for _count_tokens() — encoding-based and character-based paths."""

    def _make_impl(self, **config_kwargs) -> OpenAIResponsesImpl:
        config = CompactionConfig(**config_kwargs)
        impl = object.__new__(OpenAIResponsesImpl)
        impl.compaction_config = config
        return impl

    def test_string_input_with_encoding(self):
        impl = self._make_impl(tokenizer_encoding="cl100k_base")
        count = impl._count_tokens("hello world", model="test")
        assert count > 0
        assert isinstance(count, int)

    def test_string_input_character_fallback(self):
        impl = self._make_impl(model_tokenizer_mappings={})
        count = impl._count_tokens("hello world test string", model="unknown-model-xyz")
        assert count == len("hello world test string") // 4

    def test_message_list_character_fallback(self):
        impl = self._make_impl(model_tokenizer_mappings={})
        messages = [
            OpenAIResponseMessage(role="user", content="hello world"),
        ]
        count = impl._count_tokens(messages, model="unknown-model-xyz")
        assert count == len("hello world") // 4

    def test_compaction_item_character_fallback(self):
        impl = self._make_impl(model_tokenizer_mappings={})
        items = [
            OpenAIResponseCompaction(type="compaction", encrypted_content="summary text here"),
        ]
        count = impl._count_tokens(items, model="unknown-model-xyz")
        assert count == len("summary text here") // 4

    def test_count_tokens_with_extra_body_override(self):
        impl = self._make_impl()
        count = impl._count_tokens(
            "hello world",
            model="unknown-model",
            extra_body={"tokenizer_encoding": "cl100k_base"},
        )
        assert count > 0

    def test_count_tokens_invalid_extra_body_raises(self):
        impl = self._make_impl()
        with pytest.raises(InvalidParameterError):
            impl._count_tokens(
                "hello",
                model="test",
                extra_body={"tokenizer_encoding": "bogus"},
            )

    def test_ollama_model_uses_family_mapping(self):
        impl = self._make_impl()
        count = impl._count_tokens("hello world", model="ollama/llama3.2:3b-instruct-fp16")
        assert count > 0


class TestMaybeAutoCompactExtraBody:
    """Verify that _maybe_auto_compact passes extra_body to _count_tokens."""

    def _make_impl(self, **config_kwargs) -> OpenAIResponsesImpl:
        config = CompactionConfig(**config_kwargs)
        impl = object.__new__(OpenAIResponsesImpl)
        impl.compaction_config = config
        return impl

    async def test_extra_body_invalid_encoding_raises_through_auto_compact(self):
        impl = self._make_impl(model_tokenizer_mappings={})
        context_management = [{"type": "compaction", "compact_threshold": 10}]
        with pytest.raises(InvalidParameterError, match="tokenizer_encoding"):
            await impl._maybe_auto_compact(
                input="some text here",
                model="unknown-model",
                context_management=context_management,
                previous_usage=None,
                extra_body={"tokenizer_encoding": "bogus"},
            )


class TestCompactEndpointExtraBody:
    """Verify compact_openai_response accepts extra_body for tokenizer override."""

    def test_compact_openai_response_signature_accepts_extra_body(self):
        """Verify the method signature accepts extra_body parameter."""
        import inspect

        sig = inspect.signature(OpenAIResponsesImpl.compact_openai_response)
        assert "extra_body" in sig.parameters

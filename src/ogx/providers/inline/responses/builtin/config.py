# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import tiktoken
from pydantic import BaseModel, Field, field_validator

from ogx.core.datatypes import VectorStoresConfig
from ogx.core.storage.datatypes import ResponsesStoreReference

DEFAULT_SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary "
    "for another LLM that will resume the task.\n\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)

DEFAULT_SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary of its "
    "thinking process. You also have access to the state of the tools that were used by "
    "that language model. Use this to build on the work that has already been done and "
    "avoid duplicating work. Here is the summary produced by the other language model, "
    "use the information in this summary to assist with your own analysis:"
)


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction behavior and prompt templates."""

    summarization_prompt: str = Field(
        default=DEFAULT_SUMMARIZATION_PROMPT,
        description="Prompt template used to instruct the model to summarize conversation history during compaction.",
    )
    summary_prefix: str = Field(
        default=DEFAULT_SUMMARY_PREFIX,
        description="Text prepended to the compaction summary to frame it as a handoff for the next LLM.",
    )
    summarization_model: str | None = Field(
        default=None,
        description="Model to use for generating compaction summaries. If not set, uses the same model as the conversation.",
    )
    default_compact_threshold: int | None = Field(
        default=None,
        description="Default token threshold for auto-compaction via context_management. If set, conversations exceeding this token count will be automatically compacted.",
    )
    tokenizer_encoding: str | None = Field(
        default=None,
        description=(
            "Default tiktoken encoding name for token counting (e.g. 'o200k_base', 'cl100k_base'). "
            "Applied as a server-level default after any per-request override via extra_body. "
            "If not set, encoding is resolved from the model name via tiktoken, then model-family "
            "prefix mappings, then character-based estimation."
        ),
    )
    model_tokenizer_mappings: dict[str, str] = Field(
        default_factory=lambda: {
            "llama": "cl100k_base",
            "mistral": "cl100k_base",
            "claude": "cl100k_base",
            "gemma": "cl100k_base",
            "qwen": "cl100k_base",
            "phi": "cl100k_base",
            "deepseek": "cl100k_base",
        },
        description=(
            "Map model name prefixes to tiktoken encoding names. "
            "Used as a heuristic fallback when tiktoken cannot resolve the model name directly. "
            "Matching is case-insensitive on the model name after stripping any provider prefix "
            "(e.g., 'ollama/llama3.2:3b' matches the 'llama' prefix). "
            "Admins can extend this to support custom or fine-tuned models."
        ),
    )

    @field_validator("tokenizer_encoding")
    @classmethod
    def validate_tokenizer_encoding(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                tiktoken.get_encoding(v)
            except ValueError:
                raise ValueError(
                    f"Failed to resolve tokenizer_encoding '{v}'. "
                    "Must be a valid tiktoken encoding name (e.g. 'o200k_base', 'cl100k_base')."
                ) from None
        return v

    @field_validator("model_tokenizer_mappings")
    @classmethod
    def validate_model_tokenizer_mappings(cls, v: dict[str, str]) -> dict[str, str]:
        for prefix, enc_name in v.items():
            try:
                tiktoken.get_encoding(enc_name)
            except ValueError:
                raise ValueError(
                    f"Failed to resolve model_tokenizer_mappings['{prefix}'] = '{enc_name}'. "
                    "Must be a valid tiktoken encoding name (e.g. 'o200k_base', 'cl100k_base')."
                ) from None
        return v


class ResponsesPersistenceConfig(BaseModel):
    """Nested persistence configuration for the responses provider."""

    responses: ResponsesStoreReference


class BuiltinResponsesImplConfig(BaseModel):
    """Configuration for the built-in responses with persistence and vector store settings."""

    persistence: ResponsesPersistenceConfig

    vector_stores_config: VectorStoresConfig | None = Field(
        default=None,
        description="Configuration for vector store prompt templates and behavior",
    )

    compaction_config: CompactionConfig = Field(
        default_factory=CompactionConfig,
        description="Configuration for conversation compaction behavior and prompt templates",
    )

    moderation_endpoint: str | None = Field(
        default=None,
        description="URL of an OpenAI-compatible /v1/moderations endpoint for guardrails. "
        'The endpoint must accept POST {"input": "text"} and return '
        '{"results": [{"flagged": bool, "categories": {...}}]}.',
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence": {
                "responses": ResponsesStoreReference(
                    backend="sql_default",
                    table_name="responses",
                ).model_dump(exclude_none=True),
            }
        }

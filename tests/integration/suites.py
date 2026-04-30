# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Central definition of integration test suites. You can use these suites by passing --suite=name to pytest.
# For example:
#
# ```bash
# pytest tests/integration/ --suite=vision --setup=ollama
# ```
#
"""
Each suite defines what to run (roots). Suites can be run with different global setups defined in setups.py.
Setups provide environment variables and model defaults that can be reused across multiple suites.

CLI examples:
  pytest tests/integration --suite=responses --setup=gpt
  pytest tests/integration --suite=vision --setup=ollama
  pytest tests/integration --suite=base --setup=vllm
"""

from pathlib import Path

from pydantic import BaseModel, Field

this_dir = Path(__file__).parent


class Suite(BaseModel):
    name: str
    roots: list[str]
    default_setup: str | None = None


class Setup(BaseModel):
    """A reusable test configuration with environment and CLI defaults."""

    name: str
    description: str
    defaults: dict[str, str | int] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


# Global setups - can be used with any suite "technically" but in reality, some setups might work
# only for specific test suites.
SETUP_DEFINITIONS: dict[str, Setup] = {
    "ollama": Setup(
        name="ollama",
        description="Local Ollama provider with text + safety models",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
            "SAFETY_MODEL": "ollama/llama-guard3:1b",
        },
        defaults={
            "text_model": "ollama/llama3.2:3b-instruct-fp16",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
            "safety_model": "ollama/llama-guard3:1b",
            "safety_shield": "llama-guard",
        },
    ),
    "ollama-vision": Setup(
        name="ollama",
        description="Local Ollama provider with a vision model",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "vision_model": "ollama/llama3.2-vision:11b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
        },
    ),
    "ollama-postgres": Setup(
        name="ollama-postgres",
        description="Server-mode tests with Postgres-backed persistence",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
            "SAFETY_MODEL": "ollama/llama-guard3:1b",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ogx",
            "POSTGRES_USER": "ogx",
            "POSTGRES_PASSWORD": "ogx",
            "OGX_LOGGING": "openai_responses=info",
        },
        defaults={
            "text_model": "ollama/llama3.2:3b-instruct-fp16",
            "embedding_model": "sentence-transformers/nomic-embed-text-v1.5",
            "safety_model": "ollama/llama-guard3:1b",
            "safety_shield": "llama-guard",
        },
    ),
    "vllm": Setup(
        name="vllm",
        description="vLLM provider with a text model",
        env={
            "VLLM_URL": "http://localhost:8000/v1",
        },
        defaults={
            "text_model": "vllm/Qwen/Qwen3-0.6B",
            "embedding_model": "sentence-transformers/nomic-embed-text-v1.5",
            "rerank_model": "vllm/Qwen/Qwen3-Reranker-0.6B",
        },
    ),
    "vllm-gpu-gpt-oss": Setup(
        name="vllm-gpu",
        description="vLLM GPU provider with gpt-oss:20b reasoning model (runs on GPU with quantization)",
        env={
            "VLLM_URL": "http://0.0.0.0:8000/v1",
        },
        defaults={
            "text_model": "vllm/gpt-oss:20b",
        },
    ),
    "ollama-reasoning": Setup(
        name="ollama",
        description="Local Ollama provider with a reasoning-capable model (gpt-oss)",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "text_model": "ollama/gpt-oss:20b",
        },
    ),
    "bedrock": Setup(
        name="bedrock",
        description=(
            "AWS Bedrock via OpenAI-compatible API (OpenAI GPT-OSS; "
            "see AWS Chat Completions docs). No default vision model — GPT-OSS is text-only; "
            "tests that require vision_model_id skip unless you pass --vision-model."
        ),
        defaults={
            "text_model": "bedrock/openai.gpt-oss-20b-1:0",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
        },
    ),
    "gpt": Setup(
        name="gpt",
        description="OpenAI GPT models for high-quality responses and tool calling",
        defaults={
            "text_model": "openai/gpt-4o",
            "vision_model": "openai/gpt-4o",
            "embedding_model": "openai/text-embedding-3-small",
            "embedding_dimension": 1536,
        },
    ),
    "gpt-reasoning": Setup(
        name="gpt-reasoning",
        description="OpenAI reasoning models (o4-mini) for reasoning effort tests",
        defaults={
            "text_model": "openai/o4-mini",
        },
    ),
    "azure": Setup(
        name="azure",
        description="Azure-hosted GPT models via the Azure OpenAI-compatible endpoint",
        defaults={
            "text_model": "azure/gpt-4o",
            "vision_model": "azure/gpt-4o",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
        },
    ),
    "watsonx": Setup(
        name="watsonx",
        description="IBM WatsonX AI models",
        defaults={
            "text_model": "watsonx/meta-llama/llama-3-3-70b-instruct",
        },
    ),
    "vertexai": Setup(
        name="vertexai",
        description="Google Vertex AI with Gemini models",
        defaults={
            "text_model": "vertexai/publishers/google/models/gemini-2.0-flash",
            "vision_model": "vertexai/publishers/google/models/gemini-2.0-flash",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
        },
    ),
    "tgi": Setup(
        name="tgi",
        description="Text Generation Inference (TGI) provider with a text model",
        env={
            "TGI_URL": "http://localhost:8080",
        },
        defaults={
            "text_model": "tgi/Qwen/Qwen3-0.6B",
        },
    ),
    "together": Setup(
        name="together",
        description="Together computer models",
        defaults={
            "text_model": "together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "embedding_model": "together/togethercomputer/m2-bert-80M-32k-retrieval",
        },
    ),
    "cerebras": Setup(
        name="cerebras",
        description="Cerebras models",
        defaults={
            "text_model": "cerebras/llama-3.3-70b",
        },
    ),
    "databricks": Setup(
        name="databricks",
        description="Databricks models",
        defaults={
            "text_model": "databricks/databricks-meta-llama-3-3-70b-instruct",
            "embedding_model": "databricks/databricks-bge-large-en",
        },
    ),
    "fireworks": Setup(
        name="fireworks",
        description="Fireworks provider with a text model",
        defaults={
            "text_model": "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            "embedding_model": "fireworks/accounts/fireworks/models/qwen3-embedding-8b",
        },
    ),
    "anthropic": Setup(
        name="anthropic",
        description="Anthropic Claude models",
        defaults={
            "text_model": "anthropic/claude-3-5-haiku-20241022",
        },
    ),
    "llama-api": Setup(
        name="llama-openai-compat",
        description="Llama models from https://api.llama.com",
        defaults={
            "text_model": "llama_openai_compat/Llama-3.3-8B-Instruct",
        },
    ),
    "gemini": Setup(
        name="gemini",
        description="Google Gemini models via GenAI API",
        defaults={
            "text_model": "gemini/gemini-2.5-flash-lite",
            "embedding_model": "gemini/text-embedding-004",
            "embedding_dimension": 768,
        },
    ),
    "groq": Setup(
        name="groq",
        description="Groq models",
        defaults={
            "text_model": "groq/llama-3.3-70b-versatile",
        },
    ),
    "llama-cpp-server": Setup(
        name="llama-cpp-server",
        description="llama.cpp server provider with OpenAI-compatible API",
        env={
            "LLAMA_CPP_SERVER_URL": "http://localhost:8080",
        },
        defaults={
            "text_model": "llama-cpp-server/qwen2.5",
            "embedding_model": "sentence-transformers/nomic-embed-text-v1.5",
        },
    ),
    "vllm-qwen3next": Setup(
        name="vllm-qwen3next",
        description="Qwen3-Next model for contextual retrieval validation",
        defaults={
            "text_model": "Qwen3-Next-80B-A3B-Instruct-FP8",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
        },
    ),
}


base_roots = [
    str(p)
    for p in this_dir.glob("*")
    if p.is_dir()
    and p.name not in ("__pycache__", "fixtures", "test_cases", "recordings", "responses", "messages", "interactions")
]

SUITE_DEFINITIONS: dict[str, Suite] = {
    "base": Suite(
        name="base",
        roots=base_roots,
        default_setup="ollama",
    ),
    "base-vllm-subset": Suite(
        name="base-vllm-subset",
        roots=["tests/integration/inference"],
        default_setup="vllm",
    ),
    "responses": Suite(
        name="responses",
        roots=["tests/integration/responses"],
        default_setup="gpt",
    ),
    "vision": Suite(
        name="vision",
        roots=["tests/integration/inference/test_vision_inference.py"],
        default_setup="ollama-vision",
    ),
    "vllm-reasoning": Suite(
        name="vllm-reasoning",
        roots=["tests/integration/responses/test_reasoning.py"],
        default_setup="vllm",
    ),
    "gpt-reasoning": Suite(
        name="gpt-reasoning",
        roots=[
            "tests/integration/responses/test_openai_responses.py::test_openai_response_reasoning_effort",
            "tests/integration/responses/test_openai_responses.py::test_openai_response_reasoning_effort_streaming",
        ],
        default_setup="gpt-reasoning",
    ),
    "ollama-reasoning": Suite(
        name="ollama-reasoning",
        roots=[
            "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_reasoning_passthrough",
            "tests/integration/responses/test_reasoning.py::test_reasoning_non_streaming",
            "tests/integration/responses/test_reasoning.py::test_reasoning_multi_turn_passthrough",
        ],
        default_setup="ollama-reasoning",
    ),
    "messages": Suite(
        name="messages",
        roots=["tests/integration/messages"],
        default_setup="ollama-reasoning",
    ),
    # Exercises the /v1/messages translation path: Anthropic request format is
    # translated to OpenAI Chat Completions, dispatched to OpenAI, and the response
    # is translated back to Anthropic format. OpenAI is not in _NATIVE_MESSAGES_MODULES,
    # so this setup guarantees the translation codepath in providers/inline/messages/impl.py
    # is covered end-to-end (rather than the native passthrough used by the ollama suite).
    "messages-openai": Suite(
        name="messages-openai",
        roots=["tests/integration/messages"],
        default_setup="gpt",
    ),
    "interactions": Suite(
        name="interactions",
        roots=["tests/integration/interactions"],
        default_setup="gemini",
    ),
    # Bedrock-specific tests with pre-recorded responses (no live API calls in CI)
    "bedrock": Suite(
        name="bedrock",
        roots=[
            "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_non_streaming",
            "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_streaming",
            "tests/integration/inference/test_openai_completion.py::test_inference_store",
        ],
        default_setup="bedrock",
    ),
    # Bedrock responses suite — subset of tests that reliably pass with GPT-OSS via
    # the Mantle API. Structured output, parallel tool calls, and some multi-turn
    # tool tests are excluded due to model capability gaps.
    "bedrock-responses": Suite(
        name="bedrock-responses",
        roots=[
            "tests/integration/responses/test_basic_responses.py",
            "tests/integration/responses/test_conversation_responses.py",
            "tests/integration/responses/test_mcp_authentication.py",
            "tests/integration/responses/test_prompt_templates.py",
            "tests/integration/responses/test_reasoning.py",
            "tests/integration/responses/test_responses_errors.py",
        ],
        default_setup="bedrock",
    ),
}

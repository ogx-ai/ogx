# Copyright (c) Meta Platforms, Inc. and affiliates.
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
            "POSTGRES_DB": "llamastack",
            "POSTGRES_USER": "llamastack",
            "POSTGRES_PASSWORD": "llamastack",
            "LLAMA_STACK_LOGGING": "openai_responses=info",
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
        description="AWS Bedrock provider with OpenAI GPT-OSS model (us-west-2)",
        defaults={
            "text_model": "bedrock/openai.gpt-oss-20b",
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
    and p.name not in ("__pycache__", "fixtures", "test_cases", "recordings", "responses", "post_training")
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
    "ollama-reasoning": Suite(
        name="ollama-reasoning",
        roots=[
            "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_reasoning_passthrough",
        ],
        default_setup="ollama-reasoning",
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
}

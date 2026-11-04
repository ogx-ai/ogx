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
    # Set for setups that only run under a dedicated workflow (e.g. the k8s auth
    # lane) rather than the standard CI matrix. Such setups must not appear in
    # ci_matrix.json (enforced by tests/unit/test_ci_matrix.py), but their
    # recordings are still referenced: cleanup_recordings.py collects the suites
    # that default to an excluded setup when computing orphans.
    matrix_excluded: bool = False


# Global setups - can be used with any suite "technically" but in reality, some setups might work
# only for specific test suites.
SETUP_DEFINITIONS: dict[str, Setup] = {
    "ollama": Setup(
        name="ollama",
        description="Local Ollama provider with text models",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "text_model": "ollama/qwen3.5:0.8b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
        },
    ),
    "ollama-vision": Setup(
        name="ollama",
        description="Local Ollama provider with a vision model",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "vision_model": "ollama/qwen3-vl:2b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
        },
    ),
    "ollama-postgres": Setup(
        name="ollama-postgres",
        description="Server-mode tests with Postgres-backed persistence",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "ogx",
            "POSTGRES_USER": "ogx",
            "POSTGRES_PASSWORD": "ogx",
            "OGX_LOGGING": "openai_responses=info",
        },
        defaults={
            "text_model": "ollama/qwen3.5:0.8b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
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
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
        },
    ),
    "vllm-systemone": Setup(
        name="vllm",
        description=(
            "vLLM server serving the Tev1-0.8B-experimental decision model on its structured "
            "decisions API (/v1/systemone), via vLLM's CPU next-token read strategy. Small "
            "enough for the CI-pinned CPU wheel, unlike SGLang's 9B clef-flash."
        ),
        env={
            "VLLM_URL": "http://localhost:8000/v1",
        },
        defaults={
            "judge_model": "vllm/togethercomputer/Tev1-0.8B-experimental",
        },
    ),
    "sglang": Setup(
        name="sglang",
        description=(
            "SGLang server (AMX CPU build in CI) serving a text model over its OpenAI-compatible "
            "endpoint. The URL must include the /v1 prefix because the OpenAI SDK appends "
            "endpoint paths to it; the adapter strips it again for the root-level /health "
            "probe. Embeddings come from the local sentence-transformers provider."
        ),
        env={
            "SGLANG_URL": "http://localhost:30000/v1",
        },
        defaults={
            "text_model": "sglang/Qwen/Qwen3-0.6B",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
        },
    ),
    "sglang-systemone": Setup(
        name="sglang",
        description=(
            "SGLang server serving the Tev1-0.8B-experimental decision model on its CPU "
            "next-token scoring path. Recording requires an AMX runner with the pinned CPU image. The URL must include "
            "the /v1 prefix because the OpenAI SDK appends endpoint paths to it."
        ),
        env={
            "SGLANG_URL": "http://localhost:30000/v1",
        },
        defaults={
            "judge_model": "sglang/togethercomputer/Tev1-0.8B-experimental",
        },
    ),
    "ollama-reasoning": Setup(
        name="ollama",
        description="Local Ollama provider with a reasoning-capable model (deepseek-r1)",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "text_model": "ollama/deepseek-r1:1.5b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
        },
    ),
    "ollama-systemone": Setup(
        name="ollama",
        description="Local Ollama provider with a SystemOne decision model (tev1:0.8b)",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434/v1",
        },
        defaults={
            "judge_model": "ollama/tev1:0.8b",
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
    # The gpt/azure/watsonx/vertexai setups default rerank_model to the ST reranker
    # because the recorded node IDs of rerank-parameterized tests (langchain/langgraph
    # in the responses suite) include the param value, and recording hashes are keyed
    # by test_id. Setups without ST-rerank recordings (e.g. vllm) intentionally omit it.
    "gpt": Setup(
        name="gpt",
        description="OpenAI GPT models for high-quality responses, tool calling, and reasoning effort tests",
        defaults={
            "text_model": "openai/gpt-5-nano",
            "vision_model": "openai/gpt-5-nano",
            "embedding_model": "openai/text-embedding-3-small",
            "embedding_dimension": 1536,
            "rerank_model": "sentence-transformers/Qwen/Qwen3-Reranker-0.6B",
        },
    ),
    # Kubernetes-auth access-control lane. Run only by
    # .github/workflows/integration-responses-conversations-auth-tests.yml, which
    # stands up a k8s auth provider the standard runner cannot; matrix_excluded
    # keeps it out of ci_matrix.json. Recording happens through the same workflow
    # in record mode (see record-integration-tests.yml).
    "auth": Setup(
        name="gpt",
        description="OpenAI text model for the kubernetes-auth access-control lane",
        matrix_excluded=True,
        defaults={
            "text_model": "openai/gpt-5-nano",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
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
            "rerank_model": "sentence-transformers/Qwen/Qwen3-Reranker-0.6B",
        },
    ),
    "watsonx": Setup(
        name="watsonx",
        description="IBM WatsonX AI models",
        defaults={
            "text_model": "watsonx/meta-llama/llama-3-3-70b-instruct",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
            "rerank_model": "sentence-transformers/Qwen/Qwen3-Reranker-0.6B",
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
            "rerank_model": "sentence-transformers/Qwen/Qwen3-Reranker-0.6B",
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
            "text_model": "cerebras/gpt-oss-120b",
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
        # Least expensive text model at https://docs.fireworks.ai/serverless/pricing
        defaults={
            "text_model": "fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
            "embedding_model": "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
        },
    ),
    "anthropic": Setup(
        name="anthropic",
        description="Anthropic Claude models",
        defaults={
            "text_model": "anthropic/claude-haiku-5-5",
        },
    ),
    "meta": Setup(
        name="meta",
        description="Meta AI models via the api.meta.ai endpoint",
        defaults={
            "text_model": "meta/muse-spark-1.3-contributor",
        },
    ),
    # DeepSeek serves /v1/messages from a separate host path (anthropic_base_url), not
    # from base_url, so both are set explicitly.
    "deepseek": Setup(
        name="deepseek",
        description="DeepSeek models via the DeepSeek platform API",
        env={
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
            "DEEPSEEK_ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        },
        defaults={
            "text_model": "deepseek/deepseek-chat",
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
    "mock": Setup(
        name="mock",
        description=(
            "Deterministic in-repo mock model backend (scripts/mock_backend.py) standing in "
            "for a text model and an embedding model behind the llama-cpp-server provider. "
            "scripts/integration-tests.sh starts the backend on the setup's port when this "
            "setup is selected. Every chat completion replies with the fixed 'sentinel' "
            "default, which tests assert on to prove a round-trip; keep it in sync with "
            "scripts/mock_backend.py."
        ),
        env={
            "LLAMA_CPP_SERVER_URL": "http://127.0.0.1:8090/v1",
        },
        defaults={
            "text_model": "llama-cpp-server/Llama-3.1-8B-Instruct",
            "embedding_model": "llama-cpp-server/text-embedding-3-small",
            "embedding_dimension": 1536,
            "sentinel": "OGX-E2E-SENTINEL-42",
        },
    ),
    "llama-cpp-server": Setup(
        name="llama-cpp-server",
        description=(
            "llama.cpp multi-model router (llama-server --models-preset) serving a text "
            "model, an embedding model, and a rerank model over one OpenAI-compatible "
            "endpoint. The URL must include the /v1 prefix because the OpenAI SDK appends "
            "endpoint paths to it."
        ),
        env={
            "LLAMA_CPP_SERVER_URL": "http://localhost:8080/v1",
        },
        defaults={
            # Model IDs are the router's INI section names; the provider strips the
            # "llama-cpp-server/" prefix and sends the section name as the model.
            "text_model": "llama-cpp-server/qwen3-0.6b",
            "embedding_model": "llama-cpp-server/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
            "rerank_model": "llama-cpp-server/bge-reranker-v2-m3",
        },
    ),
    "llama-cpp-server-systemone": Setup(
        name="llama-cpp-server",
        description=(
            "Local llama.cpp server (>= v0.6.0) serving the SystemOne decision model laya. "
            "The URL must include the /v1 prefix because the OpenAI SDK appends endpoint "
            "paths to it."
        ),
        env={
            "LLAMA_CPP_SERVER_URL": "http://localhost:8080/v1",
        },
        defaults={
            "judge_model": "llama-cpp-server/laya",
        },
    ),
    "text-embeddings-inference": Setup(
        name="text-embeddings-inference",
        description=(
            "HuggingFace Text-Embeddings-Inference server (CPU) serving an embedding model "
            "over its OpenAI-compatible endpoint. Embedding-only: suites that need a text "
            "model must not use this setup. The URL must include the /v1 prefix because the "
            "OpenAI SDK appends endpoint paths to it."
        ),
        env={
            "TEI_URL": "http://localhost:8080/v1",
        },
        defaults={
            "embedding_model": "text-embeddings-inference/nomic-ai/nomic-embed-text-v1.5",
            "embedding_dimension": 768,
        },
    ),
    "typesafe": Setup(
        name="typesafe",
        description=(
            "TypeSafe SystemOne (judgment) models, replayed from recordings. Set "
            "TYPESAFE_API_KEY to record against the live API; without it the recorded "
            "responses are replayed."
        ),
        defaults={
            "judge_model": "typesafe/jev-latest",
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
    # Full inference directory against a single self-hosted text backend (vllm, sglang,
    # llama-cpp-server, or ollama via default_setup). These CI servers serve a text,
    # embedding, and (for llama-cpp-server) rerank model, but no vision model, so
    # vision tests skip.
    "inference": Suite(
        name="inference",
        roots=["tests/integration/inference"],
        default_setup="ollama",
    ),
    # Text-Embeddings-Inference backend. TEI only serves embeddings (no chat or
    # completion endpoint), so this suite is scoped to the embeddings tests only.
    "text-embeddings-inference": Suite(
        name="text-embeddings-inference",
        roots=["tests/integration/inference/test_openai_embeddings.py"],
        default_setup="text-embeddings-inference",
    ),
    # TypeSafe SystemOne (judgment) backend. The typesafe setup only provides a
    # judge model (no chat/completion or embedding model), so this suite is scoped
    # to the SystemOne tests only, mirroring text-embeddings-inference.
    "systemone": Suite(
        name="systemone",
        roots=["tests/integration/inference/test_systemone.py"],
        default_setup="typesafe",
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
    # Reasoning tests self-skip when the model or provider lacks the relevant
    # capability (reasoning content, reasoning_effort, reasoning summaries), so
    # a single suite can be run against any reasoning-capable setup.
    "reasoning": Suite(
        name="reasoning",
        roots=[
            "tests/integration/responses/test_reasoning.py",
            "tests/integration/responses/test_openai_responses.py::test_openai_response_reasoning_effort",
            "tests/integration/responses/test_openai_responses.py::test_openai_response_reasoning_effort_streaming",
            "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_reasoning_passthrough",
        ],
        default_setup="gpt",
    ),
    # The gpt setup exercises the Anthropic-to-OpenAI translation path, since
    # OpenAI has no native /v1/messages endpoint.
    "messages": Suite(
        name="messages",
        roots=["tests/integration/messages"],
        default_setup="ollama",
    ),
    "interactions": Suite(
        name="interactions",
        roots=["tests/integration/interactions"],
        default_setup="gemini",
    ),
    # Kubernetes-auth access-control tests. Consumed only by the dedicated auth
    # workflow (scripts/integration-responses-conversations-auth-tests.sh); the
    # auth setup is matrix_excluded so this suite never runs in the standard CI
    # matrix against a non-auth server.
    "auth": Suite(
        name="auth",
        roots=[
            "tests/integration/conversations/test_openai_conversations.py::TestConversationAccessControl",
            "tests/integration/responses/test_prompts_access_control.py::TestPromptsAccessControl",
            "tests/integration/vector_io/test_vector_stores_access_control.py::TestVectorStoresAccessControl",
            "tests/integration/responses/test_responses_access_control.py::TestResponsesAccessControl",
        ],
        default_setup="auth",
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

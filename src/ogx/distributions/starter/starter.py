# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from ogx.core.datatypes import (
    BuildProvider,
    ModelInput,
    Provider,
    ProviderSpec,
    QualifiedModel,
    RerankerModel,
    SafetyConfig,
    ShieldInput,
    VectorStoresConfig,
)
from ogx.core.storage.kvstore.config import PostgresKVStoreConfig
from ogx.core.storage.sqlstore.sqlstore import PostgresSqlStoreConfig
from ogx.core.utils.dynamic import instantiate_class_type
from ogx.distributions.template import DistributionTemplate, RunConfigSettings
from ogx.providers.inline.file_processor.pypdf.config import PyPDFFileProcessorConfig
from ogx.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from ogx.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from ogx.providers.inline.inference.transformers.config import (
    TransformersInferenceConfig,
)
from ogx.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from ogx.providers.inline.vector_io.milvus.config import MilvusVectorIOConfig
from ogx.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from ogx.providers.registry.inference import available_providers
from ogx.providers.remote.tool_runtime.brave_search.config import BraveSearchToolConfig
from ogx.providers.remote.tool_runtime.tavily_search.config import TavilySearchToolConfig
from ogx.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from ogx.providers.remote.vector_io.elasticsearch.config import ElasticsearchVectorIOConfig
from ogx.providers.remote.vector_io.infinispan.config import InfinispanVectorIOConfig
from ogx.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from ogx.providers.remote.vector_io.qdrant.config import QdrantVectorIOConfig
from ogx.providers.remote.vector_io.weaviate.config import WeaviateVectorIOConfig
from ogx_api import RemoteProviderSpec


def _get_config_for_provider(provider_spec: ProviderSpec) -> dict[str, Any]:
    """Get configuration for a provider using its adapter's config class."""
    config_class = instantiate_class_type(provider_spec.config_class)

    if hasattr(config_class, "sample_run_config"):
        config: dict[str, Any] = config_class.sample_run_config()
        return config
    return {}


ENABLED_INFERENCE_PROVIDERS = [
    "ollama",
    "vllm",
    "fireworks",
    "together",
    "gemini",
    "vertexai",
    "groq",
    "sambanova",
    "anthropic",
    "openai",
    "cerebras",
    "nvidia",
    "bedrock",
    "azure",
]

INFERENCE_PROVIDER_IDS = {
    "ollama": "${env.OLLAMA_URL:+ollama}",
    "vllm": "${env.VLLM_URL:+vllm}",
    "cerebras": "${env.CEREBRAS_API_KEY:+cerebras}",
    "nvidia": "${env.NVIDIA_API_KEY:+nvidia}",
    "vertexai": "${env.VERTEX_AI_PROJECT:+vertexai}",
    "azure": "${env.AZURE_API_KEY:+azure}",
}


def get_remote_inference_providers() -> list[Provider]:
    """Build the list of remote inference providers enabled in the starter distribution.

    Returns:
        A list of Provider instances for enabled remote inference backends.
    """
    # Filter out inline providers and some others - the starter distro only exposes remote providers
    remote_providers = [
        provider
        for provider in available_providers()
        if isinstance(provider, RemoteProviderSpec) and provider.adapter_type in ENABLED_INFERENCE_PROVIDERS
    ]

    inference_providers = []
    for provider_spec in remote_providers:
        provider_type = provider_spec.adapter_type

        if provider_type in INFERENCE_PROVIDER_IDS:
            provider_id = INFERENCE_PROVIDER_IDS[provider_type]
        else:
            provider_id = provider_type.replace("-", "_").replace("::", "_")
        config = _get_config_for_provider(provider_spec)

        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_type}",
                config=config,
            )
        )
    return inference_providers


def get_distribution_template(name: str = "starter") -> DistributionTemplate:
    """Build the starter distribution template with multiple remote providers.

    Args:
        name: the distribution name.

    Returns:
        A DistributionTemplate configured for CPU-only environments with popular remote providers.
    """
    remote_inference_providers = get_remote_inference_providers()

    providers = {
        "inference": [BuildProvider(provider_type=p.provider_type, module=p.module) for p in remote_inference_providers]
        + [
            BuildProvider(provider_type="inline::sentence-transformers"),
            BuildProvider(provider_type="inline::transformers"),
        ],
        "vector_io": [
            BuildProvider(provider_type="inline::faiss"),
            BuildProvider(provider_type="inline::sqlite-vec"),
            BuildProvider(provider_type="inline::milvus"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
            BuildProvider(provider_type="remote::qdrant"),
            BuildProvider(provider_type="remote::weaviate"),
            BuildProvider(provider_type="remote::elasticsearch"),
            BuildProvider(provider_type="remote::infinispan"),
        ],
        "files": [BuildProvider(provider_type="inline::localfs")],
        "file_processors": [BuildProvider(provider_type="inline::pypdf")],
        "safety": [
            BuildProvider(provider_type="inline::llama-guard"),
            BuildProvider(provider_type="inline::code-scanner"),
        ],
        "interactions": [BuildProvider(provider_type="inline::builtin")],
        "messages": [BuildProvider(provider_type="inline::builtin")],
        "responses": [BuildProvider(provider_type="inline::builtin")],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::file-search"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
        "batches": [
            BuildProvider(provider_type="inline::reference"),
        ],
    }
    files_config = LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}")
    files_provider = Provider(
        provider_id="builtin-files",
        provider_type="inline::localfs",
        config=files_config,
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    reranker_provider = Provider(
        provider_id="transformers",
        provider_type="inline::transformers",
        config=TransformersInferenceConfig.sample_run_config(),
    )
    default_shields = [
        # if the
        ShieldInput(
            shield_id="llama-guard",
            provider_id="${env.SAFETY_MODEL:+llama-guard}",
            provider_shield_id="${env.SAFETY_MODEL:=}",
        ),
        ShieldInput(
            shield_id="code-scanner",
            provider_id="${env.CODE_SCANNER_MODEL:+code-scanner}",
            provider_shield_id="${env.CODE_SCANNER_MODEL:=}",
        ),
    ]
    postgres_sql_config = PostgresSqlStoreConfig.sample_run_config()
    postgres_kv_config = PostgresKVStoreConfig.sample_run_config()
    default_overrides = {
        "inference": remote_inference_providers + [embedding_provider, reranker_provider],
        "vector_io": [
            Provider(
                provider_id="faiss",
                provider_type="inline::faiss",
                config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="sqlite-vec",
                provider_type="inline::sqlite-vec",
                config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="${env.MILVUS_URL:+milvus}",
                provider_type="inline::milvus",
                config=MilvusVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="${env.CHROMADB_URL:+chromadb}",
                provider_type="remote::chromadb",
                config=ChromaVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}/",
                    url="${env.CHROMADB_URL:=}",
                ),
            ),
            Provider(
                provider_id="${env.PGVECTOR_DB:+pgvector}",
                provider_type="remote::pgvector",
                config=PGVectorVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    db="${env.PGVECTOR_DB:=}",
                    user="${env.PGVECTOR_USER:=}",
                    password="${env.PGVECTOR_PASSWORD:=}",
                ),
            ),
            Provider(
                provider_id="${env.QDRANT_URL:+qdrant}",
                provider_type="remote::qdrant",
                config=QdrantVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    url="${env.QDRANT_URL:=}",
                ),
            ),
            Provider(
                provider_id="${env.WEAVIATE_CLUSTER_URL:+weaviate}",
                provider_type="remote::weaviate",
                config=WeaviateVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    cluster_url="${env.WEAVIATE_CLUSTER_URL:=}",
                ),
            ),
            Provider(
                provider_id="${env.ELASTICSEARCH_URL:+elasticsearch}",
                provider_type="remote::elasticsearch",
                config=ElasticsearchVectorIOConfig.sample_run_config(
                    f"~/.llama/distributions/{name}",
                    elasticsearch_url="${env.ELASTICSEARCH_URL:=localhost:9200}",
                    elasticsearch_api_key="${env.ELASTICSEARCH_API_KEY:=}",
                ),
            ),
            Provider(
                provider_id="${env.INFINISPAN_URL:+infinispan}",
                provider_type="remote::infinispan",
                config=InfinispanVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
        ],
        "files": [files_provider],
        "file_processors": [
            Provider(
                provider_id="pypdf",
                provider_type="inline::pypdf",
                config=PyPDFFileProcessorConfig.sample_run_config(),
            ),
        ],
        "tool_runtime": [
            Provider(
                provider_id="brave-search",
                provider_type="remote::brave-search",
                config=BraveSearchToolConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="tavily-search",
                provider_type="remote::tavily-search",
                config=TavilySearchToolConfig.sample_run_config(f"~/.llama/distributions/{name}"),
            ),
            Provider(
                provider_id="file-search",
                provider_type="inline::file-search",
            ),
            Provider(
                provider_id="model-context-protocol",
                provider_type="remote::model-context-protocol",
            ),
        ],
    }

    # Claude model aliases for zero-config Claude Code compatibility
    claude_model_aliases = [
        ModelInput(
            model_id="claude-haiku-4-5-20251001",
            provider_id="all",
            provider_model_id="auto",
        ),
        ModelInput(
            model_id="claude-sonnet-4-5-20250514",
            provider_id="all",
            provider_model_id="auto",
        ),
        ModelInput(
            model_id="claude-opus-4-6-20260314",
            provider_id="all",
            provider_model_id="auto",
        ),
    ]

    base_run_settings = RunConfigSettings(
        provider_overrides=default_overrides,
        default_models=claude_model_aliases,
        default_shields=default_shields,
        default_connectors=[],
        vector_stores_config=VectorStoresConfig(
            default_provider_id="faiss",
            default_embedding_model=QualifiedModel(
                provider_id="sentence-transformers",
                model_id="nomic-ai/nomic-embed-text-v1.5",
            ),
            default_reranker_model=RerankerModel(
                provider_id="transformers",
                model_id="Qwen/Qwen3-Reranker-0.6B",
            ),
        ),
        safety_config=SafetyConfig(
            default_shield_id="llama-guard",
        ),
    )

    postgres_run_settings = base_run_settings.model_copy(
        update={
            "storage_backends": {
                "kv_default": postgres_kv_config,
                "sql_default": postgres_sql_config,
            }
        },
        deep=True,
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running OGX with several popular providers. This distribution is intended for CPU-only environments.",
        container_image=None,
        template_path=None,
        providers=providers,
        run_configs={
            "config.yaml": base_run_settings,
            "run-with-postgres-store.yaml": postgres_run_settings,
        },
        run_config_env_vars={
            "OGX_PORT": (
                "8321",
                "Port for the OGX distribution server",
            ),
            "FIREWORKS_API_KEY": (
                "",
                "Fireworks API Key",
            ),
            "OPENAI_API_KEY": (
                "",
                "OpenAI API Key",
            ),
            "GROQ_API_KEY": (
                "",
                "Groq API Key",
            ),
            "ANTHROPIC_API_KEY": (
                "",
                "Anthropic API Key",
            ),
            "GEMINI_API_KEY": (
                "",
                "Gemini API Key",
            ),
            "VERTEX_AI_PROJECT": (
                "",
                "Google Cloud Project ID for Vertex AI",
            ),
            "VERTEX_AI_LOCATION": (
                "global",
                "Google Cloud Location for Vertex AI",
            ),
            "SAMBANOVA_API_KEY": (
                "",
                "SambaNova API Key",
            ),
            "VLLM_URL": (
                "http://localhost:8000/v1",
                "vLLM URL",
            ),
            "VLLM_INFERENCE_MODEL": (
                "",
                "Optional vLLM Inference Model to register on startup",
            ),
            "OLLAMA_URL": (
                "http://localhost:11434",
                "Ollama URL",
            ),
            "AZURE_API_KEY": (
                "",
                "Azure API Key",
            ),
            "AZURE_API_BASE": (
                "",
                "Azure API Base",
            ),
            "AZURE_API_VERSION": (
                "",
                "Azure API Version",
            ),
            "AZURE_API_TYPE": (
                "azure",
                "Azure API Type",
            ),
            "INFINISPAN_URL": (
                "http://localhost:11222",
                "Infinispan server URL",
            ),
            "INFINISPAN_USERNAME": (
                "admin",
                "Infinispan authentication username",
            ),
            "INFINISPAN_PASSWORD": (
                "",
                "Infinispan authentication password",
            ),
        },
    )

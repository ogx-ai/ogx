# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.datatypes import (
    BuildProvider,
    ModelInput,
    Provider,
    ShieldInput,
)
from ogx.distributions.template import (
    DistributionTemplate,
    RunConfigSettings,
    get_model_registry,
)
from ogx.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from ogx.providers.remote.inference.anthropic.config import AnthropicConfig
from ogx.providers.remote.inference.gemini.config import GeminiConfig
from ogx.providers.remote.inference.groq.config import GroqConfig
from ogx.providers.remote.inference.openai.config import OpenAIConfig
from ogx.providers.remote.inference.together.config import TogetherImplConfig
from ogx.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from ogx.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from ogx.providers.utils.inference.model_registry import ProviderModelEntry
from ogx_api import ModelType


def get_inference_providers() -> tuple[list[Provider], dict[str, list[ProviderModelEntry]]]:
    """Build inference providers and their model registries for the open-benchmark distribution.

    Returns:
        A tuple of (list of Provider instances, mapping of provider IDs to model entries).
    """
    # in this template, we allow each API key to be optional
    providers = [
        (
            "openai",
            [
                ProviderModelEntry(
                    provider_model_id="gpt-4o",
                    model_type=ModelType.llm,
                )
            ],
            OpenAIConfig.sample_run_config(api_key="${env.OPENAI_API_KEY:=}"),
        ),
        (
            "anthropic",
            [
                ProviderModelEntry(
                    provider_model_id="claude-3-5-sonnet-latest",
                    model_type=ModelType.llm,
                )
            ],
            AnthropicConfig.sample_run_config(api_key="${env.ANTHROPIC_API_KEY:=}"),
        ),
        (
            "gemini",
            [
                ProviderModelEntry(
                    provider_model_id="gemini/gemini-1.5-flash",
                    model_type=ModelType.llm,
                )
            ],
            GeminiConfig.sample_run_config(api_key="${env.GEMINI_API_KEY:=}"),
        ),
        (
            "groq",
            [],
            GroqConfig.sample_run_config(api_key="${env.GROQ_API_KEY:=}"),
        ),
        (
            "together",
            [],
            TogetherImplConfig.sample_run_config(api_key="${env.TOGETHER_API_KEY:=}"),
        ),
    ]
    inference_providers = []
    available_models = {}
    for provider_id, model_entries, config in providers:
        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_id}",
                config=config,
            )
        )
        available_models[provider_id] = model_entries
    return inference_providers, available_models


def get_distribution_template() -> DistributionTemplate:
    """Build the open-benchmark distribution template for running evaluations.

    Returns:
        A DistributionTemplate configured for open benchmarking.
    """
    inference_providers, available_models = get_inference_providers()
    providers = {
        "inference": [BuildProvider(provider_type=p.provider_type, module=p.module) for p in inference_providers],
        "vector_io": [
            BuildProvider(provider_type="inline::sqlite-vec"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
        ],
        "safety": [BuildProvider(provider_type="inline::llama-guard")],
        "responses": [BuildProvider(provider_type="inline::builtin")],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::file-search"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
    }
    name = "open-benchmark"

    vector_io_providers = [
        Provider(
            provider_id="sqlite-vec",
            provider_type="inline::sqlite-vec",
            config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
        ),
        Provider(
            provider_id="${env.ENABLE_CHROMADB:+chromadb}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(
                f"~/.llama/distributions/{name}", url="${env.CHROMADB_URL:=}"
            ),
        ),
        Provider(
            provider_id="${env.ENABLE_PGVECTOR:+pgvector}",
            provider_type="remote::pgvector",
            config=PGVectorVectorIOConfig.sample_run_config(
                f"~/.llama/distributions/{name}",
                db="${env.PGVECTOR_DB:=}",
                user="${env.PGVECTOR_USER:=}",
                password="${env.PGVECTOR_PASSWORD:=}",
            ),
        ),
    ]

    models, _ = get_model_registry(available_models)
    default_models = models + [
        ModelInput(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            provider_id="groq",
            provider_model_id="groq/llama-3.3-70b-versatile",
            model_type=ModelType.llm,
        ),
        ModelInput(
            model_id="meta-llama/Llama-3.1-405B-Instruct",
            provider_id="together",
            provider_model_id="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            model_type=ModelType.llm,
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Distribution for running open benchmarks",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "config.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers,
                    "vector_io": vector_io_providers,
                },
                default_models=default_models,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
            ),
        },
        run_config_env_vars={
            "OGX_PORT": (
                "8321",
                "Port for the OGX distribution server",
            ),
            "TOGETHER_API_KEY": (
                "",
                "Together API Key",
            ),
            "OPENAI_API_KEY": (
                "",
                "OpenAI API Key",
            ),
            "GEMINI_API_KEY": (
                "",
                "Gemini API Key",
            ),
            "ANTHROPIC_API_KEY": (
                "",
                "Anthropic API Key",
            ),
            "GROQ_API_KEY": (
                "",
                "Groq API Key",
            ),
        },
    )

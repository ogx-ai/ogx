# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.datatypes import BuildProvider, Provider
from ogx.distributions.template import DistributionTemplate, RunConfigSettings
from ogx.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from ogx.providers.remote.inference.watsonx import WatsonXConfig


def get_distribution_template(name: str = "watsonx") -> DistributionTemplate:
    """Build the WatsonX distribution template.

    Args:
        name: the distribution name.

    Returns:
        A DistributionTemplate configured for IBM WatsonX inference.
    """
    providers = {
        "inference": [
            BuildProvider(provider_type="remote::watsonx"),
            BuildProvider(provider_type="inline::sentence-transformers"),
        ],
        "vector_io": [BuildProvider(provider_type="inline::faiss")],
        "safety": [BuildProvider(provider_type="inline::llama-guard")],
        "responses": [BuildProvider(provider_type="inline::builtin")],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::file-search"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
        "files": [BuildProvider(provider_type="inline::localfs")],
    }

    inference_provider = Provider(
        provider_id="watsonx",
        provider_type="remote::watsonx",
        config=WatsonXConfig.sample_run_config(),
    )

    files_provider = Provider(
        provider_id="builtin-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    return DistributionTemplate(
        name=name,
        distro_type="remote_hosted",
        description="Use watsonx for running LLM inference",
        container_image=None,
        template_path=None,
        providers=providers,
        run_configs={
            "config.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "files": [files_provider],
                },
                default_models=[],
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the OGX distribution server",
            ),
            "WATSONX_API_KEY": (
                "",
                "watsonx API Key",
            ),
            "WATSONX_PROJECT_ID": (
                "",
                "watsonx Project ID",
            ),
        },
    )

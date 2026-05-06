# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available safety provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::prompt-guard",
            pip_packages=[
                "transformers[accelerate]",
                "torch --index-url https://download.pytorch.org/whl/cpu",
            ],
            module="ogx.providers.inline.safety.prompt_guard",
            config_class="ogx.providers.inline.safety.prompt_guard.PromptGuardConfig",
            description="Prompt Guard safety provider for detecting and filtering unsafe prompts and content.",
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::llama-guard",
            pip_packages=[],
            module="ogx.providers.inline.safety.llama_guard",
            config_class="ogx.providers.inline.safety.llama_guard.LlamaGuardConfig",
            api_dependencies=[
                Api.inference,
            ],
            description="Llama Guard safety provider for content moderation and safety filtering using Meta's Llama Guard model.",
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::code-scanner",
            pip_packages=[
                "codeshield",
            ],
            module="ogx.providers.inline.safety.code_scanner",
            config_class="ogx.providers.inline.safety.code_scanner.CodeScannerConfig",
            description="Code Scanner safety provider for detecting security vulnerabilities and unsafe code patterns.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="bedrock",
            provider_type="remote::bedrock",
            pip_packages=["boto3"],
            module="ogx.providers.remote.safety.bedrock",
            config_class="ogx.providers.remote.safety.bedrock.BedrockSafetyConfig",
            description="AWS Bedrock safety provider for content moderation using AWS's safety services.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="nvidia",
            provider_type="remote::nvidia",
            pip_packages=["requests"],
            module="ogx.providers.remote.safety.nvidia",
            config_class="ogx.providers.remote.safety.nvidia.NVIDIASafetyConfig",
            description="NVIDIA's safety provider for content moderation and safety filtering.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="passthrough",
            provider_type="remote::passthrough",
            pip_packages=[],
            module="ogx.providers.remote.safety.passthrough",
            config_class="ogx.providers.remote.safety.passthrough.PassthroughSafetyConfig",
            provider_data_validator="ogx.providers.remote.safety.passthrough.config.PassthroughProviderDataValidator",
            description="Passthrough safety provider that forwards moderation calls to a downstream HTTP service.",
        ),
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="sambanova",
            provider_type="remote::sambanova",
            pip_packages=["litellm", "requests"],
            module="ogx.providers.remote.safety.sambanova",
            config_class="ogx.providers.remote.safety.sambanova.SambaNovaSafetyConfig",
            provider_data_validator="ogx.providers.remote.safety.sambanova.config.SambaNovaProviderDataValidator",
            description="SambaNova's safety provider for content moderation and safety filtering.",
        ),
    ]

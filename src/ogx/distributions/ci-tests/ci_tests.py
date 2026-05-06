# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx.core.datatypes import Provider
from ogx.distributions.template import DistributionTemplate
from ogx.providers.inline.inference.sentence_transformers.config import (
    SentenceTransformersInferenceConfig,
)
from ogx.providers.remote.inference.watsonx.config import WatsonXConfig
from ogx_api import ConnectorInput, ModelInput, ModelType

from ..starter.starter import get_distribution_template as get_starter_distribution_template


def get_distribution_template() -> DistributionTemplate:
    """Build the CI tests distribution template with test-specific overrides.

    Returns:
        A DistributionTemplate based on the starter template with CI-specific connectors and auth config.
    """
    template = get_starter_distribution_template(name="ci-tests")
    template.description = "CI tests for OGX"

    # Pre-register a test MCP connector used by test_response_connector_resolution_mcp_tool.
    # The test starts an MCP server on port 5199 and references it by connector_id.
    test_mcp_connector = ConnectorInput(
        connector_id="test-mcp-connector",
        url="http://localhost:5199/sse",
    )

    # Azure model must be pre-registered because the recording system cannot
    # replay model-list discovery calls against the Azure endpoint in CI.
    azure_model = ModelInput(
        model_id="azure/gpt-4o",
        provider_id="${env.AZURE_API_KEY:+azure}",
        provider_model_id="gpt-4o",
        model_type=ModelType.llm,
    )

    # WatsonX model must be pre-registered because the recording system cannot
    # replay model-list discovery calls against the WatsonX endpoint in CI.
    watsonx_model = ModelInput(
        model_id="watsonx/meta-llama/llama-3-3-70b-instruct",
        provider_id="${env.WATSONX_API_KEY:+watsonx}",
        provider_model_id="meta-llama/llama-3-3-70b-instruct",
        model_type=ModelType.llm,
    )

    # Vertex AI model must be pre-registered because the recording system cannot
    # replay model-list discovery calls against the Vertex AI endpoint in CI.
    vertexai_model = ModelInput(
        model_id="vertexai/publishers/google/models/gemini-2.0-flash",
        provider_id="${env.VERTEX_AI_PROJECT:+vertexai}",
        provider_model_id="publishers/google/models/gemini-2.0-flash",
        model_type=ModelType.llm,
    )

    # Bedrock model must be pre-registered because the recording system cannot
    # replay model-list discovery calls against the Bedrock endpoint in CI.
    # Gate on AWS_DEFAULT_REGION (required for both bearer-token and SigV4 modes)
    # rather than AWS_BEARER_TOKEN_BEDROCK so the model registers in OIDC/IRSA CI too.
    bedrock_model = ModelInput(
        model_id="bedrock/openai.gpt-oss-20b-1:0",
        provider_id="${env.AWS_DEFAULT_REGION:+bedrock}",
        provider_model_id="openai.gpt-oss-20b-1:0",
        model_type=ModelType.llm,
    )

    # Add conditional authentication config (disabled by default for CI tests)
    # This tests the conditional auth provider feature and provides a template for users
    # To enable: export AUTH_PROVIDER=enabled and configure the auth env vars
    auth_config = {
        # Authentication is disabled by default (AUTH_PROVIDER not set)
        # To enable: export AUTH_PROVIDER=enabled
        # Then configure the required auth provider settings below
        "provider_config": {
            "type": "${env.AUTH_PROVIDER:+oauth2_token}",
            "audience": "${env.AUTH_AUDIENCE:=ogx}",
            "issuer": "${env.AUTH_ISSUER:=}",
            "jwks": {
                "uri": "${env.AUTH_JWKS_URI:=}",
                "key_recheck_period": "${env.AUTH_JWKS_RECHECK_PERIOD:=3600}",
            },
            "verify_tls": "${env.AUTH_VERIFY_TLS:=true}",
        }
    }

    watsonx_provider = Provider(
        provider_id="${env.WATSONX_API_KEY:+watsonx}",
        provider_type="remote::watsonx",
        config=WatsonXConfig.sample_run_config(),
    )

    # Override sentence-transformers to use trust_remote_code=True for CI tests
    sentence_transformers_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig(trust_remote_code=True).model_dump(),
    )

    for run_config in template.run_configs.values():
        if run_config.default_connectors is None:
            run_config.default_connectors = []
        run_config.default_connectors.append(test_mcp_connector)

        if run_config.default_models is None:
            run_config.default_models = []
        run_config.default_models.append(azure_model)
        run_config.default_models.append(watsonx_model)
        run_config.default_models.append(vertexai_model)
        run_config.default_models.append(bedrock_model)

        # Add WatsonX inference provider (vertexai is already in starter distribution)
        run_config.provider_overrides["inference"].append(watsonx_provider)

        # Replace sentence-transformers provider with one that has trust_remote_code=True
        inference_providers = run_config.provider_overrides["inference"]
        for i, provider in enumerate(inference_providers):
            if provider.provider_id == "sentence-transformers":
                inference_providers[i] = sentence_transformers_provider
                break

        # Add conditional auth config
        run_config.auth_config = auth_config

    return template

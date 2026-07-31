# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import get_args, get_origin

import pytest
from pydantic import BaseModel, HttpUrl

from ogx.core.distribution import get_provider_registry, providable_apis
from ogx.core.utils.dynamic import instantiate_class_type


class TestProviderConfigurations:
    """Test suite for testing provider configurations across all API types."""

    @pytest.mark.parametrize("api", providable_apis())
    def test_api_providers(self, api):
        provider_registry = get_provider_registry()
        providers = provider_registry.get(api, {})

        failures = []
        for provider_type, provider_spec in providers.items():
            try:
                self._verify_provider_config(provider_type, provider_spec)
            except Exception as e:
                failures.append(f"Failed to verify {provider_type} config: {str(e)}")

        if failures:
            pytest.fail("\n".join(failures))

    def _verify_provider_config(self, provider_type, provider_spec):
        """Helper method to verify a single provider configuration."""
        # Get the config class
        config_class_name = provider_spec.config_class
        config_type = instantiate_class_type(config_class_name)

        assert issubclass(config_type, BaseModel), f"{config_class_name} is not a subclass of BaseModel"

        assert hasattr(config_type, "sample_run_config"), f"{config_class_name} does not have sample_run_config method"

        sample_config = config_type.sample_run_config(__distro_dir__="foobarbaz")
        assert isinstance(sample_config, dict), f"{config_class_name}.sample_run_config() did not return a dict"

    def test_remote_inference_url_standardization(self):
        """Verify all remote inference providers use standardized base_url configuration."""
        provider_registry = get_provider_registry()
        inference_providers = provider_registry.get("inference", {})

        # Filter for remote providers only
        remote_providers = {k: v for k, v in inference_providers.items() if k.startswith("remote::")}

        failures = []
        for provider_type, provider_spec in remote_providers.items():
            try:
                config_class_name = provider_spec.config_class
                config_type = instantiate_class_type(config_class_name)

                # Check that config has base_url field (not url)
                if hasattr(config_type, "model_fields"):
                    fields = config_type.model_fields

                    # Should NOT have 'url' field (old pattern)
                    if "url" in fields:
                        failures.append(
                            f"{provider_type}: Uses deprecated 'url' field instead of 'base_url'. "
                            f"Please rename to 'base_url' for consistency."
                        )

                    # Should have 'base_url' field with HttpUrl | None type
                    if "base_url" in fields:
                        field_info = fields["base_url"]
                        annotation = field_info.annotation

                        # Check if it's HttpUrl or HttpUrl | None
                        # get_origin() returns Union for (X | Y), None for plain types
                        # get_args() returns the types inside Union, e.g. (HttpUrl, NoneType)
                        is_valid = False
                        if get_origin(annotation) is not None:  # It's a Union/Optional
                            if HttpUrl in get_args(annotation):
                                is_valid = True
                        elif annotation == HttpUrl:  # Plain HttpUrl without | None
                            is_valid = True

                        if not is_valid:
                            failures.append(
                                f"{provider_type}: base_url field has incorrect type annotation. "
                                f"Expected 'HttpUrl | None', got '{annotation}'"
                            )

            except Exception as e:
                failures.append(f"{provider_type}: Error checking URL standardization: {str(e)}")

        if failures:
            pytest.fail("URL standardization violations found:\n" + "\n".join(f"  - {f}" for f in failures))

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    """Return the list of available batch processing provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.batches,
            provider_type="inline::reference",
            pip_packages=[],
            module="ogx.providers.inline.batches.reference",
            config_class="ogx.providers.inline.batches.reference.config.ReferenceBatchesImplConfig",
            api_dependencies=[
                Api.inference,
                Api.files,
                Api.models,
            ],
            description="Reference implementation of batches API with KVStore persistence.",
        ),
    ]

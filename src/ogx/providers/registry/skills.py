# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx_api import InlineProviderSpec
from ogx_api.datatypes import Api


def available_providers() -> list[InlineProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.skills,
            provider_type="inline::builtin",
            module="ogx.providers.inline.skills.builtin",
            config_class="ogx.providers.inline.skills.builtin.config.BuiltinSkillsConfig",
            description="Built-in skills provider using Files API for bundle storage.",
            api_dependencies=[Api.files],
        ),
    ]

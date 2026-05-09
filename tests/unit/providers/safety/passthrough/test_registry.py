# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from ogx.providers.registry.safety import available_providers


def test_passthrough_in_registry():
    providers = available_providers()
    provider_types = [p.provider_type for p in providers]
    assert "remote::passthrough" in provider_types


def test_passthrough_registry_has_provider_data_validator():
    providers = available_providers()
    passthrough = next(p for p in providers if p.provider_type == "remote::passthrough")
    assert passthrough.provider_data_validator is not None
    assert "PassthroughProviderDataValidator" in passthrough.provider_data_validator


def test_passthrough_registry_module_path():
    providers = available_providers()
    passthrough = next(p for p in providers if p.provider_type == "remote::passthrough")
    assert passthrough.module == "ogx.providers.remote.safety.passthrough"


def test_registry_alphabetical_order():
    providers = available_providers()
    remote_providers = [p for p in providers if p.provider_type.startswith("remote::")]
    adapter_types = [p.adapter_type for p in remote_providers]
    assert adapter_types == sorted(adapter_types)

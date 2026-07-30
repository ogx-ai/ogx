# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from . import skip_in_github_actions

# How to run this test:
#
# OGX_CONFIG="nvidia" pytest -v tests/integration/providers/nvidia/test_datastore.py


@pytest.fixture(autouse=True)
def skip_if_no_nvidia_provider(ogx_client):
    provider_types = {p.provider_type for p in ogx_client.providers.list() if p.api == "datasetio"}
    if "remote::nvidia" not in provider_types:
        pytest.skip("datasetio=remote::nvidia provider not configured, skipping")


# nvidia provider only
@skip_in_github_actions
@pytest.mark.parametrize(
    "provider_id",
    [
        "nvidia",
    ],
)
def test_register_and_unregister(ogx_client, provider_id):
    purpose = "eval/messages-answer"
    source = {
        "type": "uri",
        "uri": "hf://datasets/ogx/simpleqa?split=train",
    }
    dataset_id = f"test-dataset-{provider_id}"
    dataset = ogx_client.datasets.register(
        dataset_id=dataset_id,
        purpose=purpose,
        source=source,
        metadata={"provider_id": provider_id, "format": "json", "description": "Test dataset description"},
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == provider_id
    assert dataset.identifier == dataset_id

    dataset_list = ogx_client.datasets.list()
    provider_datasets = [d for d in dataset_list if d.provider_id == provider_id]
    assert any(provider_datasets)
    assert any(d.identifier == dataset_id for d in provider_datasets)

    ogx_client.datasets.unregister(dataset.identifier)
    dataset_list = ogx_client.datasets.list()
    provider_datasets = [d for d in dataset_list if d.identifier == dataset.identifier]
    assert not any(provider_datasets)

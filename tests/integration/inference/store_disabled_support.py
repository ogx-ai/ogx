# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared support for inference-store-disabled integration tests."""

from ogx.core.datatypes import StackConfig
from ogx.core.stack import get_stack_run_config_from_distro

TEXT_MODEL = "openai/gpt-4o"

NON_STREAMING_PROMPT = "Say hello."
STREAMING_PROMPT = "Say hello in one sentence."


def build_inference_store_disabled_run_config() -> StackConfig:
    """Build the minimal ci-tests configuration used by this test scenario."""
    run_config = get_stack_run_config_from_distro("ci-tests")
    # The ci-tests distribution always configures the inference store; disable
    # persistence through the explicit flag while keeping the reference valid.
    run_config.storage.stores.inference.enabled = False
    # Vector-store model validation is unrelated to chat completion persistence
    # and loads the sentence-transformers stack during an in-process boot.
    run_config.vector_stores = None
    return run_config

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""OpenAI client wrapper — configurable base_url is the only difference between backends."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_SAAS_URL = "https://api.openai.com/v1"
OGX_URL = os.getenv("OGX_URL", "http://localhost:8321/v1")


def create_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """Create an OpenAI client pointing at the given base_url.

    Args:
        base_url: API base URL. Defaults to OPENAI_SAAS_URL.
        api_key: API key. Defaults to OPENAI_API_KEY env var.
    """
    return OpenAI(
        base_url=base_url or OPENAI_SAAS_URL,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )


def backend_label(base_url: str) -> str:
    """Human-readable label for a base_url."""
    if "openai.com" in base_url:
        return "openai"
    return "ogx"

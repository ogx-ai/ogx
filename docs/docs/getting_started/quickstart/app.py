# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from openai import DefaultHttpxClient, OpenAI

client = OpenAI(
    base_url=os.environ["OGX_BASE_URL"],
    # Servers started with --no-auth accept any key
    api_key=os.environ.get("OGX_API_KEY", "unused"),
    # ogx go serves a self-signed certificate, so skip verification for local use
    http_client=DefaultHttpxClient(verify=False),
)

response = client.responses.create(
    model="ollama/llama3.2:3b",
    input="In one sentence, what is an API server?",
)
print(response.output_text)

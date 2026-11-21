# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from openai import DefaultHttpxClient, OpenAI

client = OpenAI(
    base_url="https://localhost:8321/v1",
    api_key=os.environ["OGX_API_KEY"],
    # ogx go serves a self-signed certificate, so skip verification for local use
    http_client=DefaultHttpxClient(verify=False),
)

# Upload a document
file = client.files.create(
    file=open("my-document.pdf", "rb"),
    purpose="assistants",
)

# Create a vector store and index the file
vector_store = client.vector_stores.create(
    name="my-docs",
    file_ids=[file.id],
)

# Ask questions with file search
response = client.responses.create(
    model="ollama/llama3.2:3b",
    input="Summarize the key points",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store.id],
        }
    ],
)
print(response.output_text)

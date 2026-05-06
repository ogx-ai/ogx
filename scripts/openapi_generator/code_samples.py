# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI Python client code samples for OpenAPI spec endpoints.

Adds x-codeSamples to OpenAI-compatible endpoints so that the
docusaurus-plugin-openapi-docs plugin renders OpenAI Python client
examples instead of the default http.client boilerplate.
"""

from typing import Any

# Each entry maps (path, method) to a Python code sample using the OpenAI client.
# Only OpenAI-compatible v1 endpoints are included.
_OPENAI_CODE_SAMPLES: dict[tuple[str, str], str] = {
    # --- Chat Completions ---
    ("/v1/chat/completions", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.chat.completions.list()
print(response)
""",
    ("/v1/chat/completions", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print(response.choices[0].message.content)
""",
    ("/v1/chat/completions/{completion_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

completion = client.chat.completions.retrieve("completion_id")
print(completion)
""",
    # --- Completions ---
    ("/v1/completions", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The capital of France is",
)
print(response.choices[0].text)
""",
    # --- Embeddings ---
    ("/v1/embeddings", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.embeddings.create(
    model="all-MiniLM-L6-v2",
    input="OGX is awesome",
)
print(response.data[0].embedding[:5])
""",
    # --- Models ---
    ("/v1/models", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

models = client.models.list()
for model in models:
    print(model.id)
""",
    ("/v1/models/{model_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

model = client.models.retrieve("meta-llama/Llama-3.1-8B-Instruct")
print(model)
""",
    # --- Files ---
    ("/v1/files", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

files = client.files.list()
for f in files:
    print(f.id, f.filename)
""",
    ("/v1/files", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

file = client.files.create(
    file=open("data.jsonl", "rb"),
    purpose="batch",
)
print(file.id)
""",
    ("/v1/files/{file_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

file = client.files.retrieve("file-abc123")
print(file)
""",
    ("/v1/files/{file_id}", "delete"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.files.delete("file-abc123")
print(response)
""",
    ("/v1/files/{file_id}/content", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

content = client.files.content("file-abc123")
print(content.text)
""",
    # --- Batches ---
    ("/v1/batches", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batches = client.batches.list()
for batch in batches:
    print(batch.id, batch.status)
""",
    ("/v1/batches", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.batches.create(
    input_file_id="file-abc123",
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print(batch.id)
""",
    ("/v1/batches/{batch_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.batches.retrieve("batch_abc123")
print(batch)
""",
    ("/v1/batches/{batch_id}/cancel", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.batches.cancel("batch_abc123")
print(batch)
""",
    # --- Responses ---
    ("/v1/responses", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

responses = client.responses.list()
for r in responses:
    print(r.id)
""",
    ("/v1/responses", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What is the capital of France?",
)
print(response.output_text)
""",
    ("/v1/responses/{response_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.responses.retrieve("resp_abc123")
print(response)
""",
    ("/v1/responses/{response_id}", "delete"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.responses.delete("resp_abc123")
print(response)
""",
    ("/v1/responses/{response_id}/input_items", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

input_items = client.responses.input_items.list("resp_abc123")
for item in input_items:
    print(item)
""",
    ("/v1/responses/{response_id}/cancel", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.responses.cancel("resp_abc123")
print(response)
""",
    # --- Moderations ---
    ("/v1/moderations", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.moderations.create(
    model="meta-llama/Llama-Guard-3-8B",
    input="Some text to classify",
)
print(response.results[0].flagged)
""",
    # --- Vector Stores ---
    ("/v1/vector_stores", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

vector_stores = client.vector_stores.list()
for vs in vector_stores:
    print(vs.id, vs.name)
""",
    ("/v1/vector_stores", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

vector_store = client.vector_stores.create(
    name="my-vector-store",
)
print(vector_store.id)
""",
    ("/v1/vector_stores/{vector_store_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

vector_store = client.vector_stores.retrieve("vs_abc123")
print(vector_store)
""",
    ("/v1/vector_stores/{vector_store_id}", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

vector_store = client.vector_stores.update(
    "vs_abc123",
    name="updated-name",
)
print(vector_store)
""",
    ("/v1/vector_stores/{vector_store_id}", "delete"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.vector_stores.delete("vs_abc123")
print(response)
""",
    # --- Vector Store Files ---
    ("/v1/vector_stores/{vector_store_id}/files", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

files = client.vector_stores.files.list("vs_abc123")
for f in files:
    print(f.id)
""",
    ("/v1/vector_stores/{vector_store_id}/files", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

file = client.vector_stores.files.create(
    vector_store_id="vs_abc123",
    file_id="file-abc123",
)
print(file)
""",
    ("/v1/vector_stores/{vector_store_id}/files/{file_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

file = client.vector_stores.files.retrieve(
    vector_store_id="vs_abc123",
    file_id="file-abc123",
)
print(file)
""",
    ("/v1/vector_stores/{vector_store_id}/files/{file_id}", "delete"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

response = client.vector_stores.files.delete(
    vector_store_id="vs_abc123",
    file_id="file-abc123",
)
print(response)
""",
    ("/v1/vector_stores/{vector_store_id}/files/{file_id}/content", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

content = client.vector_stores.files.content(
    vector_store_id="vs_abc123",
    file_id="file-abc123",
)
print(content)
""",
    # --- Vector Store File Batches ---
    ("/v1/vector_stores/{vector_store_id}/file_batches", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.vector_stores.file_batches.create(
    vector_store_id="vs_abc123",
    file_ids=["file-abc123", "file-def456"],
)
print(batch)
""",
    ("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.vector_stores.file_batches.retrieve(
    vector_store_id="vs_abc123",
    batch_id="vsfb_abc123",
)
print(batch)
""",
    ("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

batch = client.vector_stores.file_batches.cancel(
    vector_store_id="vs_abc123",
    batch_id="vsfb_abc123",
)
print(batch)
""",
    ("/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files", "get"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

files = client.vector_stores.file_batches.list_files(
    vector_store_id="vs_abc123",
    batch_id="vsfb_abc123",
)
for f in files:
    print(f.id)
""",
    # --- Vector Store Search ---
    ("/v1/vector_stores/{vector_store_id}/search", "post"): """\
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

results = client.vector_stores.search(
    vector_store_id="vs_abc123",
    query="What is OGX?",
)
for result in results:
    print(result)
""",
}

# Google GenAI SDK code samples for Interactions API endpoints.
_GOOGLE_CODE_SAMPLES: dict[tuple[str, str], str] = {
    ("/v1alpha/interactions", "post"): """\
from google import genai
from google.genai import types

client = genai.Client(
    api_key="fake",
    http_options=types.HttpOptions(
        base_url="http://localhost:8321",
        api_version="v1alpha",
    ),
)
interaction = client.interactions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What is the capital of France?",
)
print(interaction.outputs[0].text)
""",
}


# Anthropic SDK code samples for the Messages API endpoints.
_ANTHROPIC_CODE_SAMPLES: dict[tuple[str, str], list[dict[str, str]]] = {
    ("/v1/messages", "post"): [
        {
            "lang": "Python",
            "label": "Anthropic SDK",
            "source": """\
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8321/v1",
    api_key="fake",
)

message = client.messages.create(
    model="llama-3.3-70b",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is OGX?"}
    ],
)

print(message.content[0].text)""",
        },
        {
            "lang": "TypeScript",
            "label": "Anthropic SDK",
            "source": """\
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://localhost:8321/v1",
  apiKey: "fake",
});

const message = await client.messages.create({
  model: "llama-3.3-70b",
  max_tokens: 1024,
  messages: [{ role: "user", content: "What is OGX?" }],
});

console.log(message.content[0].text);""",
        },
    ],
}


def _add_anthropic_code_samples(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Add x-codeSamples with Anthropic SDK examples to Anthropic-compatible endpoints."""
    paths = openapi_schema.get("paths", {})
    samples_added = 0

    for (path, method), samples in _ANTHROPIC_CODE_SAMPLES.items():
        if path not in paths:
            continue
        if method not in paths[path]:
            continue

        operation = paths[path][method]
        if "x-codeSamples" not in operation:
            operation["x-codeSamples"] = []
        operation["x-codeSamples"].extend(samples)
        samples_added += len(samples)

    print(f"Added Anthropic SDK code samples to {samples_added} operations")
    return openapi_schema


def _add_openai_code_samples(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Add x-codeSamples with OpenAI Python client examples to OpenAI-compatible endpoints."""
    paths = openapi_schema.get("paths", {})
    samples_added = 0

    for (path, method), source_code in _OPENAI_CODE_SAMPLES.items():
        if path not in paths:
            continue
        if method not in paths[path]:
            continue

        code_sample = {
            "lang": "Python",
            "label": "OpenAI",
            "source": source_code.rstrip("\n"),
        }

        operation = paths[path][method]
        if "x-codeSamples" not in operation:
            operation["x-codeSamples"] = []
        operation["x-codeSamples"].append(code_sample)
        samples_added += 1

    print(f"Added OpenAI Python code samples to {samples_added} operations")
    return openapi_schema


def _add_google_code_samples(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Add x-codeSamples with Google GenAI SDK examples to Interactions API endpoints."""
    paths = openapi_schema.get("paths", {})
    samples_added = 0

    for (path, method), source_code in _GOOGLE_CODE_SAMPLES.items():
        if path not in paths:
            continue
        if method not in paths[path]:
            continue

        code_sample = {
            "lang": "Python",
            "label": "Google GenAI",
            "source": source_code.rstrip("\n"),
        }

        operation = paths[path][method]
        if "x-codeSamples" not in operation:
            operation["x-codeSamples"] = []
        operation["x-codeSamples"].append(code_sample)
        samples_added += 1

    print(f"Added Google GenAI Python code samples to {samples_added} operations")
    return openapi_schema

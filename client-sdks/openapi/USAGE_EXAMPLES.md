# ogx-open-client Usage Examples

This document provides end-to-end usage examples for the `ogx-open-client` Python SDK.

## Installation

```bash
# From PyPI (once published)
pip install ogx-open-client

# From TestPyPI (for testing)
pip install --index-url https://test.pypi.org/simple/ ogx-open-client

# From source (local development)
cd client-sdks/openapi
make sdk OPEN=1
cd sdks/python
pip install -e .
```

## Basic Usage

### Initialize the Client

```python
from ogx_open_client import OgxClient

# Connect to local OGX server
client = OgxClient(host="http://localhost:8000")

# Connect to remote server
client = OgxClient(host="https://api.example.com")

# With authentication
client = OgxClient(
    host="http://localhost:8000",
    header_name="Authorization",
    header_value="Bearer YOUR_API_KEY",
)
```

### Chat Completions

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Simple chat completion
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
# Output: "The capital of France is Paris."
```

### Streaming Chat Completions

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Streaming response
stream = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Write a short poem about Python."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # New line after stream completes
```

### Embeddings

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Generate embeddings
response = client.embeddings.create(
    model="text-embedding-3-small", input="The quick brown fox jumps over the lazy dog"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First few values: {embedding[:5]}")
```

### Tool Calling (Function Calling)

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# Request with tools
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)

# Check if model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

## Responses API (Agentic Orchestration)

### Basic Responses Request

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Create a responses session
response = client.responses.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
)

print(response.output.text)
```

### File Search with Responses API

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Upload files to vector store
vector_store = client.vector_stores.create(name="my-documents")

# Upload file
with open("document.pdf", "rb") as f:
    file = client.files.create(file=f, purpose="assistants")

client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

# Create responses session with file search
response = client.responses.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Summarize the document"}],
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

print(response.output.text)
```

## Async Usage

```python
import asyncio
from ogx_open_client import AsyncOgxClient


async def main():
    client = AsyncOgxClient(host="http://localhost:8000")

    # Async chat completion
    response = await client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": "Hello, async world!"}],
    )

    print(response.choices[0].message.content)

    # Async streaming
    stream = await client.chat.completions.create(
        model="llama-3.3-70b",
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# Run async code
asyncio.run(main())
```

## Models API

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# List available models
models = client.models.list()
for model in models:
    print(f"{model.identifier} - {model.model_type}")

# Get specific model details
model = client.models.retrieve(model_id="llama-3.3-70b")
print(f"Provider: {model.provider_id}")
print(f"Type: {model.model_type}")
```

## Vector Stores API

```python
from ogx_open_client import OgxClient

client = OgxClient(host="http://localhost:8000")

# Create vector store
vector_store = client.vector_stores.create(
    name="my-knowledge-base", metadata={"domain": "technical-docs"}
)

# List vector stores
stores = client.vector_stores.list()
for store in stores:
    print(f"{store.id}: {store.name}")

# Update vector store
updated = client.vector_stores.update(
    vector_store_id=vector_store.id, name="updated-knowledge-base"
)

# Delete vector store
client.vector_stores.delete(vector_store_id=vector_store.id)
```

## Error Handling

```python
from ogx_open_client import OgxClient
from ogx_open_client.exceptions import (
    BadRequestError,
    NotFoundError,
    RateLimitError,
    InternalServerError,
)

client = OgxClient(host="http://localhost:8000")

try:
    response = client.chat.completions.create(
        model="non-existent-model", messages=[{"role": "user", "content": "Hello"}]
    )
except NotFoundError as e:
    print(f"Model not found: {e.message}")
except BadRequestError as e:
    print(f"Bad request: {e.message}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e.message}")
except InternalServerError as e:
    print(f"Server error: {e.message}")
```

## Context Manager Usage

```python
from ogx_open_client import OgxClient

# Automatic cleanup with context manager
with OgxClient(host="http://localhost:8000") as client:
    response = client.chat.completions.create(
        model="llama-3.3-70b", messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)
# Client resources automatically cleaned up here
```

## Advanced Configuration

```python
from ogx_open_client import OgxClient, Configuration

# Custom configuration
config = Configuration(
    host="http://localhost:8000",
    # Custom timeout (in seconds)
    timeout=60,
)

client = OgxClient(configuration=config)

# Custom headers
client = OgxClient(
    host="http://localhost:8000",
    default_headers={"X-Custom-Header": "value", "X-Request-ID": "unique-request-id"},
)
```

## Comparison with Official SDK

The `ogx-open-client` provides the same functionality as the official `ogx_client` SDK:

```python
# Official SDK (ogx_client)
from ogx_client import OgxClient as OfficialClient

official = OfficialClient(base_url="http://localhost:8000")

# OpenAPI SDK (ogx-open-client)
from ogx_open_client import OgxClient as OpenAPIClient

openapi = OpenAPIClient(host="http://localhost:8000")

# Both provide identical API surface
# Choose based on your tooling preferences
```

## Troubleshooting

### Import Error

If you see: `ModuleNotFoundError: No module named 'ogx_open_client'`

Solution: Ensure the package is installed:

```bash
pip install ogx-open-client
```

### Connection Error

```python
# If you see: Connection refused
# Solution: Ensure OGX server is running
# Start server: uv run ogx stack run starter
```

### Type Checking

```python
# For type checking with mypy
from ogx_open_client import OgxClient
from ogx_open_client.types import ChatCompletion, Message


def process_response(response: ChatCompletion) -> str:
    return response.choices[0].message.content
```

## Additional Resources

- **Main Documentation**: <https://ogx-ai.github.io/docs>
- **OpenAI API Reference**: <https://platform.openai.com/docs/api-reference> (OGX is compatible)
- **GitHub Issues**: <https://github.com/ogx-ai/ogx/issues>
- **Discord Community**: <https://discord.gg/ZAFjsrcw>

---

**Note**: This SDK is auto-generated from the OGX OpenAPI specification. For the latest examples and API coverage, refer to the main OGX documentation.

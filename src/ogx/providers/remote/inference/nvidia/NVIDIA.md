# NVIDIA Inference Provider for OGX

This provider enables running inference using NVIDIA NIM.

## Features

- Endpoints for completions, chat completions, and embeddings for registered models

## Getting Started

### Prerequisites

- OGX with NVIDIA configuration
- Access to NVIDIA NIM deployment
- NIM for model to use for inference is deployed

### Setup

Build the NVIDIA environment:

```bash
uv pip install ogx-client
uv run ogx list-deps nvidia | xargs -L1 uv pip install
```

### Basic Usage using the OGX Python Client

#### Initialize the client

```python
import os

os.environ["NVIDIA_API_KEY"] = (
    ""  # Required if using hosted NIM endpoint. If self-hosted, not required.
)
os.environ["NVIDIA_BASE_URL"] = "http://nim.test"  # NIM URL

from ogx.core.library_client import OGXAsLibraryClient

client = OGXAsLibraryClient("nvidia")
client.initialize()
```

### Create Chat Completion

The following example shows how to create a chat completion for an NVIDIA NIM.

```python
response = client.chat.completions.create(
    model="nvidia/meta/llama-3.1-8b-instruct",
    messages=[
        {
            "role": "system",
            "content": "You must respond to each message with only one word",
        },
        {
            "role": "user",
            "content": "Complete the sentence using one word: Roses are red, violets are:",
        },
    ],
    stream=False,
    max_tokens=50,
)
print(f"Response: {response.choices[0].message.content}")
```

### Tool Calling Example

The following example shows how to do tool calling for an NVIDIA NIM.

```python
tool_definition = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit (celsius or fahrenheit)",
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    },
}

tool_response = client.chat.completions.create(
    model="nvidia/meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=[tool_definition],
)

print(f"Response content: {tool_response.choices[0].message.content}")
if tool_response.choices[0].message.tool_calls:
    for tool_call in tool_response.choices[0].message.tool_calls:
        print(f"Tool Called: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### Structured Output Example

The following example shows how to do structured output for an NVIDIA NIM.

```python
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "occupation": {"type": "string"},
    },
    "required": ["name", "age", "occupation"],
}

structured_response = client.chat.completions.create(
    model="nvidia/meta/llama-3.1-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Create a profile for a fictional person named Alice who is 30 years old and is a software engineer. ",
        }
    ],
    extra_body={"nvext": {"guided_json": person_schema}},
)
print(f"Structured Response: {structured_response.choices[0].message.content}")
```

### Create Embeddings

The following example shows how to create embeddings for an NVIDIA NIM.

```python
response = client.embeddings.create(
    model="nvidia/nvidia/llama-3.2-nv-embedqa-1b-v2",
    input=["What is the capital of France?"],
    extra_body={"input_type": "query"},
)
print(f"Embeddings: {response.data}")
```

### Vision Language Models Example

The following example shows how to run vision inference by using an NVIDIA NIM.

```python
def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        img_bytes = image_file.read()
        return base64.b64encode(img_bytes).decode("utf-8")


image_path = {path_to_the_image}
demo_image_b64 = load_image_as_base64(image_path)

vlm_response = client.chat.completions.create(
    model="nvidia/meta/llama-3.2-11b-vision-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{demo_image_b64}",
                    },
                },
                {
                    "type": "text",
                    "text": "Please describe what you see in this image in detail.",
                },
            ],
        }
    ],
)

print(f"VLM Response: {vlm_response.choices[0].message.content}")
```

### Rerank Example

The following example shows how to rerank documents using an NVIDIA NIM.

```python
rerank_response = client.alpha.inference.rerank(
    model="nvidia/nvidia/llama-3.2-nv-rerankqa-1b-v2",
    query="query",
    items=[
        "item_1",
        "item_2",
        "item_3",
    ],
)

for i, result in enumerate(rerank_response):
    print(f"{i+1}. [Index: {result.index}, " f"Score: {(result.relevance_score):.3f}]")
```

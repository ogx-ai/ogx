# NVIDIA Safety Provider for OGX

This provider enables safety checks and guardrails for LLM interactions using NVIDIA's NeMo Guardrails service.

## Features

- Run safety checks for messages

## Getting Started

### Prerequisites

- OGX with NVIDIA configuration
- Access to NVIDIA NeMo Guardrails service
- NIM for model to use for safety check is deployed

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

os.environ["NVIDIA_API_KEY"] = "your-api-key"
os.environ["NVIDIA_GUARDRAILS_URL"] = "http://guardrails.test"

from ogx.core.library_client import OGXAsLibraryClient

client = OGXAsLibraryClient("nvidia")
client.initialize()
```

#### Create a safety shield

```python
from ogx_api.safety import Shield
from ogx_api.inference import Message

# Create a safety shield
shield = Shield(
    shield_id="your-shield-id",
    provider_resource_id="safety-model-id",  # The model to use for safety checks
    description="Safety checks for content moderation",
)

# Register the shield
await client.safety.register_shield(shield)
```

#### Run safety checks

```python
# Messages to check
messages = [Message(role="user", content="Your message to check")]

# Run safety check
response = await client.safety.run_shield(
    shield_id="your-shield-id",
    messages=messages,
)

# Check for violations
if response.violation:
    print(f"Safety violation detected: {response.violation.user_message}")
    print(f"Violation level: {response.violation.violation_level}")
    print(f"Metadata: {response.violation.metadata}")
else:
    print("No safety violations detected")
```

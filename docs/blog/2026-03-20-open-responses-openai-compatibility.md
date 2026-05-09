---
slug: open-responses-openai-compatibility
title: "OGX Achieves 100% Open Responses Compliance: Enterprise-Grade OpenAI Compatibility for Your Infrastructure"
authors: [franciscojavierarceo, cdoern]
tags: [openai-compatibility, open-responses, enterprise, mcp, connectors]
date: 2026-03-20
---

We're excited to share that OGX has achieved **100% compliance with the Open Responses specification** and been officially recognized as part of the [Open Responses community](https://github.com/openresponses/openresponses/pull/29). This milestone represents more than just compatibility: it's about bringing enterprise-grade AI capabilities to your own infrastructure with the familiarity of OpenAI APIs.

With comprehensive support for Files, Vector Stores, Search, Conversations, Prompts, Chat Completions, the full Responses API, plus powerful extensions like MCP tool integration, Tool Calling, and Connectors, OGX offers something unique in the AI infrastructure landscape: a SaaS-like experience that runs entirely on your terms.

<!--truncate-->

## Recognition by the Open Responses Community

The [Open Responses initiative](https://www.openresponses.org/) represents a collaborative effort to standardize agentic AI interfaces across the industry, with backing from OpenAI, Hugging Face, and leading providers like Ollama, vLLM, and LM Studio. Our acceptance into this community validates OGX's commitment to open standards and interoperability.

What makes this recognition particularly meaningful is our approach to compliance. We don't just aim for compatibility—**we run the full Open Responses acceptance test suite on every pull request as a blocking requirement**. This means our perfect 6/6 test pass rate isn't a one-time achievement; it's a maintained standard that ensures consistent, reliable behavior for developers building on open standards.

## Comprehensive OpenAI API Feature Support

OGX delivers comprehensive feature parity across multiple API surfaces, giving you the full power of modern AI APIs.

> **A note on model IDs:** The model ID you pass depends on the inference provider backing your OGX server. For example, with Ollama you'd use `ollama/llama3.2:3b`, while with Fireworks or Together you'd use the HuggingFace-style `meta-llama/Llama-3.2-3B-Instruct`. The API calls are identical either way.

### **Files API** - OpenAI-Compatible Document Management

Upload, manage, and process documents with the same interface you'd use with OpenAI:

```python
# Works identically with OpenAI or OGX clients
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

file = client.files.create(file=open("document.pdf", "rb"), purpose="assistants")

# List and manage files
files = client.files.list()
content = client.files.content(file.id)
```

### **Vector Stores API** - RAG Without Vendor Lock-in

Build retrieval-augmented generation applications using the full Vector Stores API:

```python
# Create vector stores with nested file management
vector_store = client.vector_stores.create(name="knowledge_base")

# Add files and manage vector store content
client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

# Search functionality built-in
results = client.vector_stores.search(
    vector_store_id=vector_store.id, query="What is our refund policy?"
)
```

### **Conversations API** - Persistent Context Management

Manage conversation state and continuity across interactions:

```python
# Create a conversation
conversation = client.conversations.create()

# Add items to a conversation
client.conversations.items.create(
    conversation_id=conversation.id,
    items=[
        {"role": "user", "content": "Tell me about our product features"},
        {"role": "assistant", "content": "I'd be happy to explain..."},
    ],
)

# Retrieve conversation history
items = client.conversations.items.list(conversation_id=conversation.id)
```

### **Chat Completions & Responses** - Simple Chat to Agentic Workflows

From straightforward inference to multi-tool orchestration:

```python
# Standard chat completions (e.g., with Ollama)
completion = client.chat.completions.create(
    model="ollama/gpt-oss:20b", messages=[{"role": "user", "content": "Explain RAG"}]
)

# Advanced responses with tool orchestration (e.g., with Fireworks)
response = client.responses.create(
    model="ollama/gpt-oss:20b",
    input="What documents mention our pricing strategy?",
    tools=[{"type": "file_search"}],
)
```

### **Prompts API** - Programmatic Prompt Management

OGX extends OpenAI compatibility with full programmatic prompt management. With OpenAI, prompts are created through their admin portal and referenced by ID in the Responses API. OGX provides the same referencing pattern, plus a complete CRUD API for creating and managing prompts programmatically:

```python
from ogx_client import OgxClient

ls_client = OgxClient()

# Create reusable prompt templates with variables
prompt = ls_client.prompts.create(
    prompt="You are a {{ role }} assistant. Analyze this: {{ content }}",
    variables=["role", "content"],
)

# Reference prompts in responses — compatible with OpenAI's pattern
response = client.responses.create(
    model="ollama/gpt-oss:20b",
    input=[{"role": "user", "content": "Review our Q1 report"}],
    prompt={
        "id": prompt.prompt_id,
        "variables": {
            "role": {"type": "input_text", "text": "financial analyst"},
            "content": {"type": "input_text", "text": "Q1 2026 earnings report"},
        },
    },
)
```

This gives you the best of both worlds: compatibility with OpenAI's prompt referencing pattern in the Responses API, plus the ability to manage prompts as code rather than through a web interface.

### **MCP Integration** - Extensible Tool Ecosystem

Leverage the Model Context Protocol to connect to any MCP server and dynamically discover tools:

```python
# Connect to MCP servers for dynamic tool discovery
response = client.responses.create(
    model="ollama/gpt-oss:20b",
    input="What parks are in Rhode Island, and are there upcoming events?",
    tools=[
        {
            "type": "mcp",
            "server_label": "parks-service",
            "server_url": "http://parks-mcp-server:8000/sse",
        }
    ],
)
```

MCP tools support per-request authorization, allowed tool filtering, and automatic session management. Connect to databases, APIs, and internal services through the growing ecosystem of standard MCP servers—no custom integration work required.

### **Connectors** - Declarative Service Integration

Connectors provide a configuration-driven approach to integrating external services with your OGX deployment. Define your data sources and services in your stack configuration, and they're automatically available as tools for your agents to use.

## The Value Proposition: SaaS Experience, Your Infrastructure

### **Data Sovereignty & Security**

For regulated industries like finance, healthcare, and government, sending sensitive documents to external APIs isn't an option. OGX solves this by running entirely on your infrastructure:

- **Documents never leave your environment**: RAG pipelines, vector storage, and model inference all happen locally
- **Compliance-ready**: Meet HIPAA, SOC 2, GDPR, and other regulatory requirements
- **Audit trails**: Full visibility into data processing and model decisions

### **Cost Control & Predictability**

Unlike consumption-based pricing models, OGX offers:

- **Fixed infrastructure costs**: Pay for compute, not tokens
- **No usage surprises**: Predictable costs regardless of application load
- **Efficient resource utilization**: Choose the right model size for your use case

### **Model Freedom**

Break free from vendor-specific models:

```python
# Same API, different models — swap without code changes
for model in ["ollama/gpt-oss:20b", "ollama/llama3.2:3b", "your-org/custom-model"]:
    response = client.chat.completions.create(model=model, messages=messages)
```

## Getting Started in Minutes

Whether you're prototyping locally or deploying at scale, OGX makes it easy:

### **Local Development**

```bash
# Set up your environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -U ogx
uv run ogx list-deps starter | xargs -L1 uv pip install

# Start Ollama and pull a model
ollama serve
ollama run gpt-oss:20b

# Launch OGX with the starter distribution

OLLAMA_URL=http://localhost:11434/v1 uv run ogx stack run starter
```

```python
# Use with the OpenAI client
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="none")

response = client.responses.create(
    model="ollama/gpt-oss:20b", input="Write a haiku about open source."
)
print(response.output_text)
```

### **Production Deployment**

```bash
# Deploy with your preferred infrastructure
# Docker, Kubernetes, or bare metal — your choice
docker run -p 8321:8321 ogx/distribution-starter:latest
```

## Framework Ecosystem Compatibility

One of OGX's biggest advantages is **drop-in compatibility** with existing tooling:

### **Direct OpenAI Client**

```python
from openai import OpenAI

# Same code, different backend
client = OpenAI(base_url="http://your-ogx/v1", api_key="none")
```

### **LangChain Integration**

```python
from langchain_openai import ChatOpenAI

# Point to your OGX server
llm = ChatOpenAI(
    base_url="http://your-ogx/v1/openai/v1",
    api_key="none",
    model="ollama/gpt-oss:20b",
)
```

### **Native OGX Client**

```python
from ogx_client import OgxClient

# Access the full OGX API surface
client = OgxClient(base_url="http://your-ogx")
```

## Built for Open Standards

Our 100% Open Responses compliance reflects a broader philosophy: **open standards enable innovation**. When you build on OGX, you're not just adopting our implementation—you're investing in an ecosystem where:

- **Applications are portable**: Move between providers without rewriting code
- **Standards evolve collaboratively**: Community-driven development rather than vendor dictates
- **Innovation is shared**: Improvements benefit the entire ecosystem

## Technical Excellence Through Testing

Achieving 100% Open Responses compliance required rigorous engineering:

- **Perfect conformance testing**: Every PR runs the full Open Responses test suite with 6/6 passing tests
- **Automated compliance validation**: Blocking requirements ensure compliance is maintained, not achieved once
- **Production testing**: Integration tests with real workloads and multi-provider scenarios
- **Comprehensive API coverage**: Full implementation of the Open Responses specification

## What's Next

OGX's OpenAI compatibility is just the beginning. We're actively working on:

- **Enhanced streaming support**: Improved real-time response handling
- **Extended MCP ecosystem**: Deeper tool integration and connector development
- **Performance optimizations**: Faster inference and better resource utilization
- **Broader OpenAI API coverage**: Expanding compatibility beyond our current feature set

## Join the Open AI Infrastructure Movement

OGX represents something new in the AI infrastructure landscape: **enterprise-grade capabilities without vendor lock-in**. Whether you're a startup building your first AI application or an enterprise looking to bring AI workloads in-house, OGX provides the reliability, security, and compatibility you need.

Ready to get started?

- **📚 [Documentation](https://ogx-ai.github.io/docs/)**: Comprehensive guides and API references
- **🚀 [Getting Started](https://ogx-ai.github.io/docs/getting_started/)**: Quick setup tutorials
- **🔧 [OpenAI Implementation Guide](https://ogx-ai.github.io/docs/providers/openai)**: Detailed compatibility examples
- **🔌 [MCP Integration](https://ogx-ai.github.io/docs/building_applications/)**: Tool ecosystem and connector guides
- **💬 [Community](https://github.com/ogx-ai/ogx)**: Join discussions and contribute

The future of AI infrastructure is open, interoperable, and under your control. Welcome to OGX.

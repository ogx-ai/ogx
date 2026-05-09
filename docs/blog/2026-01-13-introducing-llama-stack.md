---
slug: introducing-ogx
title: Introducing OGX - The Open-Source Platform for Building AI Applications
authors: [ogx-team]
tags: [announcement, introduction, getting-started]
date: 2026-01-22
---

Welcome to our blog!

We're excited to introduce you to **OGX** - the open-source platform that simplifies building production-ready generative AI applications.

<!--truncate-->

## What is OGX?

OGX defines and standardizes the core building blocks needed to bring generative AI applications to market, centered on the [Open Responses specification](https://www.openresponses.org/). By aligning with OpenAI’s open-sourced Responses API, OGX provides a consistent, interoperable foundation for building agentic and generative systems. It offers a growing suite of open-source APIs—including prompts, conversations, files, models, embeddings, fine-tuning, and MCP—enabling seamless transitions from local development to production across providers and environments.

Think of OGX as a universal interface that abstracts away the complexity of working with different AI tools and provider (e.g., vector databases, model inference providers, and deployment environments). Whether you're building locally, deploying on-premises, or scaling in the cloud, OGX provides a consistent developer experience.

## Key Features

### Unified API Layer

OGX provides standardized APIs across six core capabilities:

- **Inference**: Run models locally or in the cloud with a consistent interface
- **Vector Stores**: Build knowledge and agentic retrieval systems
- **Agents**: Create intelligent agent flows with responses/conversations
- **Tools and MCP**: Integrate with external tools and services directly or via MCP
- **Moderations**: Built-in safety guardrails and content filtering via moderations api

### Plugin Architecture

The plugin architecture supports a rich ecosystem of API implementations across different environments:

- **Local Development**: Start with CPU-only setups for rapid iteration
- **On-Premises**: Deploy in your own infrastructure
- **Cloud**: Scale with hosted providers

### Prepackaged Distributions

Distributions are pre-configured bundles of provider implementations that make it easy to get started. You can begin with a local setup using Ollama and seamlessly transition to production with vLLM - all without changing your application code.

### Multiple Developer Interfaces

OGX supports various developer interfaces:

- **CLI**: Command-line tools for server management
- **Python SDK**: [`ogx-client-python`](https://github.com/ogx-ai/ogx-client-python)
- **TypeScript SDK**: [`ogx-client-typescript`](https://github.com/ogx-ai/ogx-client-typescript)

## Why OGX?

### Flexibility Without Compromise

Developers can choose their preferred infrastructure without changing APIs. This means you can:

- Start locally for development
- Test with different providers
- Deploy to production with your chosen infrastructure
- Switch providers as your needs evolve

All while maintaining the same codebase and APIs.

### Consistent Experience

With unified APIs, OGX makes it easier to:

- Build applications with consistent behavior
- Test across different environments
- Deploy with confidence
- Maintain and update your codebase

### Robust Ecosystem

OGX integrates with distribution partners including:

- **Cloud Providers**: AWS Bedrock, Together, Fireworks, and more
- **Hardware Vendors**: NVIDIA, Cerebras, SambaNova
- **Vector Databases**: ChromaDB, Milvus, Qdrant, Weaviate, PostgreSQL, ElasticSearch
- **AI Companies**: OpenAI, Anthropic, Google Gemini

For a complete list, check out our [Providers Documentation](/docs/providers).

## How It Works

OGX consists of two main components:

1. **Server**: A server with pluggable API providers that can run in various environments
2. **Client SDKs**: Libraries for your applications to interact with the server

The server handles all the complexity of managing different providers, while the client SDKs provide a simple, consistent interface for your application code.

Refer to the [Quick Start Guide](https://ogx-ai.github.io/docs/getting_started/quickstart) to get started building your first AI application with OGX.

## What's Next?

See the [OGX Office Hours Content Calendar](https://docs.google.com/document/d/1it-OsGFgAIwAUctQRQ-j1CBxFHhvSm530YR67eYGW1I/edit?tab=t.4uf22mux1a94) for upcoming topics and the blog roadmap.

## Join the Community

We'd love to have you join our growing community:

- [Star us on GitHub](https://github.com/ogx-ai/ogx)
- [Join our Discord](https://discord.gg/ZAFjsrcw)
- [Read the Documentation](/docs)
- [Report Issues](https://github.com/ogx-ai/ogx/issues)

## Conclusion

OGX is designed to make building AI applications simpler, more flexible, and more maintainable. By providing unified APIs and a rich ecosystem of providers, we're enabling developers to focus on what matters most - building great applications.

Whether you're just getting started with AI or building production systems at scale, OGX has something to offer. We're excited to see what you'll build!

---
slug: consistent-agentic-api-layer
title: "Every Protocol. Every Framework. Zero Code Changes."
authors: [leseb]
tags: [agentic, multi-sdk, sovereignty, architecture]
date: 2026-05-11
---

Agents shouldn't change a line of code to run on your infrastructure.

That sentence sounds simple, but it represents a fundamental shift in how enterprises can adopt AI agents. Today, every agentic framework speaks a different protocol. Teams using Claude Agents talk Anthropic Messages. Teams using ADK talk Google Interactions. Most agents still call OpenAI Chat Completions or the newer Responses API. Each choice creates a hard dependency on a vendor's infrastructure, SDK, and API contract.

OGX exists to break that coupling. It's a server that speaks every major agentic protocol natively, translating them to any model running on any infrastructure. No vendor lock-in. No SDK rewrites.

<!--truncate-->

## The problem: protocol fragmentation

Enterprise AI adoption doesn't happen in a vacuum. Different teams pick different frameworks based on their use case, their experience, and their model preference. One team builds with Claude Code. Another ships with OpenAI's Responses API. A third experiments with Google's ADK. Each choice is reasonable in isolation.

But at the platform level, this creates a mess. Every protocol means a different API contract. Every API contract means a different deployment path, a different set of credentials, a different operational surface. The platform team ends up managing $N$ distinct integration points instead of one.

Worse, the choice of SDK becomes coupled to the choice of model and infrastructure. Want to use the Anthropic SDK with an open-weight model running on-prem? Want to use the OpenAI SDK against a model served by vLLM? Want to switch providers without rewriting your application? In the current landscape, each of those requires work.

This is the fragmentation tax. It slows adoption, increases operational burden, and creates vendor lock-in at the API layer — even when the models themselves are open.

## OGX as the translation layer

OGX solves this by implementing three API surfaces natively on a single server:

**OpenAI API** — Chat Completions, Responses, Embeddings, and the full suite of supporting endpoints. This is the baseline most agents and frameworks already use.

**Anthropic Messages API** — the multi-turn conversation protocol that Claude Agents and Claude Code speak natively.

**Google Interactions API** — the emerging agentic protocol from the ADK and Gemini ecosystem.

All three share the same underlying inference providers. A single OGX deployment can serve clients using different SDKs simultaneously, against the same models:

```python
from openai import OpenAI

openai_client = OpenAI(base_url="http://ogx-server:8321/v1", api_key="key")

from anthropic import Anthropic

anthropic_client = Anthropic(base_url="http://ogx-server:8321/v1", api_key="key")

from google import genai
from google.genai import types

google_client = genai.Client(
    api_key="key",
    http_options=types.HttpOptions(
        base_url="http://ogx-server:8321",
        api_version="v1alpha",
    ),
)
```

Three SDKs. One server. Same model. No translation code in your application.

## More than a proxy

OGX isn't just routing requests to different backends. That would be a reverse proxy, and those already exist. What makes OGX different is that it runs a **server-side agentic loop** — inference, tool calling, RAG, MCP integration, conversation state management — on supported API surfaces, with feature parity continuing to expand across protocols.

When a client sends a request through the Responses API with `file_search` and MCP tools attached, the server handles the entire orchestration: searching vector stores, calling MCP servers, chaining tool results back into the model, and synthesizing a final answer. Your client gets a response. It doesn't manage a loop.

This is the difference between a proxy and an application server. A proxy translates formats. OGX translates formats *and* runs the agentic logic that makes those formats useful.

```python
from openai import OpenAI

client = OpenAI(base_url="http://ogx-server:8321/v1", api_key="key")

response = client.responses.create(
    model="llama-3.3-70b",
    input="Summarize the Q1 compliance updates and check if any affect our deployment timeline.",
    tools=[
        {"type": "file_search", "vector_store_ids": ["vs_compliance"]},
        {
            "type": "mcp",
            "server_label": "project-tracker",
            "server_url": "http://internal-api:8000/sse",
        },
    ],
)
```

One API call. Multiple tools. Multi-step reasoning. All server-side.

## What sovereignty actually means

The conversation around AI sovereignty usually focuses on model weights. Can we run the model on our own hardware? Can we fine-tune it? Do we control the training data? These are important questions, and open-weight models answer them.

But sovereignty at the model layer isn't enough. If your application is written against a proprietary API, you're still locked in — just at a different layer. Switching from one provider to another means rewriting your agent code, re-testing your tool integrations, and re-validating your orchestration logic.

OGX extends sovereignty to the API contract. Your application talks to OGX using whatever SDK your team prefers. OGX talks to whatever inference provider you've deployed. The two decisions are independent:

- Use the Anthropic SDK with Ollama running on-prem
- Use the OpenAI SDK with vLLM on your Kubernetes cluster
- Use the Google SDK with Bedrock in AWS
- Switch any of these without touching application code

And once the platform owns the API contract, it can enforce policy at the protocol layer. Attribute-based access control determines which models, tools, or APIs a given team or user can reach — regardless of which SDK they're calling through. A data science team gets access to large reasoning models and file search. An internal chatbot gets a smaller model and no tool access. The controls live in the platform, not scattered across framework configs.

This is something you can't do when each framework talks directly to its own provider. Centralized policy enforcement is a natural consequence of centralized API translation.

The customer owns the API contract, not just the model weights. That's what sovereignty means at the platform level.

## Framework-agnostic by design

The protocol diversity in the agentic ecosystem isn't going away. Claude Code, Codex, ADK, Strands, LangGraph, OpenClaw — each framework has its own strengths and its own protocol preferences. Telling teams to standardize on one framework is unrealistic and counterproductive.

OGX takes a different approach: let teams choose whatever framework fits their use case, and make the platform handle the protocol differences. A team building with Claude Code doesn't need to know or care that the platform is running open-weight models on OpenShift. A team using ADK doesn't need to rewrite their agents if the organization decides to switch inference providers.

This is what "zero code changes" means in practice. Not zero effort to set up the platform — that takes real engineering. But zero changes to the agent code that teams have already built and tested.

## Where this is heading

OGX already supports the core API surfaces. The work ahead is deepening that support:

- **Richer tool orchestration** across all three protocols — parallel tool calls, code interpretation, and response branching
- **Full agentic feature parity** — making sure every protocol surface gets the same server-side capabilities, not just basic inference
- **Production hardening** — storage persistence, multi-provider auth, telemetry, and upgrade paths that enterprise deployments require

The goal is straightforward: any agent, any framework, any model, any infrastructure. One server.

If your team is building agents and doesn't want to bet on a single vendor's API contract, [get started with OGX](https://ogx-ai.github.io/docs) or join the conversation on [Discord](https://discord.gg/ZAFjsrcw).

---
slug: from-llama-stack-to-ogx
title: "From Llama Stack to OGX: A New Name, A Sharper Mission"
authors: [ogx-team]
tags: [announcement, ogx, rename, agentic, multi-sdk]
date: 2026-04-28
---

Llama Stack is now OGX. The name changed, but more importantly, so did the mission.

When this project started, it was an API standardization effort — a set of specs for building AI applications, anchored to the Llama model family. That framing attracted contributors and integrations, but it also created confusion about what the project actually *is*. People thought it was a spec. Or a Llama-only thing. Or another framework.

It's none of those. OGX is a server. Specifically, it's a **server-side agentic loop** that speaks the native API of every major frontier lab — OpenAI, Anthropic, and Google — so your application code doesn't have to care which one you're using.

This post explains why we renamed, what changed in the project's direction, and what that means for you.

<!--truncate-->

## Why the rename

Three problems with "Llama Stack":

**The Llama association was limiting.** The project supports 23 inference providers. You can run GPT-4, Claude, Gemini, Mistral, or any model you want behind OGX. But "Llama Stack" made people think it only worked with Llama models. That was never true, and the gap between name and reality kept growing.

**"Stack" suggested a framework.** Developers hear "stack" and think of a collection of libraries you import into your code — something like LangChain or LlamaIndex. OGX is fundamentally different: it's an HTTP server with a pluggable provider architecture. Your application talks to it over the network, in any language, with any client. That architectural distinction matters and the old name obscured it.

**The project outgrew its origin.** What started as an API around Meta's models became a multi-provider, multi-SDK server implementing the OpenAI Responses API, the Anthropic Messages API, and the Google Interactions API. The name needed to reflect where the project is, not where it started.

OGX is short, neutral, and doesn't tie the project to a single model family or company. It's a name that can grow with the project.

## What actually changed

The rename touched 1,696 files across the codebase. Source directories moved from `llama_stack` to `ogx`. The CLI changed from `llama` to `ogx`. Environment variables changed from `LLAMA_STACK_*` to `OGX_*`. The GitHub org moved to `ogx-ai`.

But the rename was also a moment to clarify what OGX is *for*. The project's identity shifted from "an AI API standard" to something more specific and more useful:

**OGX is a server-side agentic loop that speaks any frontier lab API.**

Let's break that down.

## Server-side agentic loop

Before the Responses API existed, building an agent meant writing a client-side orchestration loop. Your code called the model, checked if it wanted to use a tool, executed the tool, sent results back, and repeated. Every application reimplemented this loop, and every implementation had its own bugs.

The Responses API moved that loop to the server. You send a question and a set of tools. The server handles the planning, tool execution, and synthesis internally. Your client gets a final answer.

OGX implements this loop with full support for:

- **Built-in RAG** via `file_search` — the server searches your vector store, retrieves relevant passages, and grounds the response, all within the agentic loop
- **MCP integration** — connect any MCP server and the agent discovers and uses its tools automatically
- **Multi-step reasoning** — the server handles chains of tool calls, not just single-shot inference
- **Conversation state** — persistent context across interactions without client-side state management

This is the core value proposition. OGX doesn't just route inference requests to different backends. It runs the agentic loop — the reasoning, tool calling, and synthesis — server-side, so your application stays simple.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

# Single API call. Server handles multi-step tool orchestration.
response = client.responses.create(
    model="llama-3.3-70b",
    input="What documents mention our pricing strategy?",
    tools=[
        {"type": "file_search", "vector_store_ids": ["vs_123"]},
        {
            "type": "mcp",
            "server_label": "internal-api",
            "server_url": "http://api-server:8000/sse",
        },
    ],
)
```

## Speaks any frontier lab API

OGX isn't an inference server — it routes to inference backends like vLLM, Ollama, or Bedrock. What makes it different is that it doesn't just expose the OpenAI API in front of those backends. It implements three API surfaces natively, so teams using different SDKs can all point at the same server.

**OpenAI API** — `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, and the full suite of supporting endpoints (files, vector stores, batches, models).

**Anthropic Messages API** — `/v1/messages`. Use the Anthropic Python or TypeScript SDK directly. OGX handles the format translation server-side.

**Google Interactions API** — `/v1alpha/interactions`. Use the Google GenAI SDK directly.

All three hit the same underlying inference providers. The same OGX server can serve clients using different SDKs simultaneously:

```python
# All three work against the same OGX server, same model
from openai import OpenAI

openai_client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")

from anthropic import Anthropic

anthropic_client = Anthropic(base_url="http://localhost:8321/v1", api_key="fake")

from google import genai

google_client = genai.Client(api_version="v1alpha", api_key="fake")
```

This decouples two decisions that used to be coupled: **which SDK your team prefers** and **which model or provider you deploy**. Use the Anthropic SDK with Ollama. Use the Google SDK with vLLM. Use the OpenAI SDK with Bedrock. The server translates; your code doesn't change.

For providers that natively support a given API format, OGX passes requests through directly — no translation overhead. For everything else, it converts between formats transparently.

## What OGX is not

It's worth being explicit about what the project is *not*, because the old framing created some misconceptions:

**Not primarily an API standard.** OGX implements existing standards where they exist (OpenAI API, Open Responses, Anthropic Messages API, Google Interactions API). Where no frontier lab provides an equivalent — like file processing for document ingestion — OGX defines its own APIs to fill the gap. If a frontier lab later ships a standard for one of those gaps, we'll adopt it rather than maintain our own. The project's value is in the server, not in spec authorship.

**Server-first, not a framework.** The primary deployment model is an HTTP server that your application talks to over the network, in any language. A library mode exists for embedding OGX in-process, but the architecture is designed around the server: pluggable providers, API translation, and agentic orchestration all happen on the server side.

**Not Llama-only.** It never was, but now the name makes that clear. OGX works with any model accessible through its 23 inference providers.

**Not just inference routing.** Proxy servers that forward requests to different backends are useful but limited. OGX runs the full agentic loop server-side: inference, tool calling, RAG, MCP integration, conversation management, safety, and file processing. That's the difference between a proxy and an application server.

## Migration from Llama Stack

If you're upgrading from `llama-stack`:

- **Package name**: `llama-stack` → `ogx`
- **CLI**: `llama` → `ogx`
- **Environment variables**: `LLAMA_STACK_*` → `OGX_*`
- **HTTP headers**: `x-llamastack-*` → `x-ogx-*`
- **Client SDK**: The Python client package (`llama_stack_client`) has not changed yet — that migration will happen separately

The server API itself is unchanged. If your application talks to OGX over HTTP (as it should), your client code doesn't need to change at all. Just point it at the new server.

## What's next

The rename clears the way for the project to grow in the direction it's already heading:

- **Deeper multi-SDK support** — expanding Anthropic and Google API coverage beyond basic inference to include tool calling and agentic features
- **Extended agentic capabilities** — richer server-side orchestration patterns, better conversation management, and more built-in tools
- **Performance** — faster inference routing, more efficient agentic loops, and better resource utilization across providers

The name is new. The mission is sharper. The server is the same one you've been using — just with a name that finally matches what it does.

Get started at [ogx-ai.github.io/docs](https://ogx-ai.github.io/docs), or join the conversation on [Discord](https://discord.gg/ZAFjsrcw).

— Charlie, Francisco, Matt, Raghu, Seb

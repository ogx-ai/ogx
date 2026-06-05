---
title: 'OGX: An Open-Source, Vendor-Neutral Generative AI Application Server'
tags:
  - Python
  - artificial intelligence
  - large language models
  - agentic AI
  - retrieval-augmented generation
  - OpenAI API
  - server-side orchestration
  - Kubernetes
  - multitenancy
authors:
  - name: Francisco Javier Arceo
    orcid: 0009-0009-7432-2006
    affiliation: 1
    corresponding: true
  - name: Sébastien Han
    affiliation: 1
  - name: Matthew Farrellee
  - name: Charlie Doern
    affiliation: 1
  - name: Yuan Tang
    affiliation: 1
  - name: Derek Higgins
    affiliation: 1
  - name: Varsha Prasad Narsing
    orcid: 0009-0006-4421-3632
    affiliation: 1
  - name: Raghotham Murthy
    affiliation: 2
  - name: Gordon Sim
    affiliation: 1
  - name: Sumanth Kamenani
    affiliation: 1
  - name: Ben Browning
    affiliation: 1
affiliations:
  - name: Red Hat AI, USA
    index: 1
  - name: Meta, USA
    index: 2
date: 2 June 2026
bibliography: paper.bib
---

# Summary

OGX (Open GenAI Stack), formerly Llama Stack [@llamastack], is an open-source AI application server and Python library that implements the APIs of major frontier labs (OpenAI, Anthropic, Google) with pluggable backend providers [@ogx]. It allows teams to build AI systems---including retrieval-augmented generation (RAG) pipelines, multi-turn conversational agents, and tool-calling workflows---against a single, stable API surface and then deploy them with any combination of inference engine, vector database, and safety backend, without changing application code.

OGX's primary API focus is the Responses API for server-side agentic orchestration, conforming to the Open Responses specification [@openresponses]. The server also exposes Chat Completions, Embeddings, Vector Stores, Files, and Batches endpoints. Beyond OpenAI compatibility, OGX natively supports the Anthropic Messages API (`/v1/messages`) and Google GenAI Interactions API (`/v1alpha/interactions`), allowing teams using any of the three major client SDKs to connect to the same server. OGX supports over 20 inference providers (including vLLM, Ollama, OpenAI, Anthropic, Bedrock, and Gemini), 13 vector store backends, and 7 safety providers. It can run as an HTTP server for production deployments or be imported directly as a Python library for scripting and notebooks. A companion Kubernetes Operator [@ogxk8soperator] automates deployment lifecycle management through custom resources, supporting multi-architecture builds, hot-swappable distribution images, and both shared and per-tenant isolation topologies. Together, OGX and its operator serve as the self-hosted, model-agnostic backend for AI-powered developer tools such as Claude Code, Codex CLI, OpenCode, and OpenHands.

# Statement of Need

AI application development today is tightly coupled to proprietary API providers. An application written against OpenAI's API often cannot switch to an open-weight model served by vLLM or Ollama without rewriting its inference, retrieval, and tool-calling logic. This coupling limits reproducibility, makes comparisons across model providers difficult, and prevents teams from running AI workloads on controlled infrastructure---a requirement in regulated, privacy-sensitive, and air-gapped environments.

Existing open-source tools address parts of this problem but not the whole. Inference engines like vLLM [@kwon2023vllm] and SGLang [@sglang] serve models efficiently but do not provide retrieval, tool calling, or conversation management. Gateway proxies like LiteLLM route requests across providers but do not execute the agentic loop or manage vector stores. Client-side frameworks like LangChain [@langchain] and LangGraph [@langgraph] provide developer abstractions but push orchestration, state management, and security enforcement to the application layer.

OGX fills this gap by providing a complete, self-hosted AI application server that implements the OpenAI API surface---with the Responses API as its primary focus---alongside Anthropic Messages and Google GenAI Interactions compatibility layers. Developers write code against standard endpoints (`/v1/responses`, `/v1/chat/completions`, `/v1/vector_stores`) and swap the underlying infrastructure through configuration, not code changes. This decouples three decisions that are currently entangled: which SDK to use, which model to run, and where to deploy.

# State of the Field

The AI application ecosystem has stratified into layers that each solve a subset of the deployment problem. OGX continues the Llama Stack project under a renamed, model-agnostic mission; the relevant comparison is therefore the server-side API layer that OGX provides versus adjacent inference engines, gateways, and client-side frameworks.

**Inference engines** (vLLM [@kwon2023vllm], SGLang [@sglang], Ollama) focus on efficient model serving. They optimize throughput and latency but do not provide retrieval, conversation state, tool execution, or safety guardrails. An application using vLLM for inference must separately integrate a vector database, implement its own agentic loop, and manage multi-turn state.

**API gateways** (LiteLLM, OpenRouter) provide a unified interface across multiple inference providers but act as pass-through proxies. They do not manage vector stores, execute tool calls, or maintain conversation history---they translate request formats between SDKs and providers.

**Client-side frameworks** (LangChain [@langchain], LangGraph [@langgraph], LlamaIndex [@llamaindex], CrewAI [@crewai], Haystack [@haystack]) provide rich developer abstractions for building agents and RAG pipelines. However, they execute orchestration client-side, distributing security-critical logic across application code. These frameworks are complementary to OGX: they compose agent logic while OGX provides the server-side execution target they call into.

**Proprietary platforms** (OpenAI's Responses API [@openaiResponsesAPI], Databricks Mosaic AI [@databricksAgentFramework]) offer integrated experiences but couple applications to a specific vendor's infrastructure and pricing.

OGX occupies a distinct position: a self-hosted, vendor-neutral server that implements the full API surface---inference, retrieval, tool execution, conversation management, and safety---with pluggable providers at every layer. Its conformance to the Open Responses specification [@openresponses] ensures interoperability with any client that speaks the same protocol. It does not compete with client-side frameworks; it is the infrastructure they deploy against when reproducibility, provider portability, and centralized policy enforcement matter.

# Software Design

## Provider Architecture

OGX's core abstraction is the pluggable provider. Each API capability (inference, vector storage, safety, tool runtime, file processing) is defined by a Protocol interface in the lightweight `ogx-api` package, allowing third-party providers to implement the contract without depending on the full server. Concrete providers implement these interfaces for specific backends: `remote::openai` and `remote::anthropic` for hosted APIs, `remote::vllm` for self-hosted GPU inference, `inline::faiss` and `remote::pgvector` for vector search, and so on. A routing layer dispatches requests to provider instances based on logical resource identifiers, enabling multiple providers to serve the same API simultaneously---for example, Ollama handling local models while OpenAI handles hosted models, both behind `/v1/chat/completions`.

A *distribution* packages a specific set of providers and configuration into a deployable unit, decoupling application logic from infrastructure selection. This design favors reproducible configuration over ad hoc application code: teams can prototype with lightweight inline providers (Ollama, sqlite-vec [@sqlitevec]) and deploy to production backends (vLLM, pgvector) by changing the distribution, not the application.

## Dual Deployment Model

OGX runs in two modes. **Server mode** exposes HTTP endpoints accessible from any language or tool. **Library mode** allows direct Python import with zero network overhead, suitable for notebooks and scripts. Both modes use identical provider routing and API semantics.

## Multi-SDK Compatibility

OGX serves three client SDK protocols from a single server. The **OpenAI-compatible endpoints** (`/v1/chat/completions`, `/v1/responses`, `/v1/vector_stores`) are the primary interface and the Responses API implementation conforms to the Open Responses specification [@openresponses]. The **Anthropic Messages endpoint** (`/v1/messages`) and **Google GenAI Interactions endpoint** (`/v1alpha/interactions`) provide native compatibility for teams using those SDKs. This portability comes with a trade-off: OGX must normalize provider-specific behavior into stable API contracts while still exposing enough backend-specific configuration for real deployments.

## Server-Side Agentic Orchestration

The Responses API implements server-side agentic orchestration: the inference-tool-inference loop executes within the server process, not the client [@openaiResponsesAPI]. This centralizes security enforcement, tool authorization, and conversation state management, at the cost of moving some flexibility from application code into server configuration. Built-in tools include file search (RAG over vector stores), web search, code interpretation, and Model Context Protocol (MCP) [@mcp] integration for external tool servers.

OGX is extending server-side execution with a Containers API and a Skills API. The Containers API (`/v1/containers`) manages sandboxed execution environments---matching the OpenAI Containers API---enabling models to execute shell commands in isolated Docker, Podman, or Kubernetes containers via a `shell` tool in the Responses API. The Skills API (`/v1alpha/skills`) manages versioned skill bundles that package tools, prompts, and configuration into reusable, composable units. Together, these APIs move code execution and task composition onto the server, extending the same trust-boundary principle that governs retrieval and tool authorization.

Unlike inference engines and API gateways that treat requests as stateless, OGX manages state natively: the Conversations API persists multi-turn history with tenant-scoped isolation, the Prompts API provides versioned prompt templates, and a resource registry tracks models, vector stores, and files as first-class server objects. A Compaction API summarizes long histories to manage context window limits. This state layer is what makes OGX a complete application server rather than a stateless proxy.

For multitenant deployments, OGX provides attribute-based access control (ABAC) that enforces tenant isolation at the retrieval, tool execution, state management, and API routing layers. The security properties of this architecture have been formally analyzed and empirically validated [@arceo2026securing].

## Kubernetes Operator

The OGX Kubernetes Operator [@ogxk8soperator] provides declarative deployment through the `OGXServer` custom resource. A single CR specifies the distribution, replica count, storage, inference backend, and network policies. The operator manages full lifecycle reconciliation with support for both vanilla Kubernetes and OpenShift, ConfigMap-driven image overrides for fleet-wide updates, multi-architecture builds (amd64/arm64) with FIPS-compliant images, and shared instances with ABAC isolation, per-tenant namespace isolation, and hybrid topologies.

# Example Usage

The following example demonstrates building a RAG agent using the standard OpenAI SDK against an OGX server. The same code works regardless of which inference provider or vector store backend is configured:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="unused")

# Create a vector store and upload documents
vector_store = client.vector_stores.create(name="docs")
with open("manual.pdf", "rb") as file:
    client.vector_stores.files.upload(vector_store_id=vector_store.id, file=file)

# Query with server-side RAG via the Responses API
response = client.responses.create(
    model="llama-3.3-70b-instruct",
    input="What are the installation requirements?",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
)
print(response.output_text)
```

Switching from a local Ollama backend to a production vLLM cluster requires changing only the server's distribution configuration---the client code above remains identical. Published tutorials demonstrate this portability across diverse enterprise backends, including IBM watsonx.ai with Milvus vector storage [@ibm_rag_milvus] and Oracle Cloud Infrastructure with OCI AI Blueprints [@oracle_oci_ogx].

# Research Impact Statement

OGX has realized impact through both public deployments and customer production use. Publicly documented examples under the former Llama Stack name include an intelligent OpenShift operations agent combining RAG, web search, and MCP tool integration for automated incident response [@redhat_ops_agent], enterprise RAG pipelines on IBM watsonx.data with Milvus [@ibm_rag_milvus], and generative AI application development on Oracle Cloud Infrastructure [@oracle_oci_ogx]. Maintainers report production deployments with customers in telecommunications, semiconductor manufacturing, financial services, insurance, and consulting. The framework has been presented at Meta Connect [@meta_connect_ogx] and IBM TechXchange [@ibm_techxchange_ogx] as a standardization layer for enterprise AI applications.

The security architecture---specifically, the multitenant isolation model combining ABAC-gated retrieval, server-side orchestration, and pluggable provider backends---was formally analyzed in a peer-reviewed publication at the ACM Conference on AI and Agentic Systems [@arceo2026securing]. OGX conforms to the Open Responses specification [@openresponses] and serves as a reference implementation for open, vendor-neutral agentic AI APIs.

As of June 2026, the project has over 8,400 GitHub stars, 242 contributors, 4,000 commits, and 68 releases across nearly two years of public development. Community engagement includes weekly contributor calls, an active Discord server, and integrations contributed by external organizations including Red Hat, IBM, Oracle, and Infinispan.

# AI Usage Disclosure

Generative AI tools, including GitHub Copilot and Anthropic Claude models available during development, were used for code completion, documentation drafting, and paper drafting. Assistance was limited to generating candidate text or code that human contributors reviewed, edited, tested, and validated. Core architectural decisions, API design, the security model, and final paper content were made by human authors.

# Acknowledgements

We thank Meta for creating and open-sourcing Llama Stack, now renamed OGX. We are grateful to Red Hat for supporting the development of OGX and the Kubernetes Operator through employee time and infrastructure support. We thank the OGX contributor community for their sustained contributions to the project.

# References

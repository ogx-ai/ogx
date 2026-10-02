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
    affiliation: 3
  - name: Charlie Doern
    affiliation: 1
  - name: Yuan Tang
    affiliation: 1
  - name: Derek Higgins
    affiliation: 1
  - name: Varsha Prasad Narsing
    orcid: 0009-0006-4421-3632
    affiliation: 1
  - name: Gordon Sim
    affiliation: 1
  - name: Sumanth Kamenani
    affiliation: 1
  - name: Ben Browning
    affiliation: 1
  - name: Raghotham Murthy
    affiliation: 2
affiliations:
  - name: Red Hat AI, USA
    index: 1
  - name: Meta, USA
    index: 2
  - name: Independent
    index: 3
date: 2 June 2026
bibliography: paper.bib
---

# Summary

OGX (Open GenAI Stack), formerly Llama Stack [@llamastack], is an open-source AI application server and Python library with interchangeable backend providers [@ogx]. Teams building retrieval-augmented generation (RAG) pipelines, conversational agents, and tool-calling workflows can change their inference engine and vector database through server configuration without rewriting application code.

OGX implements OpenAI, Anthropic, and Google APIs, with the Responses API for server-side orchestration as its primary focus [@openresponses]. The cited v1.0.3 release registers 23 inference provider types and 13 vector I/O provider types spanning 10 distinct storage backends; inline and remote variants count separately as provider types. A companion Kubernetes Operator [@ogxk8soperator] manages deployments, including shared and per-tenant instances. Together, OGX and its operator provide a self-hosted backend for AI-powered developer tools such as Claude Code, Codex CLI, OpenCode, and OpenHands.

# Statement of Need

AI application development today is tightly coupled to proprietary API providers. While inference-only workloads can increasingly be swapped across providers---vLLM, for example, supports the Responses API for basic inference---applications that rely on the full stack (retrieval, tool calling, conversation state, safety guardrails) remain difficult to migrate without rewriting significant application logic. This coupling limits reproducibility, makes comparisons across model providers difficult, and prevents teams from running AI workloads on controlled infrastructure---a requirement in regulated, privacy-sensitive, and air-gapped environments.

OGX addresses this need through standard APIs and configurable infrastructure. Researchers can compare backends while keeping retrieval, tool authorization, and conversation handling in one server. This separates the choice of client SDK, model, and deployment environment, supporting reproducible configurations on controlled infrastructure.

# State of the Field

OGX continues Llama Stack under a renamed, model-agnostic mission. Its server-side API layer complements inference engines, gateways, and client-side frameworks, which address different parts of the deployment problem.

**Inference engines** (vLLM [@kwon2023vllm], SGLang [@sglang], Ollama) focus on efficient model serving. They optimize throughput and latency but do not provide retrieval, conversation state, tool execution, or safety guardrails. An application using vLLM for inference must separately integrate a vector database, implement its own agentic loop, and manage multi-turn state.

**API gateways** (LiteLLM, OpenRouter) provide a unified interface across multiple inference providers but act as pass-through proxies. They do not manage vector stores, execute tool calls, or maintain conversation history---they translate request formats between SDKs and providers.

**Client-side frameworks** (LangChain [@langchain], LangGraph [@langgraph], LlamaIndex [@llamaindex], CrewAI [@crewai], Haystack [@haystack]) provide rich developer abstractions for building agents and RAG pipelines. However, they execute orchestration client-side, distributing security-critical logic across application code. These frameworks are complementary to OGX: they compose agent logic while OGX provides the server-side execution target they call into.

**Proprietary platforms** (OpenAI's Responses API [@openaiResponsesAPI], Databricks Mosaic AI [@databricksAgentFramework]) offer integrated experiences but couple applications to a specific vendor's infrastructure and pricing.

OGX integrates inference, retrieval, tool execution, conversation management, and safety behind Open Responses-compatible APIs [@openresponses]. Combining these responsibilities in a server addresses a different trust boundary from extending a client-side framework: applications share centralized policy enforcement and provider configuration while retaining their own agent logic.

# Software Design

## Provider Architecture

OGX's core abstraction is the pluggable provider. Provider-backed capabilities (inference, vector storage, tool runtime, file processing) are defined by Protocol interfaces in the lightweight `ogx-api` package, allowing third-party providers to implement the contract without depending on the full server. Concrete providers implement these interfaces for specific backends: `remote::openai` and `remote::anthropic` for hosted APIs, `remote::vllm` for self-hosted GPU inference, `inline::faiss` and `remote::pgvector` for vector search, and so on. A routing layer dispatches requests to provider instances based on logical resource identifiers, enabling multiple providers to serve the same API simultaneously---for example, Ollama handling local models while OpenAI handles hosted models, both behind `/v1/chat/completions`.

A *distribution* packages a specific set of providers and configuration into a deployable unit, decoupling application logic from infrastructure selection. This design favors reproducible configuration over ad hoc application code: teams can prototype with lightweight local backends (Ollama, sqlite-vec [@sqlitevec]) and deploy to production backends (vLLM, pgvector) by changing the distribution, not the application.

Content moderation uses a separate integration path. Optional Responses guardrails send input and generated text to a server-configured, OpenAI-compatible `moderation_endpoint`. A request enables these checks with `guardrails=true`; there is no Safety provider protocol. Changing the moderation service requires a compatible endpoint and updated server configuration.

## Dual Deployment Model

OGX runs in two modes. **Server mode** exposes HTTP endpoints accessible from any language or tool. **Library mode** allows direct Python import with zero network overhead, suitable for notebooks and scripts. Both modes use identical provider routing and API semantics.

## Multi-SDK Compatibility

OGX serves three client SDK protocols from a single server. The stable **OpenAI-compatible endpoints** (`/v1/chat/completions`, `/v1/responses`, `/v1/vector_stores`) are the primary interface and the Responses API implementation conforms to the Open Responses specification [@openresponses]. The **Anthropic Messages endpoint** (`/v1/messages`) is also stable. The **Google GenAI Interactions endpoint** (`/v1alpha/interactions`) is experimental and may change incompatibly between releases. This portability comes with a trade-off: OGX must normalize provider-specific behavior into API contracts while still exposing enough backend-specific configuration for real deployments.

## Server-Side Agentic Orchestration

The Responses API implements server-side agentic orchestration: the inference-tool-inference loop executes within the server process, not the client [@openaiResponsesAPI]. This centralizes security enforcement, tool authorization, and conversation state management, at the cost of moving some flexibility from application code into server configuration. Built-in tools include file search (RAG over vector stores), web search, code interpretation, and Model Context Protocol (MCP) [@mcp] integration for external tool servers.

Unlike inference engines and API gateways that treat requests as stateless, OGX manages state natively: the Conversations API persists multi-turn history with tenant-scoped isolation, the Prompts API provides versioned prompt templates, and a resource registry tracks models, vector stores, and files as first-class server objects. A Compaction API summarizes long histories to manage context window limits. Telemetry is built on OpenTelemetry (OTEL) with MLflow [@mlflow] tracing integration for logging spans, tool calls, and retrieval steps to existing ML experiment tracking infrastructure. This state and observability layer is what makes OGX a complete application server rather than a stateless proxy.

For multitenant deployments, OGX provides attribute-based access control (ABAC) that enforces tenant isolation at the retrieval, tool execution, state management, and API routing layers. The security properties of this architecture have been formally analyzed and empirically validated [@arceo2026securing].

## Kubernetes Operator

The OGX Kubernetes Operator [@ogxk8soperator] provides declarative deployment through the `OGXServer` custom resource. A single CR specifies the distribution, replica count, storage, inference backend, and network policies. The operator manages full lifecycle reconciliation with support for both vanilla Kubernetes and OpenShift, ConfigMap-driven image overrides for fleet-wide updates, multi-architecture builds (amd64/arm64) with FIPS-compliant images, and shared instances with ABAC isolation, per-tenant namespace isolation, and hybrid topologies.

# Example Usage

This RAG example uses the standard OpenAI SDK against OGX, with inference and vector storage selected through server configuration:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="unused")

# Create a vector store and upload documents
vector_store = client.vector_stores.create(name="docs")
with open("manual.pdf", "rb") as file:
    client.vector_stores.files.upload(vector_store_id=vector_store.id, file=file)

# Query with server-side RAG via the Responses API
response = client.responses.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    input="What are the installation requirements?",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store.id]}],
)
print(response.output_text)
```

Switching from Ollama to vLLM changes the server configuration while preserving this client code. Published tutorials cover IBM watsonx.ai with Milvus [@ibm_rag_milvus] and Oracle Cloud Infrastructure with OCI AI Blueprints [@oracle_oci_ogx].

# Research Impact Statement

OGX has realized impact through both public deployments and customer production use. Publicly documented examples under the former Llama Stack name include an intelligent OpenShift operations agent combining RAG, web search, and MCP tool integration for automated incident response [@redhat_ops_agent], enterprise RAG pipelines on IBM watsonx.data with Milvus [@ibm_rag_milvus], and generative AI application development on Oracle Cloud Infrastructure [@oracle_oci_ogx]. Beyond these publicly documented cases, maintainers are also aware of additional production use by enterprise customers in sectors including telecommunications, semiconductor manufacturing, financial services, insurance, and consulting; these engagements are confidential and not independently documented, so we report them here as self-reported maintainer knowledge rather than citable evidence. The framework has been presented at Meta Connect [@meta_connect_ogx] and IBM TechXchange [@ibm_techxchange_ogx] as a standardization layer for enterprise AI applications.

The security architecture---specifically, the multitenant isolation model combining ABAC-gated retrieval, server-side orchestration, and pluggable provider backends---was formally analyzed in a peer-reviewed publication at the ACM Conference on AI and Agentic Systems [@arceo2026securing]. OGX conforms to the Open Responses specification [@openresponses] and serves as a reference implementation for open, vendor-neutral agentic AI APIs.

As of June 2026, the project has over 8,400 GitHub stars, 242 contributors, 4,000 commits, and 68 releases across nearly two years of public development. Community engagement includes weekly contributor calls, an active Discord server, and integrations contributed by external organizations including Red Hat, IBM, Oracle, and Infinispan.

# AI Usage Disclosure

Generative AI tools, including GitHub Copilot, Anthropic Claude, and OpenAI Codex (GPT-6), were used for code completion, documentation drafting, and paper drafting. Assistance was limited to generating candidate text or code that human contributors reviewed, edited, tested, and validated. Core architectural decisions, API design, the security model, and final paper content were made by human authors.

# Acknowledgements

We thank Meta for creating and open-sourcing Llama Stack, now renamed OGX. We are grateful to Red Hat for supporting the development of OGX through employee time and infrastructure support. We thank the OGX contributor community for their sustained contributions to the project.

# References

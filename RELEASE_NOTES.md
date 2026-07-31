# OGX Release Notes

This file provides a summary of each release. For detailed release notes with migration instructions, code examples, and full contributor credits, see the linked documents.

<!-- New releases go here, at the top. See docs/releases/ for detailed notes. -->

## [0.7.0](docs/releases/RELEASE_NOTES_0.7.md) - 2026-04-01

Release 0.7 is a major release focused on completing the transition to OpenAI API conformance, introducing comprehensive observability metrics, and significant API cleanup. This release removes the fine-tuning API, completes the FastAPI router migration, removes legacy providers (TGI, HuggingFace), renames core concepts for clarity, and adds structured logging via structlog.

### Highlights

- **Agents API renamed to Responses API** aligning with OpenAI naming ([#5195](https://github.com/ogx-ai/ogx/pull/5195))
- **Reasoning output support** in Responses API ([#5206](https://github.com/ogx-ai/ogx/pull/5206))
- **Comprehensive observability metrics** for API, inference, and vector IO ([#5201](https://github.com/ogx-ai/ogx/pull/5201), [#5320](https://github.com/ogx-ai/ogx/pull/5320), [#5096](https://github.com/ogx-ai/ogx/pull/5096))
- **Structured logging via structlog** with key-value output ([#5215](https://github.com/ogx-ai/ogx/pull/5215))
- **Inline neural rerank for RAG** without external services ([#4877](https://github.com/ogx-ai/ogx/pull/4877))
- **Inline Docling provider** for structure-aware PDF parsing ([#5049](https://github.com/ogx-ai/ogx/pull/5049))
- **Infinispan vector-io provider** for distributed vector storage ([#4839](https://github.com/ogx-ai/ogx/pull/4839))
- **Connector API promoted to v1beta** ([#5129](https://github.com/ogx-ai/ogx/pull/5129))
- **FastAPI router migration complete** with `@webmethod` removal ([#5248](https://github.com/ogx-ai/ogx/pull/5248))
- **Performance**: lazy-loading of torch, numpy, faiss, and braintrust to reduce startup memory ([#5116](https://github.com/ogx-ai/ogx/pull/5116), [#5118](https://github.com/ogx-ai/ogx/pull/5118), [#5078](https://github.com/ogx-ai/ogx/pull/5078))

### Breaking Changes

| Change | Type | PR |
|--------|------|-----|
| Fine-tuning API removed | Hard | [#5104](https://github.com/ogx-ai/ogx/pull/5104) |
| `meta-reference` providers renamed to `builtin` | Hard | [#5131](https://github.com/ogx-ai/ogx/pull/5131) |
| `knowledge_search` renamed to `file_search` | Hard | [#5186](https://github.com/ogx-ai/ogx/pull/5186) |
| Agents API renamed to Responses API | Hard | [#5195](https://github.com/ogx-ai/ogx/pull/5195) |
| `tool_groups` removed from public API | Hard | [#4997](https://github.com/ogx-ai/ogx/pull/4997) |
| TGI and HuggingFace providers removed | Hard | [#5333](https://github.com/ogx-ai/ogx/pull/5333) |
| `register`/`unregister` model endpoints removed | Hard | [#5341](https://github.com/ogx-ai/ogx/pull/5341) |
| `@webmethod` decorator removed | Hard | [#5248](https://github.com/ogx-ai/ogx/pull/5248) |
| `rag-runtime` provider renamed to `file-search` | Hard | [#5187](https://github.com/ogx-ai/ogx/pull/5187) |
| Duplicate `dataset_id` parameter removed | Hard | [#4849](https://github.com/ogx-ai/ogx/pull/4849) |
| `/files/{file_id}` GET response unified | Hard | [#5154](https://github.com/ogx-ai/ogx/pull/5154) |
| OpenAI API schema transforms | Hard | [#5166](https://github.com/ogx-ai/ogx/pull/5166) |
| `starter-gpu` distribution removed | Hard | [#5279](https://github.com/ogx-ai/ogx/pull/5279) |
| `sentence_transformers` `trust_remote_code` defaults to `False` | Behavior | [#4602](https://github.com/ogx-ai/ogx/pull/4602) |

See the [full release notes](docs/releases/RELEASE_NOTES_0.7.md) for migration instructions and detailed upgrade guide.

## [0.5.0](docs/releases/RELEASE_NOTES_0.5.md) - 2026-02-05

Release 0.5 brings significant improvements to API consistency, OpenAI conformance, provider capabilities, and a major architectural refactoring of all APIs to use FastAPI routers.

### Highlights

- **Connectors API** for managing MCP server connections ([#4263](https://github.com/ogx-ai/ogx/pull/4263))
- **Unified network configuration** with TLS/mTLS, proxy, and timeout support for all remote providers ([#4748](https://github.com/ogx-ai/ogx/pull/4748))
- **Endpoint authorization** via YAML-based access control ([#4448](https://github.com/ogx-ai/ogx/pull/4448))
- **Rerankers** for hybrid search in vector stores ([#4456](https://github.com/ogx-ai/ogx/pull/4456))
- **Response API enhancements** including `reasoning.effort`, `max_output_tokens`, and `parallel_tool_calls`
- **New providers**: Elasticsearch and OCI 26ai vector stores
- **PGVector improvements**: HNSW/IVFFlat indexes, configurable distance metrics
- **FastAPI router migration** across all APIs for better OpenAPI docs and validation
- **ARM64 container image** support ([#4474](https://github.com/ogx-ai/ogx/pull/4474))

### Breaking Changes

| Change | Type | PR |
|--------|------|-----|
| Post-Training API endpoints restructured (path params) | Hard | [#4606](https://github.com/ogx-ai/ogx/pull/4606) |
| Embeddings API rejects explicit `null` for optional fields | Hard | [#4644](https://github.com/ogx-ai/ogx/pull/4644) |
| Safety API provider interface changed to request objects | Hard | [#4643](https://github.com/ogx-ai/ogx/pull/4643) |
| Builtin GPU inference provider removed | Hard | [#4828](https://github.com/ogx-ai/ogx/pull/4828) |
| Scope-based endpoint authorization removed | Hard | [#4734](https://github.com/ogx-ai/ogx/pull/4734) |
| `image_name` renamed to `distro_name` | Deprecated | [#4396](https://github.com/ogx-ai/ogx/pull/4396) |
| Eval API calling convention uses request objects | Deprecated | [#4425](https://github.com/ogx-ai/ogx/pull/4425) |
| vLLM `tls_verify` moved to `network.tls.verify` | Deprecated | [#4748](https://github.com/ogx-ai/ogx/pull/4748) |

See the [full release notes](docs/releases/RELEASE_NOTES_0.5.md) for migration instructions and detailed upgrade guide.

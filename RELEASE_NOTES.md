# OGX Release Notes

This file provides a summary of each release. For detailed release notes with migration instructions, code examples, and full contributor credits, see the linked documents.

<!-- New releases go here, at the top. See docs/releases/ for detailed notes. -->

## [2.0.0](docs/releases/RELEASE_NOTES_2.0.md) - 2026-11-26

Release 2.0 is a major release focused on the SystemOne judgment model API, a routed web search API with the new SearchApi backend, the retirement of the `tool_runtime`/`tool_groups` APIs, namespaced function tools, and a new SGLang inference provider. Python support moves to 3.13–3.14 (3.12 dropped), and engine pins advance across the board (llama.cpp b12360, Ollama 0.40.16, vLLM 0.32.0, SGLang 0.5.23).

### Highlights

- **SystemOne judgment model API** with the TypeSafe provider, and judgment support on Ollama, llama.cpp, SGLang, and vLLM ([#6748](https://github.com/ogx-ai/ogx/pull/6748), [#6769](https://github.com/ogx-ai/ogx/pull/6769), [#6772](https://github.com/ogx-ai/ogx/pull/6772), [#6782](https://github.com/ogx-ai/ogx/pull/6782), [#6866](https://github.com/ogx-ai/ogx/pull/6866))
- **New SGLang inference provider** with Tev1 support and integration CI enabled ([#6776](https://github.com/ogx-ai/ogx/pull/6776), [#6822](https://github.com/ogx-ai/ogx/pull/6822))
- **Web search is now a routed API** — moved from the `tool_runtime` tools to the `web_search` endpoint, with a new SearchApi backend and a built-in placeholder ([#7014](https://github.com/ogx-ai/ogx/pull/7014), [#6927](https://github.com/ogx-ai/ogx/pull/6927), [#6878](https://github.com/ogx-ai/ogx/pull/6878), [#6957](https://github.com/ogx-ai/ogx/pull/6957))
- **`tool_runtime`/`tool_groups` APIs retired** — web and file search run inside the Responses API; MCP tools served directly by the Responses API ([#7039](https://github.com/ogx-ai/ogx/pull/7039), [#7033](https://github.com/ogx-ai/ogx/pull/7033))
- **Namespaced function tools** with identity-exact mapping through the chat-completions layer ([#6941](https://github.com/ogx-ai/ogx/pull/6941))
- **`ogx connect` CLI overhaul** — shared `--url`, `-k/--insecure`, API-key pass-through, `https://localhost` default ([#6928](https://github.com/ogx-ai/ogx/pull/6928), [#6929](https://github.com/ogx-ai/ogx/pull/6929), [#7076](https://github.com/ogx-ai/ogx/pull/7076))
- **Python 3.13–3.14** (3.12 dropped) ([#6973](https://github.com/ogx-ai/ogx/pull/6973))
- Engine pins advanced: **llama.cpp b12360 (v1.0.0)** ([#7010](https://github.com/ogx-ai/ogx/pull/7010)), **Ollama 0.40.16** ([#7011](https://github.com/ogx-ai/ogx/pull/7011)), **vLLM 0.32.0** ([#6840](https://github.com/ogx-ai/ogx/pull/6840)), **SGLang 0.5.23** ([#6986](https://github.com/ogx-ai/ogx/pull/6986))
- Responses robustness: proper 400s for oversized prompts and incomplete `text.format`, malformed tool-call arguments fed back to the model, no more silently dropped turns ([#6907](https://github.com/ogx-ai/ogx/pull/6907))

### Breaking Changes

| Change | Type | PR |
|--------|------|-----|
| Python 3.12 no longer supported (3.13–3.14 are) | Hard | [#6973](https://github.com/ogx-ai/ogx/pull/6973) |
| `tool_runtime` and `tool_groups` APIs removed (incl. `remote::model-context-protocol`) | Hard | [#7039](https://github.com/ogx-ai/ogx/pull/7039) |
| `remote::passthrough` inference provider removed | Hard | [#7034](https://github.com/ogx-ai/ogx/pull/7034) |
| `/v1alpha/file-processors/process` route removed; use the jobs API | Hard | [#7078](https://github.com/ogx-ai/ogx/pull/7078) |
| `knowledge_search` compat alias removed (deprecated since v0.7) | Hard | [#6803](https://github.com/ogx-ai/ogx/pull/6803) |
| `remote::wolfram-alpha` tool provider removed | Hard | [#6804](https://github.com/ogx-ai/ogx/pull/6804) |
| String forms of `Models.register_model`/`unregister_model` removed | Hard | [#7082](https://github.com/ogx-ai/ogx/pull/7082) |
| Stack/distribution configs using the `tool_runtime` section no longer load; use `web_search` | Hard | [#7079](https://github.com/ogx-ai/ogx/pull/7079) |
| Deprecated CLI shims removed: `ogx letsgo`, `ogx stack run`, `ogx stack go`, `ogx connect codex --exec` | Hard | [#7076](https://github.com/ogx-ai/ogx/pull/7076) |
| Deprecated config keys removed; `~/.llama` → `~/.ogx` auto-migration gone | Hard | [#7077](https://github.com/ogx-ai/ogx/pull/7077) |
| `ogx connect`: per-agent `--url`/`--host/--port` flags removed; single shared `--url` | Hard | [#6929](https://github.com/ogx-ai/ogx/pull/6929) |
| Anthropic message batch metadata persisted in `AuthorizedSqlStore`; SQL backend required at startup | Behavior | [#7047](https://github.com/ogx-ai/ogx/pull/7047) |
| `response_format.json_schema.name` required; incomplete `text.format` → 400 at parse time | Behavior | [#6907](https://github.com/ogx-ai/ogx/pull/6907) |

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

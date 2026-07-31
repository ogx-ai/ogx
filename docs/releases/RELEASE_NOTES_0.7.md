<!-- markdownlint-disable MD036 -->
# OGX 0.7 Release Notes

**Release Date:** April 2026

Release 0.7 is a major release focused on completing the transition to OpenAI API conformance, introducing comprehensive observability metrics, and significant API cleanup. This release removes the fine-tuning API, completes the FastAPI router migration, removes legacy providers (TGI, HuggingFace), renames core concepts for clarity (agents to responses, knowledge_search to file_search, meta-reference to builtin), and adds structured logging via structlog. New providers include Infinispan vector-io and inline Docling for PDF parsing.

## Breaking Changes

### Summary

> **Note:** Each change is described in detail with migration instructions in the sections below.

**Hard Breaking Changes (action required before upgrading):**

| Change | Migration | PR |
|--------|-----------|-----|
| Fine-tuning API removed | Remove all `/post-training` and `/fine-tuning` API usage | [#5104](https://github.com/ogx-ai/ogx/pull/5104) |
| `meta-reference` providers renamed to `builtin` | Replace `inline::meta-reference` with `inline::builtin` in configs | [#5131](https://github.com/ogx-ai/ogx/pull/5131) |
| `knowledge_search` renamed to `file_search` | Replace `knowledge_search` with `file_search` in tool names and API calls | [#5186](https://github.com/ogx-ai/ogx/pull/5186) |
| Agents API renamed to Responses API | Replace `/agents` endpoints with `/responses` | [#5195](https://github.com/ogx-ai/ogx/pull/5195) |
| `tool_groups` removed from public API | Remove `tool_groups` registration; providers auto-register tools | [#4997](https://github.com/ogx-ai/ogx/pull/4997) |
| TGI and HuggingFace inference providers removed | Switch to `remote::vllm`, `remote::ollama`, or other providers | [#5333](https://github.com/ogx-ai/ogx/pull/5333) |
| `register/unregister` model endpoints removed | Use standard CRUD endpoints for model management | [#5341](https://github.com/ogx-ai/ogx/pull/5341) |
| `@webmethod` decorator removed | All APIs now use FastAPI routers exclusively | [#5248](https://github.com/ogx-ai/ogx/pull/5248) |
| `rag-runtime` provider renamed to `file-search` | Replace `inline::rag-runtime` with `inline::file-search` in configs | [#5187](https://github.com/ogx-ai/ogx/pull/5187) |
| Duplicate `dataset_id` parameter removed from append-rows | Remove `dataset_id` from request body (use path parameter only) | [#4849](https://github.com/ogx-ai/ogx/pull/4849) |
| `/files/{file_id}` GET response format unified | Update clients expecting provider-specific response shapes | [#5154](https://github.com/ogx-ai/ogx/pull/5154) |
| OpenAI API schema transforms added | Review response schemas for new conformance fields | [#5166](https://github.com/ogx-ai/ogx/pull/5166) |
| `starter-gpu` distribution removed | Use `starter` distribution with remote inference providers | [#5279](https://github.com/ogx-ai/ogx/pull/5279) |

**Behavior Changes (no code changes required, but be aware):**

| Change | Note | PR |
|--------|------|-----|
| `sentence_transformers` `trust_remote_code` defaults to `False` | Set `trust_remote_code: true` in config if using custom models | [#4602](https://github.com/ogx-ai/ogx/pull/4602) |
| Inline neural rerank for RAG | Reranking now available as built-in capability | [#4877](https://github.com/ogx-ai/ogx/pull/4877) |
| Logging migrated to structlog | Log output format changed to structured key-value pairs | [#5215](https://github.com/ogx-ai/ogx/pull/5215) |

---

### Hard Breaking Changes

These changes take effect immediately and require updates before upgrading.

#### 1. Fine-Tuning API Removed ([#5104](https://github.com/ogx-ai/ogx/pull/5104))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** All users of the post-training/fine-tuning API

The entire fine-tuning API has been removed from OGX. This includes all `/post-training` endpoints and related provider implementations.

**Migration:** Remove all fine-tuning API calls. Use external fine-tuning services directly.

---

#### 2. `meta-reference` Providers Renamed to `builtin` ([#5131](https://github.com/ogx-ai/ogx/pull/5131))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** All users with `meta-reference` in their distribution configs

**Before:**

```yaml
provider_type: inline::meta-reference
```

**After:**

```yaml
provider_type: inline::builtin
```

**Migration:** Search and replace `inline::meta-reference` with `inline::builtin` in all config files.

```bash
grep -r "meta-reference" your-config-directory/
```

---

#### 3. `knowledge_search` Renamed to `file_search` ([#5186](https://github.com/ogx-ai/ogx/pull/5186))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** All users referencing `knowledge_search` tool name

**Before:**

```text
tools=[{"type": "knowledge_search", ...}]
```

**After:**

```text
tools=[{"type": "file_search", ...}]
```

**Migration:**

```bash
grep -r "knowledge_search" your-project/
```

---

#### 4. Agents API Renamed to Responses API ([#5195](https://github.com/ogx-ai/ogx/pull/5195))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** All users of the agents API endpoints

The agents API has been renamed to the Responses API to align with OpenAI's naming convention. All `/agents` endpoints are now served under `/responses`.

**Migration:** Update API endpoint paths from `/agents/*` to `/responses/*`.

---

#### 5. `tool_groups` Removed from Public API ([#4997](https://github.com/ogx-ai/ogx/pull/4997))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Users who manually register tool groups

Tool groups are now auto-registered from provider specs. The public API for registering/unregistering tool groups has been removed.

**Migration:** Remove any `tool_groups` registration calls. Ensure your provider specs include `toolgroup_id` where needed.

---

#### 6. TGI and HuggingFace Inference Providers Removed ([#5333](https://github.com/ogx-ai/ogx/pull/5333))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Users relying on `remote::tgi` or `remote::huggingface` providers

**Migration:** Switch to `remote::vllm`, `remote::ollama`, or another supported inference provider.

---

#### 7. Deprecated `register`/`unregister` Model Endpoints Removed ([#5341](https://github.com/ogx-ai/ogx/pull/5341))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Users calling deprecated model registration endpoints

**Migration:** Use standard model management endpoints.

---

#### 8. `@webmethod` Decorator Removed — FastAPI Router Migration Complete ([#5248](https://github.com/ogx-ai/ogx/pull/5248))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Custom provider authors who used `@webmethod`

All APIs now use FastAPI routers exclusively. The legacy `@webmethod` decorator has been removed.

**Migration:** Convert any custom endpoints to use FastAPI router decorators.

---

#### 9. `rag-runtime` Provider Renamed to `file-search` ([#5187](https://github.com/ogx-ai/ogx/pull/5187))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Users with `inline::rag-runtime` or `builtin::rag` in configs

**Before:**

```yaml
provider_type: inline::rag-runtime
toolgroup_id: builtin::rag
```

**After:**

```yaml
provider_type: inline::file-search
toolgroup_id: builtin::file-search
```

---

#### 10. Duplicate `dataset_id` Removed from Append-Rows ([#4849](https://github.com/ogx-ai/ogx/pull/4849))

*Contributed by Eoin Fennessy (Red Hat) — @eoinfennessy*

**Impact:** API clients sending `dataset_id` in both the URL path and request body

**Migration:** Remove `dataset_id` from the request body; it is now taken exclusively from the URL path parameter.

---

#### 11. `/files/{file_id}` GET Response Unified ([#5154](https://github.com/ogx-ai/ogx/pull/5154))

*Contributed by @r3v5*

**Impact:** Clients parsing file metadata responses

The GET endpoint now returns a consistent response shape across all providers, eliminating provider-specific differences.

---

#### 12. OpenAI API Schema Transforms ([#5166](https://github.com/ogx-ai/ogx/pull/5166))

*Contributed by Nathan Weinberg (Red Hat) — @nathan-weinberg*

**Impact:** Clients relying on the exact API response schema

Schema transforms and new types have been added to improve OpenAI API conformance. Response shapes may differ from previous versions.

---

#### 13. `starter-gpu` Distribution Removed ([#5279](https://github.com/ogx-ai/ogx/pull/5279))

*Contributed by Sebastien Han (Red Hat) — @leseb*

**Impact:** Users of the `starter-gpu` distribution

**Migration:** Use the `starter` distribution with remote inference providers instead.

---

### Behavior Changes

#### `sentence_transformers` `trust_remote_code` Now Defaults to `False` ([#4602](https://github.com/ogx-ai/ogx/pull/4602))

*Contributed by Derek Higgins (Red Hat) — @derekhiggins*

For security, `trust_remote_code` now defaults to `False` for sentence_transformers models. If you use custom models that require remote code execution, set `trust_remote_code: true` in your provider config.

#### Structured Logging via structlog ([#5215](https://github.com/ogx-ai/ogx/pull/5215))

*Contributed by Sebastien Han (Red Hat) — @leseb*

All logging has been migrated to structlog with structured key-value output. Log parsing tools that rely on the previous format may need to be updated.

---

## New Features

### Reasoning Output Support

- **Reasoning output in Responses API** — Models can now return reasoning/thinking traces as part of response output ([#5206](https://github.com/ogx-ai/ogx/pull/5206) by @robinnarsinghranabhat)
- **Reasoning as valid conversation item** — Reasoning traces can be included in conversation history ([#5392](https://github.com/ogx-ai/ogx/pull/5392))
- **Reasoning output types in OpenAI spec** — Added `reasoning` output type to the Responses API spec ([#5357](https://github.com/ogx-ai/ogx/pull/5357))

### Comprehensive Observability Metrics

- **API-level request metrics** — Track request counts, latency, and error rates at the API layer ([#5201](https://github.com/ogx-ai/ogx/pull/5201) by @gyliu513)
- **Inference metrics** — Token throughput, latency, and model-level metrics ([#5320](https://github.com/ogx-ai/ogx/pull/5320) by @gyliu513)
- **Vector IO metrics** — Performance metrics for vector store operations ([#5096](https://github.com/ogx-ai/ogx/pull/5096) by @gyliu513)
- **Parameter usage metrics for Responses API** — Track parameter usage patterns ([#5255](https://github.com/ogx-ai/ogx/pull/5255) by @gyliu513)

### Inline Neural Rerank for RAG ([#4877](https://github.com/ogx-ai/ogx/pull/4877))

*Contributed by @r3v5*

RAG pipelines can now use built-in neural reranking without external services, improving search quality with cross-encoder models.

### Inline Docling Provider for PDF Parsing ([#5049](https://github.com/ogx-ai/ogx/pull/5049))

*Contributed by @alinaryan*

Structure-aware PDF parsing using Docling for high-quality document ingestion into vector stores.

### Background Response Cancellation ([#5268](https://github.com/ogx-ai/ogx/pull/5268))

*Contributed by Charlie Doern (Red Hat) — @cdoern*

Added a cancel endpoint for background responses, allowing clients to abort long-running response generation.

### Connector API Promoted to v1beta ([#5129](https://github.com/ogx-ai/ogx/pull/5129))

*Contributed by Sebastien Han (Red Hat) — @leseb*

The Connector API for MCP server management has been promoted from `v1alpha` to `v1beta`, signaling increased API stability.

### `stream_options` Parameter Support ([#4815](https://github.com/ogx-ai/ogx/pull/4815))

*Contributed by @gyliu513*

Added support for the `stream_options` parameter in chat completions, enabling `include_usage` in streaming responses for OpenAI conformance.

### Forward Headers for Inference Passthrough ([#5134](https://github.com/ogx-ai/ogx/pull/5134))

*Contributed by @skamenan7*

The inference passthrough provider now supports forwarding custom headers to upstream backends.

### Form-Encoded Content Type for Responses API ([#5193](https://github.com/ogx-ai/ogx/pull/5193))

*Contributed by @r3v5*

The Responses API now accepts `application/x-www-form-urlencoded` content type in addition to JSON.

### PGVector Filter Support ([#5111](https://github.com/ogx-ai/ogx/pull/5111))

*Contributed by @franciscojavierarceo*

PGVector now supports metadata filters for vector search queries, and f-string usage in table names has been replaced for safety.

### Configurable asyncpg Connection Pools ([#5160](https://github.com/ogx-ai/ogx/pull/5160))

*Contributed by @iamemilio*

PostgreSQL connection pool settings (min/max connections, timeouts) are now configurable via provider config.

### Provider Compatibility Matrix ([#5113](https://github.com/ogx-ai/ogx/pull/5113), [#5115](https://github.com/ogx-ai/ogx/pull/5115))

*Contributed by Sebastien Han (Red Hat) — @leseb*

A provider compatibility matrix for the Responses API and provider version tracking have been added to help users understand which features each provider supports.

### Responses API Test Coverage Analyzer ([#5101](https://github.com/ogx-ai/ogx/pull/5101))

Conformance annotations and a test coverage analyzer for the Responses API have been added to track OpenAI specification coverage.

## New Providers

### Infinispan Vector-IO Provider ([#4839](https://github.com/ogx-ai/ogx/pull/4839))

*Contributed by @rigazilla*

New vector store provider using Infinispan for distributed, high-performance vector storage.

## Refactoring

- **WatsonX: LiteLLM replaced with OpenAI mixin** — Cleaner, more maintainable WatsonX provider ([#5133](https://github.com/ogx-ai/ogx/pull/5133) by @c99cd1a93)
- **`file_search` decoupled from legacy `knowledge_search` tool_groups** ([#5175](https://github.com/ogx-ai/ogx/pull/5175) by @leseb)
- **Large files split into focused modules** ([#5281](https://github.com/ogx-ai/ogx/pull/5281), [#5299](https://github.com/ogx-ai/ogx/pull/5299) by @leseb)
- **Tools API converted to FastAPI router** ([#5246](https://github.com/ogx-ai/ogx/pull/5246) by @leseb)
- **Unused `LiteLLMOpenAIMixin` removed** ([#5159](https://github.com/ogx-ai/ogx/pull/5159))

## Performance Improvements

- **Lazy-load numpy, faiss, sqlite_vec** in vector_io providers to reduce startup memory ([#5118](https://github.com/ogx-ai/ogx/pull/5118))
- **Lazy-load torch and transformers** in prompt_guard ([#5117](https://github.com/ogx-ai/ogx/pull/5117))
- **Lazy-load torch** in embedding_mixin to reduce startup memory ([#5116](https://github.com/ogx-ai/ogx/pull/5116))
- **Lazy-load braintrust autoevals** to reduce idle memory (~63MB) ([#5078](https://github.com/ogx-ai/ogx/pull/5078))

## Security Fixes

- **Path traversal and header injection defenses** ([#5086](https://github.com/ogx-ai/ogx/pull/5086) by @rhdedgar)
- **CVE-2026-33236**: Bump nltk to 3.9.4 ([#5259](https://github.com/ogx-ai/ogx/pull/5259) by @eoinfennessy)
- **CVE-2026-30922**: Bump pyasn1 to 0.6.3 ([#5207](https://github.com/ogx-ai/ogx/pull/5207) by @eoinfennessy)
- **CVE-2026-32597**: Bump pyjwt to 2.12.0 ([#5127](https://github.com/ogx-ai/ogx/pull/5127) by @eoinfennessy)

## Bug Fixes

- Fix provider_data_var context leak ([#5227](https://github.com/ogx-ai/ogx/pull/5227) by @jaideepr97)
- Prevent OTel context leak in fire-and-forget background tasks ([#5168](https://github.com/ogx-ai/ogx/pull/5168) by @iamemilio)
- Disable asyncpg OTel auto-instrumentation to prevent duplicate DB spans ([#5158](https://github.com/ogx-ai/ogx/pull/5158))
- Fix asyncio event loop mismatch via operation deferral ([#5130](https://github.com/ogx-ai/ogx/pull/5130) by @derekhiggins)
- Improve chat completions OpenAI conformance ([#5108](https://github.com/ogx-ai/ogx/pull/5108) by @cdoern)
- Multi-worker cache synchronization for vector stores ([#5076](https://github.com/ogx-ai/ogx/pull/5076) by @elinacse)
- Replace blocking `requests` calls with async `httpx` in WatsonX ([#5280](https://github.com/ogx-ai/ogx/pull/5280) by @gyliu513) and remote providers ([#5162](https://github.com/ogx-ai/ogx/pull/5162) by @gyliu513)
- Use SDK-native model names for Vertex AI ([#5169](https://github.com/ogx-ai/ogx/pull/5169) by @major)
- Fix vLLM health() and rerank() TLS and auth credentials ([#5340](https://github.com/ogx-ai/ogx/pull/5340) by @gyliu513)
- Fix vLLM rerank() provider-data-aware API key lookup ([#5374](https://github.com/ogx-ai/ogx/pull/5374))
- Fix require_approval field check in ApprovalFilter ([#5288](https://github.com/ogx-ai/ogx/pull/5288) by @jaideepr97)
- Gate conversation sync on store flag to prevent data leak when store=false ([#5305](https://github.com/ogx-ai/ogx/pull/5305) by @jaideepr97)
- Fix assistant message rewriting in _separate_tool_calls ([#5303](https://github.com/ogx-ai/ogx/pull/5303) by @jaideepr97)
- Handle asyncio.CancelledError in metrics try/except blocks ([#5336](https://github.com/ogx-ai/ogx/pull/5336) by @gyliu513)
- Race condition fix in background response cancel ([#5363](https://github.com/ogx-ai/ogx/pull/5363) by @leseb)
- Allow multi-worker server with dual-stack IPv6 support ([#5284](https://github.com/ogx-ai/ogx/pull/5284) by @derekhiggins)
- Auto-expand provider dependencies for `--providers` in stack CLI ([#4654](https://github.com/ogx-ai/ogx/pull/4654) by @gyliu513)
- Make InmemoryKVStore.delete consistent with other backends ([#5289](https://github.com/ogx-ai/ogx/pull/5289) by @gyliu513)
- Convert Path to str in _build_ssl_context() for httpx compatibility ([#5380](https://github.com/ogx-ai/ogx/pull/5380) by @gyliu513)
- Surface tiktoken encoding check at provider startup ([#5401](https://github.com/ogx-ai/ogx/pull/5401) by @Bobbins228)
- Pre-cache tiktoken cl100k_base encoding at image build time ([#5391](https://github.com/ogx-ai/ogx/pull/5391) by @Bobbins228)
- Fix milvus hybrid ranker usage ([#5312](https://github.com/ogx-ai/ogx/pull/5312) by @jakub-walaszczyk)
- Remove duplicate decode ([#5177](https://github.com/ogx-ai/ogx/pull/5177))
- Optimize connector listing ([#5164](https://github.com/ogx-ai/ogx/pull/5164))
- Remove references to defunct inline::builtin inference provider ([#5174](https://github.com/ogx-ai/ogx/pull/5174) by @leseb)

## Documentation

- Rewrite README and docs to lead with OpenAI API compatibility ([#5323](https://github.com/ogx-ai/ogx/pull/5323))
- Add AGENTS.md with guidelines for AI coding agents ([#5211](https://github.com/ogx-ai/ogx/pull/5211))
- Add architecture documentation and module-level READMEs ([#5213](https://github.com/ogx-ai/ogx/pull/5213))
- Blog post on Open Responses compliance and OpenAI compatibility ([#5232](https://github.com/ogx-ai/ogx/pull/5232))
- Blog post on OGX observability ([#5387](https://github.com/ogx-ai/ogx/pull/5387))
- Agentic flows tutorial blog post ([#5035](https://github.com/ogx-ai/ogx/pull/5035))
- Blog post about Responses API ([#5196](https://github.com/ogx-ai/ogx/pull/5196))
- Docling provider setup and usage docs ([#5329](https://github.com/ogx-ai/ogx/pull/5329))
- Multi-tenant isolation example for conversations and responses ([#5176](https://github.com/ogx-ai/ogx/pull/5176))
- Update stale documentation to reflect current architecture ([#5393](https://github.com/ogx-ai/ogx/pull/5393))
- Mintlify-inspired documentation UI improvements ([#5405](https://github.com/ogx-ai/ogx/pull/5405))
- Add docstrings to public classes and functions ([#5267](https://github.com/ogx-ai/ogx/pull/5267))
- Update README badges with logos, conformance score, and DeepWiki ([#5389](https://github.com/ogx-ai/ogx/pull/5389))

## CI/CD Improvements

- Auto-record integration tests on PRs with multi-provider support ([#5123](https://github.com/ogx-ai/ogx/pull/5123))
- Add Bedrock to responses CI suite with recordings ([#5254](https://github.com/ogx-ai/ogx/pull/5254))
- Add WatsonX Responses API integration test recordings ([#5120](https://github.com/ogx-ai/ogx/pull/5120))
- Test Responses API against Azure AI Foundry ([#5107](https://github.com/ogx-ai/ogx/pull/5107))
- Add GCP Workload Identity Federation for Vertex AI recording workflow ([#5276](https://github.com/ogx-ai/ogx/pull/5276))
- Cache HuggingFace models and datasets for offline replay tests ([#5382](https://github.com/ogx-ai/ogx/pull/5382))
- Add conventional-pre-commit for commit validation ([#5251](https://github.com/ogx-ai/ogx/pull/5251))
- Add markdownlint and actionlint pre-commit hooks ([#5271](https://github.com/ogx-ai/ogx/pull/5271), [#5285](https://github.com/ogx-ai/ogx/pull/5285))
- Add mypy pre-commit hook enforcement ([#5269](https://github.com/ogx-ai/ogx/pull/5269))
- Replace Mergify queue with GitHub merge queue ([#5383](https://github.com/ogx-ai/ogx/pull/5383))
- Test last 3 release branches in scheduled CI ([#5277](https://github.com/ogx-ai/ogx/pull/5277))
- Remove docker mode from integration test matrix ([#5311](https://github.com/ogx-ai/ogx/pull/5311))

## Upgrade Guide

### Before Upgrading

These hard breaking changes require updates before you can run 0.7:

1. **Check for fine-tuning API usage:**

   ```bash
   grep -r "post-training\|fine.tuning\|fine_tuning" your-project/
   ```

   Remove all fine-tuning API calls.

2. **Update provider names in configs:**

   ```bash
   grep -r "meta-reference" your-config-directory/
   grep -r "rag-runtime" your-config-directory/
   grep -r "starter-gpu" your-config-directory/
   ```

   - `inline::meta-reference` -> `inline::builtin`
   - `inline::rag-runtime` -> `inline::file-search`
   - `builtin::rag` -> `builtin::file-search`
   - `starter-gpu` -> `starter`

3. **Update tool names:**

   ```bash
   grep -r "knowledge_search" your-project/
   ```

   Replace with `file_search`.

4. **Update API endpoints:**

   ```bash
   grep -r "/agents" your-project/
   ```

   Replace agents API calls with responses API equivalents.

5. **Remove tool_groups registration:**

   ```bash
   grep -r "tool_groups\|register_tool" your-project/
   ```

   Tool groups are now auto-registered from provider specs.

6. **Check for removed providers:**

   ```bash
   grep -r "remote::tgi\|remote::huggingface" your-config-directory/
   ```

   Switch to `remote::vllm`, `remote::ollama`, or another supported provider.

7. **Check for deprecated model endpoints:**

   ```bash
   grep -r "register_model\|unregister_model" your-project/
   ```

### After Upgrading

- Review log output if you have log parsing tools, as logging now uses structured key-value format via structlog.
- If using `sentence_transformers` with custom models requiring remote code, add `trust_remote_code: true` to your provider config.

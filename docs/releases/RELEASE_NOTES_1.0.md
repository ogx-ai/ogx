<!-- markdownlint-disable MD036 -->
# OGX 1.0 Release Notes

**Release Date:** May 2026

OGX 1.0 marks the project's first major-stable release. The `/v1` HTTP API surface is now covered by the stability contract documented in [`docs/docs/concepts/apis/api_leveling.mdx`](../concepts/apis/api_leveling.mdx): no breaking changes to `/v1` datatypes or on-disk storage schemas within the 1.x line. APIs that are not yet stable continue to live under `/v1alpha` and `/v1beta`.

The headline themes for 1.0:

- **Multi-tenancy core for MaaS deployments** — first-class tenant isolation across storage, vector stores, prompts, and conversations.
- **Authorization as a first-class storage concern** — all access-controlled APIs now go through `AuthorizedSqlStore`.
- **Admin/data plane split** — administrative routes (`tools`, `connectors`) moved out of the user-facing `/v1` surface.
- **Safety API removed** — replaced by the OpenAI-compatible moderation endpoint.
- **`ogx-api` namespace split** — `ogx_api.types` (datatypes) and `ogx_api.provider` (provider authoring) are now distinct surfaces with their own stability rules.
- **Gateway-first server architecture** — edge concerns (auth, rate limiting, tenancy) consolidated at the gateway.

## Breaking Changes

### Summary

> **Note:** Each change is described in detail with migration instructions in the sections below.

**Hard Breaking Changes (action required before upgrading):**

| Change | Migration | PR |
|--------|-----------|-----|
| Safety API removed; use `/v1/moderations` | Replace `run-shield` and `shields` calls with the moderation endpoint | [#5291](https://github.com/ogx-ai/ogx/pull/5291), [#5744](https://github.com/ogx-ai/ogx/pull/5744) |
| Tools routes moved to `/v1/admin/tools` | Update tool-management clients to `/v1/admin/tools` | [#5787](https://github.com/ogx-ai/ogx/pull/5787) |
| Connectors routes moved to `/v1alpha/admin/connectors` | Update connector-management clients | [#5659](https://github.com/ogx-ai/ogx/pull/5659) |
| Multi-tenancy enforcement on by default | Provide tenant credentials/headers; review per-tenant data scoping | [#5756](https://github.com/ogx-ai/ogx/pull/5756) |
| `AuthorizedSqlStore` required for access-controlled APIs | Custom providers must migrate from raw SqlStore | [#5776](https://github.com/ogx-ai/ogx/pull/5776) |
| Connectors and batches KVStore migrated to `AuthorizedSqlStore` | On-disk migration runs on first boot; back up state first | [#5757](https://github.com/ogx-ai/ogx/pull/5757) |
| `/v1/models` response shape now multi-SDK | Clients parsing the legacy shape must update | [#5522](https://github.com/ogx-ai/ogx/pull/5522) |
| `ogx-api` split into `ogx_api.types` and `ogx_api.provider` | Update imports in provider code and downstream consumers | [#5740](https://github.com/ogx-ai/ogx/pull/5740) |
| `logprobs` type changed from `bool` to `int` on Completions | Update clients passing `logprobs=True/False` to an integer | [#5343](https://github.com/ogx-ai/ogx/pull/5343) |
| `ogx stack rm` CLI command removed | Use distribution-specific teardown instead | [#5735](https://github.com/ogx-ai/ogx/pull/5735) |

---

### Hard Breaking Changes

#### 1. Safety API Removed; Replaced by Moderation Endpoint ([#5291](https://github.com/ogx-ai/ogx/pull/5291), [#5744](https://github.com/ogx-ai/ogx/pull/5744))

**Impact:** All users of `/v1/safety/run-shield` and the `/v1/shields` list/get endpoints.

The standalone Safety API has been removed. Content moderation is now served by the OpenAI-compatible `/v1/moderations` endpoint.

**Migration:** Replace `run-shield` calls with `/v1/moderations`. Shield registration and listing endpoints are removed; configure moderation models via provider config instead.

---

#### 2. Tools Routes Moved to `/v1/admin/tools` ([#5787](https://github.com/ogx-ai/ogx/pull/5787))

**Impact:** Clients calling `/v1/tools` for tool management.

Tool-management endpoints are now under the administrative path `/v1/admin/tools`, separating the control plane from the user-facing API.

**Migration:** Update tool-management clients to the new path. End-user code that *invokes* tools through Responses/Chat Completions is unaffected.

---

#### 3. Connectors Routes Moved to `/v1alpha/admin/connectors` ([#5659](https://github.com/ogx-ai/ogx/pull/5659))

**Impact:** Clients managing MCP connectors.

Connector management has moved to `/v1alpha/admin/connectors`, reflecting its admin-plane role and pre-stable API level.

**Migration:** Update connector management clients to the new path.

---

#### 4. Multi-Tenancy Core for MaaS Deployments ([#5756](https://github.com/ogx-ai/ogx/pull/5756))

**Impact:** Any deployment running without explicit tenant configuration.

OGX now ships with first-class multi-tenant isolation suitable for Model-as-a-Service deployments. Tenant context is propagated through storage, vector stores, conversations, and prompts. Vector store metadata isolation ([#5782](https://github.com/ogx-ai/ogx/pull/5782)) and prompts tenant isolation tests ([#5758](https://github.com/ogx-ai/ogx/pull/5758)) ship alongside.

**Migration:** Single-tenant deployments must explicitly opt into the default tenant or configure tenant credentials. Review any custom providers that touched storage directly — they must now go through `AuthorizedSqlStore` (see below).

---

#### 5. `AuthorizedSqlStore` Required for Access-Controlled APIs ([#5776](https://github.com/ogx-ai/ogx/pull/5776))

**Impact:** Custom provider authors using raw `SqlStore` for tenant-scoped data.

APIs that require access control now enforce use of `AuthorizedSqlStore`, which integrates tenant identity into every query.

**Migration:** Update custom providers to construct `AuthorizedSqlStore` rather than `SqlStore` for tenant-scoped tables.

---

#### 6. Connectors and Batches KVStore Migrated to AuthorizedSqlStore ([#5757](https://github.com/ogx-ai/ogx/pull/5757))

**Impact:** Existing deployments with stored connector or batch state.

On first boot of 1.0, connector and batch state migrates from the legacy KVStore layout to `AuthorizedSqlStore`-backed tables.

**Migration:** Back up storage before upgrading. The migration is one-way; downgrade is not supported.

---

#### 7. Multi-SDK Response Shapes for `/v1/models` ([#5522](https://github.com/ogx-ai/ogx/pull/5522))

**Impact:** Clients parsing the legacy `/v1/models` response.

`/v1/models` now returns response shapes aligned with each SDK convention (OpenAI, Anthropic, etc.) rather than a single OGX-specific shape.

**Migration:** Update clients parsing model metadata to the new shape. See the API reference for the full schema.

---

#### 8. `ogx-api` Namespace Split into `types` and `provider` ([#5740](https://github.com/ogx-ai/ogx/pull/5740))

**Impact:** Code importing from `ogx_api` directly.

`ogx_api` is now split into two surfaces with distinct stability rules:

- `ogx_api.types` — Pydantic datatypes used over the wire; follows the `/v1` stability contract.
- `ogx_api.provider` — base classes and protocols for provider authors; evolves with provider architecture.

See [#5719](https://github.com/ogx-ai/ogx/pull/5719) for the datatype stability documentation.

**Migration:** Update imports. The old top-level imports remain as transitional shims in 1.0 but will be removed in 2.0.

---

#### 9. `logprobs` Type on Completions: `bool` → `int` ([#5343](https://github.com/ogx-ai/ogx/pull/5343))

**Impact:** Clients passing `logprobs=True` or `logprobs=False` to `/v1/completions`.

The `logprobs` parameter now takes an integer (number of top logprobs to return), matching OpenAI's Completions API.

**Migration:** Replace `True` with an integer (e.g., `1` or `5`); replace `False` with omission of the field.

---

#### 10. `ogx stack rm` CLI Command Removed ([#5735](https://github.com/ogx-ai/ogx/pull/5735))

**Impact:** Users running `ogx stack rm`.

The command was undiscoverable and rarely used.

**Migration:** Use the distribution-specific teardown for your deployment (container stop, `uv` venv removal, etc.).

---

## New Features

### Multi-Tenancy and Authorization

- **Multi-tenancy core for MaaS deployments** ([#5756](https://github.com/ogx-ai/ogx/pull/5756))
- **Tenant isolation for vector store metadata** ([#5782](https://github.com/ogx-ai/ogx/pull/5782))
- **`AuthorizedSqlStore` enforcement across access-controlled APIs** ([#5776](https://github.com/ogx-ai/ogx/pull/5776))
- **Connectors and batches migrated to `AuthorizedSqlStore`** ([#5757](https://github.com/ogx-ai/ogx/pull/5757))
- **Prompts tenant isolation CI workflow** ([#5758](https://github.com/ogx-ai/ogx/pull/5758))
- **GitHub org-membership fetching for RBAC attribute mapping** ([#5711](https://github.com/ogx-ai/ogx/pull/5711))

### Gateway-First Server Architecture ([#5750](https://github.com/ogx-ai/ogx/pull/5750))

Edge concerns — authentication, rate limiting, tenant resolution — are now consolidated at the gateway layer rather than scattered across handlers.

### New Inference Providers and Compatibility

- **Gemini and Azure support in `letsgo`** ([#5706](https://github.com/ogx-ai/ogx/pull/5706))
- **Claude Code compatibility in `letsgo`** ([#5709](https://github.com/ogx-ai/ogx/pull/5709))
- **`inline::auto` file processor in `letsgo`** ([#5704](https://github.com/ogx-ai/ogx/pull/5704))
- **CLI showcase for Claude Code, Codex, and OpenCode on the landing page** ([#5716](https://github.com/ogx-ai/ogx/pull/5716))

### File Processing

- **`inline::auto` composite file processor** ([#5673](https://github.com/ogx-ai/ogx/pull/5673))
- **`inline::markitdown` provider wired into the auto dispatcher** ([#5688](https://github.com/ogx-ai/ogx/pull/5688))

### Conversations and Interactions

- **`previous_interaction_id` for multi-turn conversations** ([#5669](https://github.com/ogx-ai/ogx/pull/5669))
- **Enhanced OpenAI API coverage for Conversations** ([#5748](https://github.com/ogx-ai/ogx/pull/5748))
- **Chat completion message listing endpoint** ([#5459](https://github.com/ogx-ai/ogx/pull/5459))
- **Cursor pagination and `has_more` for conversation list_items** ([#5612](https://github.com/ogx-ai/ogx/pull/5612))

### Files

- **Enhanced OpenAI API coverage for Files API** ([#5747](https://github.com/ogx-ai/ogx/pull/5747))

### CLI

- **Top-level `ogx run` and `ogx letsgo` shortcuts** ([#5689](https://github.com/ogx-ai/ogx/pull/5689))

### `ogx-api` Package

- **`ogx_api.provider` and `ogx_api.types` namespaces introduced** ([#5740](https://github.com/ogx-ai/ogx/pull/5740))
- **Datatype stability and package-surface documentation** ([#5719](https://github.com/ogx-ai/ogx/pull/5719))

## Performance Improvements

- **asyncpg connection pool in PostgreSQL kvstore** ([#5734](https://github.com/ogx-ai/ogx/pull/5734))
- **psycopg2 → asyncpg migration in PostgreSQL KV store** ([#5739](https://github.com/ogx-ai/ogx/pull/5739))
- **Batched guardrail checks during Responses API streaming** ([#5664](https://github.com/ogx-ai/ogx/pull/5664))
- **Explicit HTTP timeouts and improved connection pooling** ([#5737](https://github.com/ogx-ai/ogx/pull/5737))

## Security Fixes

- **Python SAST via Ruff bandit rules and CodeQL scanning** ([#5738](https://github.com/ogx-ai/ogx/pull/5738))
- **High-severity CVE patches in python-multipart, protobuf, lxml, and npm packages** ([#5775](https://github.com/ogx-ai/ogx/pull/5775))
- **Critical and high-severity dependency upgrades** ([#5742](https://github.com/ogx-ai/ogx/pull/5742))
- **CVE-pinned transitive deps moved to constraint-dependencies** ([#5707](https://github.com/ogx-ai/ogx/pull/5707))
- **Dependabot alert resolution in `ogx_api` constraint-dependencies** ([#5778](https://github.com/ogx-ai/ogx/pull/5778))
- **NVIDIA provider: proper URL parsing for hostname validation** ([#5777](https://github.com/ogx-ai/ogx/pull/5777))

## Bug Fixes

- **Thread-safe `OGXAsLibraryClient`** ([#5773](https://github.com/ogx-ai/ogx/pull/5773))
- **SQLite-vec WAL mode and `busy_timeout`** ([#5428](https://github.com/ogx-ai/ogx/pull/5428))
- **Gemini streaming: strip per-chunk usage to prevent token overcounting** ([#5171](https://github.com/ogx-ai/ogx/pull/5171))
- **Responses API: honor `include` on input item retrieval** ([#5605](https://github.com/ogx-ai/ogx/pull/5605))
- **Auth: return 503 only for auth-service outages** ([#5715](https://github.com/ogx-ai/ogx/pull/5715))
- **Auth: respect `verify_tls` in OAuth2 introspection** ([#5710](https://github.com/ogx-ai/ogx/pull/5710))
- **OpenAI: clamp `max_tokens` to per-model limits** ([#5696](https://github.com/ogx-ai/ogx/pull/5696))
- **Ollama: tolerate dict-backed reasoning messages in OpenAI preprocessing** ([#5638](https://github.com/ogx-ai/ogx/pull/5638))
- **Vertex AI: preserve `reasoning_content` as thought parts in multi-turn conversations** ([#5677](https://github.com/ogx-ai/ogx/pull/5677))
- **NVIDIA safety: use `provider_resource_id` for NeMoGuardrails model** ([#5726](https://github.com/ogx-ai/ogx/pull/5726))
- **PostgreSQL storage: filter expired rows in `keys_in_range()`** ([#5712](https://github.com/ogx-ai/ogx/pull/5712))
- **SQL store: propagate real add-column errors** ([#5713](https://github.com/ogx-ai/ogx/pull/5713))
- **KVStore: namespace support and expiration filtering across all backends** ([#5731](https://github.com/ogx-ai/ogx/pull/5731))
- **Vector IO: improved error reporting for file processor rejections** ([#5690](https://github.com/ogx-ai/ogx/pull/5690))
- **Starter extra: add missing dependencies** ([#5674](https://github.com/ogx-ai/ogx/pull/5674))
- **Test parametrization: use `provider_type` instead of `provider_id`** ([#5263](https://github.com/ogx-ai/ogx/pull/5263))

## Refactoring

- **Server adopts gateway-first architecture for edge concerns** ([#5750](https://github.com/ogx-ai/ogx/pull/5750))
- **Dead code removal across the codebase** ([#5779](https://github.com/ogx-ai/ogx/pull/5779))

## Documentation

- **Datatype stability and `ogx-api` package surfaces** ([#5719](https://github.com/ogx-ai/ogx/pull/5719))
- **Consistent agentic API layer** blog post ([#5687](https://github.com/ogx-ai/ogx/pull/5687))
- **Landing page promotes library mode** ([#5761](https://github.com/ogx-ai/ogx/pull/5761))
- **Provider card / DocCardList redesign** ([#5694](https://github.com/ogx-ai/ogx/pull/5694))
- **Claude Code `--model` examples and routing clarification** ([#5692](https://github.com/ogx-ai/ogx/pull/5692))

## CI/CD Improvements

- **Pin `record-integration-tests` action refs to merge commit SHA** ([#5762](https://github.com/ogx-ai/ogx/pull/5762))
- **Break record-integration-tests feedback loop** ([#5781](https://github.com/ogx-ai/ogx/pull/5781))
- **All Ollama variants in the re-record workflow** ([#5746](https://github.com/ogx-ai/ogx/pull/5746))
- **Reduced GitHub Actions runner usage for free-plan constraints** ([#5751](https://github.com/ogx-ai/ogx/pull/5751))
- **Server-only client on PR integration tests** ([#5697](https://github.com/ogx-ai/ogx/pull/5697))
- **Conventional-commit scope allowed in breaking-change acknowledgment regex** ([#5718](https://github.com/ogx-ai/ogx/pull/5718))

## Upgrade Guide

### Before Upgrading

These hard breaking changes require updates before you can run 1.0:

1. **Back up storage.** The connectors/batches KVStore migration is one-way.

2. **Replace Safety API usage.**

   ```bash
   grep -r "/v1/safety\|run-shield\|/v1/shields" your-project/
   ```

   Switch to `/v1/moderations`.

3. **Update admin route clients.**

   ```bash
   grep -r "/v1/tools\b" your-project/
   grep -r "/v1/connectors\b\|/v1alpha/connectors\b" your-project/
   ```

   - `/v1/tools` → `/v1/admin/tools`
   - `/v1/connectors` → `/v1alpha/admin/connectors`

4. **Configure tenancy.** Single-tenant deployments must opt into the default tenant or provide tenant credentials. Review provider configs that touch storage.

5. **Update `logprobs` callers on Completions.**

   ```bash
   grep -rn "logprobs.*=\s*\(True\|False\)" your-project/
   ```

   Replace booleans with integers (or omit the field).

6. **Update `/v1/models` parsers** to the new multi-SDK response shape.

7. **Update `ogx_api` imports** to use `ogx_api.types` or `ogx_api.provider` as appropriate. Transitional shims are in place for 1.0 but will be removed in 2.0.

### After Upgrading

- Custom provider authors: migrate any direct `SqlStore` usage to `AuthorizedSqlStore` for access-controlled tables.
- Review auth/rate-limit configuration in light of the new gateway-first architecture.
- Verify multi-tenant data scoping with the new vector-store metadata isolation if you have shared deployments.

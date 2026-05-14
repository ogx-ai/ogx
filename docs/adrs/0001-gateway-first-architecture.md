# ADR-0001: Gateway-First Architecture

- **Status:** Accepted
- **Date:** 2025-05-06
- **Authors:** Sebastien Han

## Context

OGX is an OpenAI-compatible agentic API server with pluggable inference
backends. As the project moves toward production multi-tenant deployments,
we need to define what OGX is responsible for versus what belongs in the
infrastructure layer (gateway, service mesh, load balancer).

The server previously included middleware for TLS termination, CORS,
request quota/rate limiting, and application-level authentication and
authorization. Some of these duplicate concerns that purpose-built gateways
(Envoy, Solo, Kong) handle better.

## Decision

OGX adopts a gateway-first architecture. Edge concerns are offloaded to the
gateway. Resource authorization and tenant isolation stay in OGX because
they require application state.

### What the gateway handles

- **TLS termination at the edge.** Handled by the gateway or load balancer
  in front of OGX. OGX retains server-level TLS config for east-west
  encryption (mTLS between gateway and backend) and non-gateway
  deployments.

- **Edge rate limiting and IP allowlisting.** The gateway has better
  visibility into client identity at the network layer and can enforce these
  policies with purpose-built tooling.

- **CORS.** A browser-facing concern handled at the gateway or CDN layer.

- **Circuit breaking and load balancing.** Infrastructure-level concerns
  that belong in the service mesh or gateway.

- **Request routing across OGX instances.** The gateway or service mesh
  handles this.

- **Authentication (production).** The gateway validates tokens against
  the IdP and injects identity headers (tenant_id, user_id, claims)
  before forwarding to OGX. The gateway must strip any client-supplied
  identity headers to prevent spoofing. Direct auth providers (OAuth2,
  Kubernetes, custom, GitHub) remain supported for development and
  single-tenant deployments where a gateway is optional.

### What OGX handles

- **Identity and tenant extraction.** OGX extracts principal, tenant_id,
  and user attributes from the auth result. In gateway mode, this comes
  from injected headers via `UpstreamHeaderAuthProvider`. In direct auth
  mode, from JWT claims, Kubernetes tokens, or custom auth responses.

- **Resource authorization (ABAC).** OGX matches user attributes against
  resource-level `access_attributes` and checks resource ownership. This
  covers models, vector stores, files, shields, and tool groups.

- **Tenant isolation on stored resources.** Stateful resources
  (conversations, response items, files, vector store entries, registry
  entries) are tagged with tenant_id and user_id. All queries filter by
  tenant context.

- **Semantic request metrics.** Application-level metrics (per-API method
  latency, concurrent requests by API) that the gateway cannot produce.

- **Request context propagation.** Provider data, test context, and client
  version checks.

### Why resource authorization cannot move to the gateway

Consider a `POST /v1/responses` call that references a model, attached
files, a vector store for retrieval, and a previous conversation:

```http
POST /v1/responses
Authorization: Bearer <token>

{
  "model": "llama3",
  "input": "Summarize the Q3 report",
  "tools": [{"type": "file_search", "vector_store_ids": ["vs_abc123"]}],
  "previous_response_id": "resp_xyz789"
}
```

The gateway can validate the token and inject identity headers. It can even
see "llama3" in the request body. But it cannot verify:

- Whether `vs_abc123` belongs to this user's tenant in OGX's storage
- Whether `resp_xyz789` is a conversation this user owns
- Whether the files attached to the vector store are accessible to this user
- Whether the model's `access_attributes` match this user's claims

All of those checks require querying OGX's database. The gateway would need
to make callouts to OGX for every resource reference in every request, which
defeats the purpose of having a gateway.

### Trust boundary

The following are hard deployment invariants, not soft guidance:

- **Gateway-only reachability.** OGX must not be directly reachable by
  clients when using `upstream_header` auth. The
  `UpstreamHeaderAuthProvider` trusts gateway-injected headers without
  validation. If clients can reach OGX directly, they can forge identity
  headers. Network policy or firewall rules must enforce this.

- **Header stripping.** The gateway must strip any client-supplied
  identity headers (principal, tenant, attribute headers) before
  forwarding to OGX. If the gateway does not strip these headers, a
  client can inject arbitrary identity claims.

- **Tenant context is auth-resolved only.** Tenant partitioning always
  uses the tenant_id resolved by the auth provider from trusted sources
  (gateway headers, JWT claims, auth endpoint response). It must never
  come from client-controlled request body fields, search filters, or
  `X-OGX-Provider-Data` headers. The passthrough provider infrastructure
  already forwards arbitrary provider-data keys downstream, so these
  namespaces must remain strictly separated.

## Consequences

### Middleware removed

As a direct result of this decision, the following middleware and
configuration were removed from OGX:

- **QuotaMiddleware** and `QuotaConfig` -- gateway handles rate limiting
- **CORSMiddleware** and `CORSConfig` -- gateway handles CORS

Server-level TLS config (`tls_certfile`, `tls_keyfile`, `tls_cafile` on
`ServerConfig`) is retained for east-west encryption and non-gateway
deployments. Auth provider TLS fields (for outbound connections to auth
endpoints) are also unaffected.

### Middleware retained

- **AuthenticationMiddleware** -- extracts identity from tokens or
  gateway-injected headers
- **RouteAuthorizationMiddleware** -- enforces route-level access policies
- **RequestMetricsMiddleware** -- semantic per-API metrics
- **ProviderDataMiddleware** -- request context propagation
- **ClientVersionMiddleware** -- protocol compatibility checks
- **ZstdDecompressionMiddleware** -- request body decompression

### Planned: Identity Header Contract

There is no IETF standard for passing authenticated user identity from a
gateway to a backend service. RFC 7239 covers network-level forwarding
(IP, protocol, host) but not user claims. Each gateway has its own
mechanism for extracting JWT claims and injecting them as headers:

- Envoy Gateway: SecurityPolicy `claimToHeaders` mapping
- Istio: `RequestAuthentication` with `claim-to-header`
- Envoy ext_authz: `allowed_upstream_headers` from the auth response
- Kong, Solo, etc.: similar claim-to-header mechanisms

The common pattern is: the gateway extracts individual JWT claims and maps
each one to a named HTTP header. The backend declares which headers it
expects.

**Current limitation.** The `UpstreamHeaderAuthProvider` accepts one
`principal_header` (a plain string) and one optional `attributes_header`
(a JSON-encoded dict). The JSON blob approach forces the gateway to
serialize all claims into a single header value, which most gateway
claim-to-header mechanisms cannot do natively. They map one claim to one
header.

**Planned change.** Replace the single JSON attributes header with
per-attribute header mapping so that each user attribute gets its own
header. This aligns with how gateways actually work:

```yaml
server:
  auth:
    provider_config:
      type: upstream_header
      principal_header: x-user-id
      tenant_header: x-tenant-id
      attribute_headers:
        roles: x-user-roles
        teams: x-user-teams
        namespaces: x-user-namespaces
```

On the gateway side, the claim-to-header config maps directly:

```yaml
# Envoy Gateway SecurityPolicy example
claimToHeaders:
  - claim: sub
    header: x-user-id
  - claim: tenant
    header: x-tenant-id
  - claim: roles
    header: x-user-roles
  - claim: groups
    header: x-user-teams
```

Each header value is either a single string or a comma-separated list.
OGX normalizes each value to `list[str]` (splitting on commas for
multi-valued attributes). No JSON encoding required.

The existing `attributes_header` (JSON blob) should be retained for
backward compatibility but documented as deprecated in favor of the
per-attribute header mapping.

### Planned: Tenant Isolation Model

Multi-tenant isolation does not exist today. The current system provides
ABAC (`access_attributes` matching) and resource ownership
(`owner_principal`) but not tenant scoping.

#### Why ABAC alone is not a tenancy boundary

The current default ABAC policy is: owner OR any overlapping attribute
category OR unowned. This is a sharing mechanism, not an isolation
boundary. If two tenants accidentally share an attribute value (e.g. both
have `teams: ["engineering"]`), their data becomes visible to each other.
Using ABAC attributes as the implicit tenant boundary provides no hard
isolation guarantee.

#### Design: explicit `tenant_id` as the partition key

Treat `tenant_id` like a Kubernetes namespace: it is the hard partition
key, not just another attribute. ABAC operates within a tenant for
fine-grained sharing. One active tenant per request.

**Tenant resolution is auth-provider-agnostic.** Every auth provider can
resolve a tenant, not just upstream headers:

| Auth provider | Tenant source |
|---------------|---------------|
| `upstream_header` | Dedicated `tenant_header` |
| `oauth2_token` | JWT claim (e.g. `tenant` or `org`) via `tenant_claim` config |
| `kubernetes` | Namespace or service account annotation |
| `custom` | Auth endpoint response field |
| `github_token` | Organization |
| None (dev mode) | `default_tenant_id` in server config |

**Tenancy modes:**

- **`disabled`** (default, current behavior) — no tenant enforcement.
  Existing single-tenant and ABAC-only deployments continue unchanged.
- **`single`** — all resources belong to one configured tenant. For
  single-tenant production or dev mode. A `default_tenant_id` is set in
  config and injected into every request. No gateway required.
- **`multi`** — full tenant isolation. Every request must resolve a
  tenant_id. Requests without one are rejected on tenant-scoped routes.

#### Request context

Extend the auth result to include `tenant_id` as a first-class field
alongside `principal` and `attributes`. This becomes part of the request
context propagated through `request_headers.py` and `task.py` for
background work.

```python
# Conceptual — extend User or create a new type
class User(BaseModel):
    principal: str
    tenant_id: str | None = None
    attributes: dict[str, list[str]] | None = None
```

#### Storage enforcement

`tenant_id` is applied **before** ABAC. The query path becomes:

1. Filter by `tenant_id = current_tenant` (hard partition)
2. Then apply ABAC: owner OR matching attributes (intra-tenant sharing)

**SQL-backed storage:**

- Add `tenant_id` column to every tenant-scoped table
- `AuthorizedSqlStore` applies `WHERE tenant_id = ?` before ABAC filters
- On PostgreSQL, add Row-Level Security (RLS) as defense-in-depth after
  the app-level filter is in place

**KV-backed storage:**

- Add a `TenantScopedKVStore` wrapper that prefixes keys with tenant_id
- Applies to: vector store metadata/chunks/file batches
  (`OpenAIVectorStoreMixin`), agent state persistence, and any KV surface
  not yet migrated to SQL

**File storage:**

- Partition physical/object storage paths by tenant, not just metadata

#### Global catalog vs tenant state

Split the resource registry into two layers:

- **Global catalog.** Provider-listed models and distro-defined shared
  resources (`source: listed_from_provider`). Cached process-wide.
  Visible to all tenants.
- **Tenant registry.** User-created resources
  (`source: via_register_api`). Scoped to the creating tenant. Queried
  per-request by current tenant, not preloaded into a global cache.

The current `DistributionRegistry` is a single global KV keyspace with
process-wide caching. In multi-tenant mode, tenant-created entries must
not be cached globally, as that leaks resource existence across tenants.

#### Storage surfaces requiring tenant partitioning

All of the following currently use global keys or lack tenant scoping:

**SQL-backed (via `AuthorizedSqlStore`):**

- Conversations (`openai_conversations` table)
- Conversation messages (`conversation_messages` table)
- Response objects and response items
- File metadata
- Prompts (after migration from KV to SQL)
- Connectors (after migration from KV to SQL)
- Batches (after migration from KV to SQL)

**KV-backed:**

- Distribution registry (`DistributionRegistry` in `registry.py`)
- Vector store metadata, file metadata, chunks, file batches
  (`OpenAIVectorStoreMixin` keyspaces)
- Agent state persistence (inline responses provider)

#### Cross-tenant access

Cross-tenant admin access must not use overlapping attributes (that
defeats the isolation boundary). Instead:

- Explicit tenant switch: admin authenticates, then selects a target
  tenant via a platform role (e.g. `platform_admin`)
- Audit logging for all cross-tenant operations
- Use opaque stable tenant IDs, not human-readable names

#### Migration strategy

**Phase 1: Tenant context and config.**
Add tenancy mode config (`disabled` / `single` / `multi`), tenant
resolution to all auth providers, and `tenant_id` to request context.
Existing deployments default to `disabled` and are unaffected.

**Phase 2: SQL tenant column.**
Add `tenant_id` to conversations, responses, and file metadata tables
(already centralized through `AuthorizedSqlStore`). Backfill existing
rows. In `single` mode, backfill to the configured default tenant. In
`multi` mode, require an operator-supplied mapping. Do not auto-infer
tenants from ABAC attributes.

**Phase 3: Prompts, connectors, batches.**
Complete the KV-to-SQL migration (aligned with PR #5614's direction)
behind tenant-aware storage. These resources move to
`AuthorizedSqlStore` with `tenant_id` from the start.

**Phase 4: Registry and KV surfaces.**
Split the distribution registry into global and tenant halves. Add
`TenantScopedKVStore` for remaining KV surfaces (vector store metadata,
agent state). Stop preloading tenant-created state into process-wide
cache.

**Phase 5: Hardening.**
Enable PostgreSQL RLS as defense-in-depth. Add tenant-scoped audit
logging. Remove the `disabled` tenancy mode for new deployments.

**Backfill rules:**

- Existing rows with empty `owner_principal` are currently treated as
  public. When tenant isolation is enabled, untagged rows must not be
  accessible to any tenant (default deny).
- A migration must tag existing rows with a `tenant_id` before switching
  to `single` or `multi` mode.
- Running with mixed tagged/untagged rows and tenant isolation enabled
  is not supported.

---
slug: ogx-multi-tenant-capabilities
title: "Multi-Tenant AI Infrastructure with OGX: Tenant Isolation, ABAC, and Defense in Depth"
authors: [leseb]
tags: [ogx, multi-tenancy, access-control, security, enterprise]
date: 2026-07-14
---

Most AI gateway projects solve for single-user or single-team setups. Point your SDK at the server, get completions back. That works until your platform team needs to serve multiple tenants from the same OGX instance without them seeing each other's models, vector stores, conversations, or RAG data.

OGX ships two independent isolation layers: a hard **tenant partition key** enforced at the storage layer, and **Attribute-Based Access Control (ABAC)** for fine-grained permissions within a tenant. This post covers how they work, how to configure them, and how they compose.

<!--truncate-->

## The problem: shared infrastructure, isolated data

When multiple teams or customers share an OGX deployment, two things need to be true:

1. **Tenant A cannot see Tenant B's data** — conversations, responses, files, vector store contents. Period. No policy misconfiguration should be able to override this.
2. **Within a tenant, users should only see what they are authorized to see** — Alice's conversations are not Bob's, even if they share a tenant.

These are different problems. The first is a hard isolation boundary. The second is access control. OGX addresses them with separate mechanisms so that a bug or misconfiguration in one does not compromise the other.

## Tenant isolation: the hard partition

The primary isolation boundary in OGX is `tenant_id` — a non-bypassable partition key applied to every row in the storage layer. When tenancy is enabled, the `AuthorizedSqlStore` automatically:

- Adds a `tenant_id` column to every table.
- Stamps every write with the authenticated user's `tenant_id`.
- Applies `WHERE tenant_id = ?` to every read and mutation — before ABAC policy evaluation, before any application logic.
- In `multi` mode, if a request has no `tenant_id`, the query resolves to `WHERE 1=0` — default deny, see nothing.

This is not a policy rule. There is no ABAC condition that can override it. Tenant isolation operates below the access control layer.

### Tenancy modes

OGX supports three deployment modes:

| Mode | Behavior |
|------|----------|
| `disabled` (default) | No tenant column, no filtering. Single-user or single-team setups. |
| `single` | All records are stamped with a configured `default_tenant_id`. Useful for dedicated single-tenant deployments that want a migration path to `multi`. |
| `multi` | Every request must resolve a `tenant_id` from authentication. Requests without a tenant are rejected (401). Records are fully partitioned. |

### Tenant resolution from auth providers

Each authentication provider resolves `tenant_id` from its own source:

| Provider | Config field | Source |
|----------|-------------|--------|
| `oauth2_token` | `tenant_claim` | JWT claim (e.g., `tenant`, `org`, `tid`) |
| `kubernetes` | `tenant_claim` | Kubernetes user claim (resolved via `claims_mapping` path) |
| `custom` | `tenant_field` | Field in the auth endpoint JSON response |
| `upstream_header` | `tenant_header` | HTTP header set by the upstream gateway |

Tenant IDs are validated and normalized: lowercase, alphanumeric with hyphens and underscores, max 128 characters.

### Example: single-tenant deployment

A dedicated deployment where all data belongs to one tenant. No auth required — useful for development or single-customer deployments:

```yaml
server:
  port: 8321
  tenancy:
    mode: "single"
    default_tenant_id: "acme-corp"
```

All records are automatically stamped with `tenant_id: "acme-corp"`. This creates a clean migration path — when you later move to `multi` mode, existing data is already tagged.

### Example: multi-tenant with OAuth2/OIDC

Extract the tenant from a JWT claim:

```yaml
server:
  port: 8321
  tenancy:
    mode: "multi"
  auth:
    provider_config:
      type: oauth2_token
      tenant_claim: "tenant"
      jwks:
        uri: https://auth.example.com/.well-known/jwks.json
      issuer: https://auth.example.com
      audience: ogx
```

A decoded token with `{"sub": "alice", "tenant": "acme-corp", ...}` resolves `tenant_id: "acme-corp"`. Any claim name works — set `tenant_claim` to match your IdP's token structure.

### Example: multi-tenant with upstream gateway

The most common production pattern. An upstream gateway (Authorino, Istio, or a reverse proxy) handles authentication and injects identity headers:

```yaml
server:
  port: 8321
  tenancy:
    mode: "multi"
  auth:
    provider_config:
      type: upstream_header
      principal_header: "x-auth-user-id"
      tenant_header: "x-tenant-id"
      attributes_header: "x-auth-attributes"
```

The gateway sets `x-tenant-id` on every request. If the header is missing, OGX rejects with a 401.

## Authentication: bring your identity provider

OGX does not invent its own auth system. It integrates with what you already run:

- **OAuth2/OIDC (JWT)**: Validates tokens via JWKS or RFC 7662 introspection. Maps claims to tenant identity and access attributes. Works with Keycloak, Auth0, Okta, or any OIDC provider.
- **Kubernetes**: Validates service account tokens against the cluster API server. Extracts username, groups, and tenant from Kubernetes user info.
- **GitHub**: Validates personal access tokens or OAuth tokens directly against the GitHub API. Pulls login, user ID, and org memberships.
- **Custom endpoint**: POST your bearer token to any HTTP endpoint. The endpoint returns a principal, attributes, and optionally a tenant ID.
- **Upstream headers**: Trust headers injected by a gateway (Authorino, Istio, Envoy). No outbound validation calls.

The key output of authentication is a **User** object with a principal (identity), an optional `tenant_id`, and **access attributes** across four categories: `roles`, `teams`, `projects`, and `namespaces`. The `tenant_id` drives partition isolation. The attributes drive ABAC policy evaluation within that partition.

## Attribute-Based Access Control

ABAC governs what individual users can do **within their tenant**. It is the second isolation layer — it operates on the data that tenant isolation has already scoped.

OGX implements ABAC with a policy language inspired by [Cedar](https://www.cedarpolicy.com/). Policies are ordered lists of rules. The first matching rule wins.

Each rule has:

- A **scope**: `permit` or `forbid`, with the actions it covers (`create`, `read`, `update`, `delete`).
- A **resource pattern**: exact match (`model::llama-3`), wildcard (`vector_store::*`), or regex.
- A **condition**: natural-language-style predicates evaluated against the user and resource.

### Supported conditions

| Condition | Meaning |
|-----------|---------|
| `user is owner` | The authenticated user created the resource |
| `user is not owner` | Someone else created the resource |
| `user in owners teams` | User shares at least one team with the resource creator |
| `user not in owners teams` | User shares no teams with the resource creator |
| `user with admin in roles` | User has the value `admin` in their `roles` attribute |
| `user with admin not in roles` | User does not have `admin` in `roles` |
| `resource is unowned` | The resource has no owner (public) |

Conditions compose with `when` (permit if condition holds) and `unless` (permit unless condition holds).

Owner-based conditions (`user is owner`, `user in owners teams`, etc.) automatically deny cross-tenant access even if attribute values happen to match. Two users with `roles: ["admin"]` in different tenants cannot see each other's resources through an `admin`-based policy.

### Example: owner-only records within a tenant

```yaml
server:
  auth:
    access_policy:
      - permit:
          actions: [read]
          resource: "model::*"
        when: "resource is unowned"
        description: "all authenticated users can use globally registered models"
      - permit:
          actions: [create]
          resource: "sql_record::*"
        description: "any authenticated user can create records"
      - permit:
          actions: [read, update, delete]
          resource: "sql_record::*"
        when: "user is owner"
        description: "users can only access records they created"
```

With tenant isolation handling the cross-tenant boundary, ABAC focuses on within-tenant sharing. The `create` rule is unconditional because the ABAC check runs before ownership is assigned.

### Default policy

If you configure authentication but do not specify a custom `access_policy`, OGX applies a sensible default: users can access their own resources and resources created by users who share at least one attribute value in any category. This means team members naturally see each other's work without explicit rules.

## Route-level access control

Before resource-level ABAC evaluates, OGX supports **route policies** that control which API endpoints a user can reach at all:

```yaml
server:
  auth:
    route_policy:
      - permit:
          paths: "/v1/chat/completions"
        when: "user with developer in roles"
        description: "developers can access chat completions"
      - permit:
          paths: "/v1/files*"
        when: "user with data_engineer in roles"
        description: "data engineers can manage files"
```

Route patterns support exact paths, prefix wildcards (`/v1/files*`), and regex. This gives platform teams infrastructure-level gatekeeping before any resource checks run.

## Vector search isolation

Two tenants might store documents about the same topic. A naive vector search returns the most similar chunks regardless of who owns them. Tenant A asks "What is our revenue forecast?" and gets Tenant B's financials.

OGX addresses this at multiple levels:

### Tenant-partitioned storage

With tenancy enabled, vector store data is partitioned by `tenant_id` at the storage layer — the same `WHERE tenant_id = ?` filter that protects conversations and responses also applies to vector store queries. Tenants cannot see each other's chunks regardless of semantic similarity.

### ABAC on vector store resources

Vector store resources are subject to the same ABAC policies as every other resource. The routing table enforces access before any search runs.

### Per-tenant vector store indexes

For physical isolation, create separate vector stores per tenant. Each tenant's data lives in its own index, scoped by both tenant partition and ownership.

## Fair scheduling across tenants

When multiple tenants share an inference backend, one tenant's burst of requests should not starve the others. OGX's vLLM provider supports this natively: the `fairness_header_attribute` config option maps an authenticated user's ABAC attribute (e.g., `namespaces` or `teams`) to the `x-gateway-inference-fairness-id` header on outgoing requests. vLLM uses this header to schedule requests fairly across tenants.

Because the fairness identifier comes from the auth-resolved user attributes — not from client-supplied headers — tenants cannot spoof their identity to game scheduling priority.

## Putting it together: a complete multi-tenant configuration

```yaml
server:
  port: 8321
  tenancy:
    mode: "multi"
  auth:
    provider_config:
      type: oauth2_token
      tenant_claim: "tenant"
      issuer: https://auth.example.com
      audience: ogx
      jwks:
        uri: https://auth.example.com/.well-known/jwks.json
      claims_mapping:
        groups: teams
        resource_access.ogx.roles: roles

    access_policy:
      - permit:
          actions: [read]
          resource: "model::*"
        when: "resource is unowned"
        description: "all authenticated users can use globally listed models"
      - permit:
          actions: [create]
          resource: "sql_record::*"
        description: "any authenticated user can create records"
      - permit:
          actions: [read, update, delete]
          resource: "sql_record::*"
        when: "user is owner"
        description: "users can only access their own records"

    route_policy:
      - permit:
          paths: "/v1/chat/completions"
        description: "all authenticated users can use chat"
      - permit:
          paths: "/v1/responses*"
        description: "all authenticated users can use responses"
      - permit:
          paths: "/v1/vector_stores*"
        when: "user with data_admin in roles"
        description: "only data admins manage vector stores"
```

From the application side, nothing changes. Clients still use the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://ogx.internal:8321/v1", api_key="<tenant-jwt-token>")

response = client.chat.completions.create(
    model="llama-3-70b",
    messages=[{"role": "user", "content": "Summarize Q2 results"}],
)
```

The JWT carries the tenant identity. OGX validates it, extracts the `tenant_id` and attributes, and enforces isolation at every layer.

## Defense in depth

OGX's multi-tenant security is not a single gate. It is independent layers that each enforce without relying on the others:

1. **Tenant partition**: `WHERE tenant_id = ?` on every query. Non-bypassable. No policy rule can override it.
2. **Route-level**: Can this user reach this API endpoint at all?
3. **Resource-level**: Can this user access this specific model, vector store, or tool group?
4. **Record-level**: Can this user see this specific conversation or response within their tenant?

A misconfigured ABAC policy cannot leak data across tenants. A missing route policy cannot expose another tenant's records. The layers are independent.

## What this means for platform teams

If you are building a shared AI platform, OGX gives you:

- **One deployment, many tenants**: No need to run separate OGX instances per team or customer.
- **Hard isolation by default**: `tenant_id` partitioning is enforced at the storage layer, independent of access policies.
- **Standard identity integration**: Plug in your existing OIDC provider, Kubernetes auth, or service mesh. Tenant resolution is built into every auth provider.
- **Declarative policies**: Define within-tenant access rules in YAML, not application code.
- **No application changes**: Tenants use standard OpenAI-compatible clients. Isolation is infrastructure-level.
- **Migration path**: Start with `single` mode, move to `multi` when ready. Existing data is already tagged.

Multi-tenancy is not an add-on. It is built into OGX's storage engine, routing tables, and API middleware. For teams moving from single-tenant prototypes to shared production infrastructure, the path is configuration, not code.

---

To get started, check the [multi-tenancy configuration docs](../docs/distributions/configuration) and [PR #6126](https://github.com/ogx-ai/ogx/pull/6126) for the implementation details.

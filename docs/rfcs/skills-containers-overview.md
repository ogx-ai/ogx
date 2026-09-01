# Skills, Containers, and Code Execution in OGX

**Authors:** Francisco Javier Arceo, Varsha Prasad Narsing
**Status:** Draft | **Date:** 2026-05-14
**Implementation reference:** [`skills-containers-implementation-ref.md`](skills-containers-implementation-ref.md) (data models, protocols, file structure)

## What and why

OGX can't run code in the OGX server today. When a model wants to execute a shell command or run Python, there's no sandbox to run it in. OpenAI solves this with their [Containers API](https://developers.openai.com/api/reference/resources/containers) and [Shell tool](https://developers.openai.com/api/docs/guides/tools-shell) — we need an equivalent that works on self-hosted infrastructure.

We're adding three things:

1. **[Skills API](https://developers.openai.com/api/docs/guides/tools-skills)** — Upload versioned code bundles (zip + `SKILL.md` manifest). Standard CRUD, OpenAI-compatible. Also conforms to the [Agent Skills standard](https://agentskills.io).
2. **[Containers API](https://developers.openai.com/api/reference/resources/containers)** — Create sandboxed Debian environments, upload files, manage lifecycle. Standard CRUD, OpenAI-compatible.
3. **ContainerRuntime provider** — The abstraction that actually runs containers. Three implementations (see below).

These are separate API surfaces but one feature: skills are code bundles, containers are where they run, and the Responses API ties them together. A skill without a container is just file storage — to actually execute skill code via the `shell` tool, you need both.

All endpoints launch at `/v1alpha`.

## Architecture

OGX is an API compatibility layer, not an infrastructure orchestrator. All three `ContainerRuntime` providers are `remote::` — OGX acts as a client to an external daemon or API, consistent with how `remote::ollama` and `remote::vllm` work even when the daemon runs locally:

| Provider | OGX talks to | Sandbox lifecycle owned by | When to use it |
|----------|-------------|---------------------------|----------------|
| `remote::reference` | [Docker Engine API](https://docs.docker.com/reference/api/engine/) | OGX (via Docker/Podman daemon) | Simplest path. Docker or Podman on your machine, basic container isolation. Dev, CI, local demos. |
| `remote::openshell` | [OpenShell Gateway API](https://github.com/NVIDIA/OpenShell) | OpenShell Gateway | Policy-enforced isolation with pluggable compute drivers (container, MicroVM, or K8s). |
| `remote::kubernetes` | Kubernetes API | K8s cluster | Direct pod submission. OGX sets `sandbox_required: true`; the cluster enforces isolation via native mechanisms (RuntimeClass, NetworkPolicy, etc.). |

The key distinction between `remote::openshell` on K8s and `remote::kubernetes` is **who owns sandbox policy**. With `remote::openshell`, OGX controls policy through the Gateway. With `remote::kubernetes`, OGX submits pods and the cluster handles everything.

```text
┌──────────────────────────────────────────────────────────┐
│                      Client / Agent                      │
└──────────────────────┬───────────────────────────────────┘
                       │ POST /v1alpha/responses
                       │ (tools: [{ type: "shell", ... }])
                       ▼
┌──────────────────────────────────────────────────────────┐
│                         OGX                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Skills   │  │  Containers  │  │ Responses         │  │
│  │ API      │  │  API         │  │ (shell tool)      │  │
│  └──────────┘  └──────┬───────┘  └────────┬──────────┘  │
│                       │ delegates          │ executes    │
│                       ▼                    ▼             │
│              ┌──────────────────────────────────┐       │
│              │        ContainerRuntime           │       │
│              └──┬──────────┬────────────┬────────┘       │
└─────────────────┼──────────┼────────────┼────────────────┘
                  │          │            │
                  ▼          ▼            ▼
       ┌────────────┐  ┌─────────────┐  ┌──────────────┐
       │  Docker /  │  │  OpenShell  │  │  Kubernetes  │
       │  Podman    │  │  Gateway    │  │  API         │
       │  Engine    │  │             │  │              │
       │  API       │  │             │  │              │
       └────────────┘  └─────────────┘  └──────────────┘
```

## How it works in the Responses API

When a client includes a `shell` tool in a request, three environment modes:

- **`container_auto`** — OGX spins up an ephemeral container, runs commands, destroys it when done. Zero setup.
- **`container_reference`** — Client pre-creates a container via `POST /containers`, references it across requests. State persists between turns.
- **`local`** — Model generates commands, streams them to the client. Client executes locally and sends output back. No server-side execution.

`code_interpreter` uses the same container substrate — it's shell execution with language-specific wrappers.

## Security defaults

Matches [OpenAI's posture](https://developers.openai.com/api/docs/guides/tools-shell): network disabled by default, no root, no TTY, ephemeral by default. Operators can configure a `NetworkPolicy` with type `allowlist` to permit outbound access to specific domains (e.g., internet access while isolating from host infrastructure). Per-request policy can restrict further but never expand beyond the operator ceiling.

## Rollout

| Phase | What |
|-------|------|
| 1 | Skills + Containers APIs, `remote::reference` provider, shell tool integration, OpenAI wire compatibility validation |
| 2 | `remote::kubernetes` provider |
| 3 | `remote::openshell` provider |
| 4 | `code_interpreter` unification |
| 5 | API graduation (`/v1alpha` → `/v1beta` → `/v1`) |

## Open questions

1. **Custom container images** — Should operators bring their own base images? Flexibility vs. attack surface.
2. **Skill sharing** — Do we need a registry/marketplace, or is CRUD enough for v1alpha?
3. **GPU access** — Add a `gpu` field to container creation for ML-heavy skills?
4. **Container-to-inference networking** — When a skill calls an LLM, loopback through OGX or direct to provider? Likely a deployment concern.
5. **Multi-tenant isolation** — API-key-level for now. K8s can layer namespaces + RBAC.
6. **Wire compatibility** — Models based on docs, not tested against live API. Phase 1 validates.

## References

- [Implementation reference (data models, protocols, file structure)](skills-containers-implementation-ref.md)
- [OpenAI Skills API](https://developers.openai.com/api/docs/guides/tools-skills) | [Containers API](https://developers.openai.com/api/reference/resources/containers) | [Shell Tool](https://developers.openai.com/api/docs/guides/tools-shell) | [Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter#containers)
- [Docker Engine API](https://docs.docker.com/reference/api/engine/) | [OpenShell](https://github.com/NVIDIA/OpenShell) | [Agent Skills Standard](https://agentskills.io)

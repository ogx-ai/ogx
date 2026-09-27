# Job Execution Substrate

Runs provider work in a **separate process** instead of the server's event loop, and
makes that work **durable** so it survives restarts. Providers opt in with
`execution_mode: worker` on their `InlineProviderSpec`; today only the
`file_processors` API uses it (large/slow file parsing must not block the server).

## How it fits together

```text
server process                         worker process(es)
--------------                         ------------------
FileProcessorJobProxy  --enqueue-->  ┌──────────────┐  --lease-->  real provider impl
 (JobBackedProxy)        [ jobs ]    │  JobQueue    │              (rebuilt from a
                          table  <---│  (queue.py)  │<--complete-- ProviderDescriptor)
  process_file()  <--poll result---  └──────────────┘                (worker.py)
```

- **`queue.py` — `JobQueue`**: a durable queue backed by a SQL store. The queue table
  *is* the IPC channel between the server and its workers. Leasing is an atomic guarded
  `UPDATE` so two workers never run the same job; expired leases are reclaimed (a crashed
  worker's job is retried without exceeding its attempt budget). Job control queries are
  scoped by API, provider, principal, and tenant. Composite indexes support leasing and
  cursor pagination, and periodic maintenance expires terminal rows after seven days.
- **`worker.py` — `WorkerPool` + worker loop**: spawns OS processes (spawn context, so a
  fresh interpreter and its own GIL). Each worker rebuilds the real provider impl — and its
  direct API dependencies — from a `ProviderDescriptor`, then leases → executes → reports.
  Descriptors preserve secret config values and access policies, sibling providers are wired
  after reconstruction, and the enqueue-time authenticated identity is restored for each
  invocation and terminal cleanup. The parent supervises crashed workers, restarts them with
  bounded backoff, and exposes pool failure through the stack health endpoints.
- **`proxy.py` — `JobBackedProxy`**: the API-agnostic proxy the server mounts in place of a
  worker-mode provider. It only enqueues and reads job state (`_enqueue`, `_run_blocking`,
  `_get`, `_cancel`, `_list`). APIs register a proxy factory via `register_worker_proxy`;
  the resolver looks one up by API (`WORKER_PROXY_FACTORIES`) — nothing here is per-API.
- **`file_processor_proxy.py` — `FileProcessorJobProxy`**: the thin, file-processor-specific
  adapter. It keeps `process_file` compatible with the stable `FileProcessors` provider
  protocol while exposing job lifecycle methods as an optional server-side capability. It
  stages direct uploads into Files and deletes those temporary files when jobs become terminal.
  Cleanup itself is durably claimed and leased, so another maintenance pass retries it after
  a worker crash or transient storage failure. Job listings are lightweight, owner-scoped,
  and cursor-paginated.
- **`dispatch.py`**: per-`(api, method)` (de)serialization of payloads and results. Add a
  new worker-backed method by adding one entry.
- **`models.py`**: `JobRecord` (persisted form) and `ProviderDescriptor` (how a worker
  rebuilds an impl). The public job shape is `ogx_api.common.job_types.Job`.
- **`runtime.py` / `bootstrap.py`**: process-global handle to the queue + pool, and the
  stack-side construction of them (owned by `Stack`, started after provider resolution,
  shut down on stack shutdown).

## Data plane

Job payloads never carry file bytes. Direct uploads are staged into the Files API and the
job carries only a `file_id`; the worker reads the bytes back through its own (rebuilt)
Files provider. The queue stays small and the same shared storage backs both processes.

## Notes

- The queue uses `get_system_sqlstore()` (a plain, non-authorized store) because it is
  internal infrastructure shared across processes. The job layer persists the trusted
  authenticated identity and applies explicit API/provider/principal/tenant predicates to
  every user-facing read or mutation; workers use unscoped operations only for leasing and
  terminal state transitions.
- Because workers talk to the shared DB rather than the server directly, moving workers
  off-box later is an extension of this boundary, not a rewrite.
- Auto-routed APIs cannot yet be reconstructed inside workers. Optional auto-routed
  dependencies are omitted; a Docling worker configured with `vlm_model` fails startup
  clearly instead of silently disabling VLM processing. Run that provider inline until
  worker-side inference routing is supported.

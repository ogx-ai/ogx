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
  worker's job is retried). `reclaim_stale()` runs at startup to recover jobs left
  in-progress by a previous run.
- **`worker.py` — `WorkerPool` + worker loop**: spawns OS processes (spawn context, so a
  fresh interpreter and its own GIL). Each worker rebuilds the real provider impl — and its
  direct API dependencies — from a `ProviderDescriptor`, then leases → executes → reports.
- **`proxy.py` — `JobBackedProxy`**: the API-agnostic proxy the server mounts in place of a
  worker-mode provider. It only enqueues and reads job state (`_enqueue`, `_run_blocking`,
  `_get`, `_cancel`, `_list`). APIs register a proxy factory via `register_worker_proxy`;
  the resolver looks one up by API (`WORKER_PROXY_FACTORIES`) — nothing here is per-API.
- **`file_processor_proxy.py` — `FileProcessorJobProxy`**: the thin, file-processor-specific
  adapter. It maps the `FileProcessors` protocol onto `JobBackedProxy`: the deprecated blocking
  `process_file` waits; the async `*_process_file_job` methods return a handle; and it stages
  direct uploads into Files. This is the template another API copies to gain worker support.
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
  internal infrastructure shared across processes with no per-request owner — unlike
  user-facing tables, which must use `authorized_sqlstore()`.
- Because workers talk to the shared DB rather than the server directly, moving workers
  off-box later is an extension of this boundary, not a rewrite.

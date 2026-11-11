# Cluster management API

Applications supply approved worker identities and endpoints to the existing
MLX-VLM server. The server owns registration validation, inference coordination,
weight preparation, lifecycle state, capabilities, and telemetry. Discovery,
macOS permissions, consent, pairing, device selection, and visualization remain
application responsibilities. This API does not browse Bonjour, install apps,
configure Remote Login, or enroll SSH keys.

Inference stays on `/v1/responses` and `/v1/chat/completions` with
`distributed: true`. The cluster API is a control plane; weight bytes use the
existing **SSH/rsync transport**, not HTTP uploads.

## Authentication and identity

Set `MLX_VLM_SERVER_API_KEY` on the public server. Every cluster endpoint requires
its Bearer token; cluster management is disabled when no key is configured.
Each worker also runs MLX-VLM with an API key. Register its own credential in
`api_key`; this authenticates identity, health, inference, and session control.
The coordinator does not return that credential in node responses or events.
The private registry persists it in a mode-0600 file under
`MLX_VLM_CLUSTER_STATE_DIR` (default `~/.cache/mlx-vlm/cluster`). Protect this
state directory and use TLS or a trusted private transport for HTTP endpoints.

Worker API authentication is separate from SSH authentication. For remote
weights, `ssh.destination` names an **already accessible** SSH host/alias and
`ssh.python` selects that machine's Python for checksum verification. SSH uses
existing keys, known hosts, and configuration. The API never enables macOS SSH,
accepts host keys automatically, or installs credentials. The SSH account needs
write access to the worker's advertised staging directory. SSH/rsync are also
required by the optional existing CLI process launcher; neither capability is
inferred merely from successful API authentication.

Each installation persists a stable `node-id`; alternatively provision
`MLX_VLM_NODE_ID` and retain it across restarts. An address, name, rank, or Python
path is not an identity. Registration authenticates `/internal/distributed/identity`
and verifies both the approved ID and its physical-device fingerprint. Address
updates must present the same identity. Duplicate physical devices/endpoints
are rejected. The public coordinator and a rank-zero process can share one Mac;
selected memory is counted once per physical device.

## Endpoints

All schemas are in `/openapi.json`. Management operations are serialized.

| Endpoint | Behavior |
|---|---|
| `GET /v1/cluster` | State, public coordinator identity, rank-zero inference coordinator ID, selected/running members, active model, ownership, capabilities, aggregate memory |
| `GET /v1/cluster/nodes` | Registered nodes with authenticated identity, reachability, timestamped measurements and freshness |
| `POST /v1/cluster/nodes` | Verify and register an approved worker; identical registration is a no-op |
| `PATCH /v1/cluster/nodes/{node_id}` | Change endpoint, credential, name, or preconfigured SSH connection while retaining identity |
| `DELETE /v1/cluster/nodes/{node_id}` | Remove an idle registration; absent nodes are a no-op |
| `PUT /v1/cluster/members` | Select an ordered membership and backend; optional model/revision enables sharding preflight |
| `POST /v1/cluster/models/prepare` | Download/reuse and deliver verified weights to selected or explicitly named registered workers; does not launch or load a model |
| `POST /v1/cluster/start` | Attach a compatible existing collective or launch with the configured SSH launcher; optional model loads before readiness |
| `POST /v1/cluster/stop` | Drain, dispose retained sessions if explicitly requested, stop owned processes or detach external processes |
| `GET /v1/cluster/operations/{operation_id}` | Long-operation phase, progress/result, and terminal status/error |
| `GET /v1/cluster/events` | Throttled SSE snapshots, connection/telemetry, membership, and operation updates |

`start`, `stop`, and `models/prepare` return **202** with an `Operation`. Use
`Idempotency-Key` to retry the same operation. A different body with that key
returns `idempotency_conflict`; an overlapping management operation returns
`management_busy`. Poll the returned operation or follow events until
`status` is `succeeded` or `failed`. A 202 response alone does not imply success.
Interrupted operations become explicitly failed after a coordinator restart.

SSE records use `id`, `event`, and a JSON `ClusterEvent` containing
`id`, `type`, `timestamp`, and `data`. Reconnect with `Last-Event-ID`. Events are
bounded in memory (1024); an expired cursor, a future cursor, or a coordinator
restart produces `cluster.snapshot` so the client can reconcile current state.
`interval_seconds` (1–60, default 2) throttles telemetry refreshes per subscription.
Events and operation status contain no worker API secrets.

## Example: register, select, prepare, attach, stream

These examples use Python `httpx` against the coordinator. Discovery and user
approval happen in the application before registration. Obtain `node.id` from
an authenticated identity exchange with an already running worker endpoint;
verify the expected identity through your application's pairing mechanism.
Do not invent node IDs from addresses.

```python
import os
import time
import httpx

client = httpx.Client(
    base_url="http://127.0.0.1:8080",
    headers={"Authorization": f"Bearer {os.environ['PUBLIC_TOKEN']}"},
    timeout=30,
)

for node in approved_workers:
    client.post("/v1/cluster/nodes", json={
        "id": node.id,
        "name": node.name,
        "endpoint": node.endpoint,
        "api_key": node.worker_api_key,
        "ssh": {
            "destination": node.ssh_alias,
            "python": node.python_path,
        },
    }).raise_for_status()

model = "organization/shardable-model"
client.put("/v1/cluster/members", json={
    "node_ids": [node.id for node in approved_workers],
    "backend": "jaccl",
    "model": model,
}).raise_for_status()


def wait_operation(path, body, key):
    response = client.post(path, json=body, headers={"Idempotency-Key": key})
    response.raise_for_status()
    operation = response.json()
    while operation["status"] == "running":
        time.sleep(1)
        response = client.get(f"/v1/cluster/operations/{operation['id']}")
        response.raise_for_status()
        operation = response.json()
    if operation["status"] != "succeeded":
        raise RuntimeError(operation["error"])
    return operation

wait_operation("/v1/cluster/models/prepare", {"model": model}, "prepare-1")
wait_operation("/v1/cluster/start", {"mode": "attach", "model": model}, "start-1")

with client.stream("POST", "/v1/responses", json={
    "model": model,
    "input": "Explain why the sky is blue.",
    "distributed": True,
    "stream": True,
    "resumable": True,
    "max_output_tokens": 256,
}) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        print(line)
```

For a coordinator-local worker, `ssh` can be omitted only when its authenticated
physical identity matches the coordinator. Files are copied locally to that
worker's own staging root and verified identically. For remote workers,
registration can precede SSH configuration, but preparation/start returns
`ssh_transport_required` until the registration supplies working SSH access.
There is no HTTP checkpoint-upload endpoint or separate worker-access service.

`models/prepare` accepts optional `node_ids` (one or more registered nodes), so
weights can be delivered before selecting or starting an inference group. It
reports resolve/hash/copy/verify phases and per-node readiness. It does not
claim bytes transferred or percentage progress that rsync has not measured.
The result includes the content digest and each worker's verified local path.
Weight delivery does not validate a sharding count; selection and inference
startup perform that separate validation. A transfer failure does not fail an
unrelated running inference group. Existing corrupt committed snapshots are
preserved for diagnosis; partial snapshots cannot be loaded.

## Lifecycle and ownership

Registered nodes, selected members, and the running collective are distinct.
An identical member selection is idempotent. Changing running membership or a
running node's connection returns `group_busy`; first stop/drain the group,
then update the selection and start a new group. Never add or remove ranks from
an active collective. Model dimensions and architecture-specific constraints
are checked for the selected count: three devices are not automatically valid
for a model whose attention dimensions divide only by two/four/five.

`start` with `mode: "attach"` requires externally launched workers to present the
same nonempty `MLX_VLM_GROUP_ID`, backend, world size, and rank ordering. Code and
runtime fingerprints must agree. If they need reconfiguration, the failed
operation supplies `external_reconfiguration_required` and the required rank
plan. Applications/process supervisors launch each worker using its own Python
and filesystem paths; MLX-VLM does not install or manage that external process.

`mode: "launch"` reuses the existing `cluster.json` SSH launcher/hostfile. The
selected endpoints must match that launcher configuration; changing its physical
hostfile/topology requires reconfiguration while stopped. It does not synthesize
RDMA wiring from IP addresses. Topology is ultimately validated by initializing
the configured MLX backend and successfully running collectives, not discovery.

`stop` first rejects new distributed admissions, then waits for active requests
to drain, up to `drain_timeout_seconds` (default 60). A timeout is observable and
never silently kills active inference. Retained continuations return
`retained_continuations` unless `discard_continuations: true` was supplied.
A repeated `start` returns a failed operation with `already_running` without
changing the existing group. A rejected or cancelled `stop` before checkpoint
disposal or process shutdown restores the prior readiness/admission state; its
operation can fail while the cluster remains healthy. Once disposal or shutdown
begins, a failure leaves admissions blocked because ranks may have changed.
Explicit disposal synchronizes native checkpoint/cache disposal across ranks.
A failed collective cannot acknowledge cache disposal. Explicit discard then
invalidates coordinator handles, preserves the failure marker, and reports
`collective_disposal_acknowledged: false`; external owners also receive
`external_restart_required: true`. Fresh rank processes are required to recover.
Only processes launched by this server are stopped; externally managed workers
are detached and remain running. Local inference remains independent.

A lost rank/semantic divergence marks the group failed. Replace the entire
affected collective, then start/attach again. Recovery checks fresh process
instance IDs before clearing the failure marker; a healthy HTTP endpoint alone
does not prove a broken collective recovered. Restarting the coordinator keeps
registrations/selection but requires explicit reattachment to external workers.

## Continuations, progress, and capabilities

Distributed Responses use the existing `resumable`/`continuation` contract. Save
the latest event's continuation object, close the stream, wait for the pause to
finish, and resend the original request with that object in `continuation`.
Tokens already generated are replayed from the native decoder history; retained
KV/recurrent caches resume decoding. Prefill, pause/resume, expiry, disposal,
and cache-budget decisions are synchronized at collective scheduling boundaries.
Concurrent replacement readers return retryable `continuation_busy` rather than
claiming ranks in different orders. Continuations are process-local, tenant and
model scoped; they do not survive group/coordinator restarts.

`capabilities.resumable` is negotiated against the active workers' session
protocol. Streaming/batching and progress support are advertised separately.
Stored Responses retrieval/deletion and `previous_response_id` are **not**
supported for distributed inference; their capability fields remain false.
Reasoning, tool events, compaction capsules, usage, and terminal events retain
the normal Responses contract. MiniMax thinking delimiters are parsed across
arbitrary token boundaries, including closing-only continuations.

The same shared Responses generation path emits:

```json
{"type":"response.prompt_progress","response_id":"resp_…","status":"in_progress","processed_tokens":512,"total_tokens":2048}
{"type":"response.generation_progress","response_id":"resp_…","output_tokens":16,"timings":{"predicted_per_second":32.0}}
```

Prefill uses `started`, `in_progress`, and `completed` with monotonic, measured
counts, emitted while actual prefill runs and before the first generated token.
Generation timing fields appear only when measured. `timings.predicted_per_second`
uses the same field name as regular text events and native generation metrics.
The coordinator forwards rank zero's measurements. Timing or progress frequency
differences do not count as semantic divergence; token,
reasoning/tool, terminal, and collective decisions still do.

## Telemetry and errors

Per-node system memory reports total/used **bytes** separately from MLX
active/cache/peak allocation. Measurements include timestamps, OS, MLX and
MLX-VLM versions, backend availability, health, and freshness. Missing values
remain null; selected aggregates expose `complete`, `missing_node_ids`, and
`stale_node_ids`. Measurements older than 15 seconds are marked stale. Only
selected physical devices count toward the aggregate, including rank zero once.

Link properties come from `ibv_devinfo` when available. Unknown negotiated rates
remain null. Link-active state is not RDMA proof: `rdma_validated` becomes true
only after this JACCL worker has successfully verified inference collectives.
`coordinator` identifies the public server; `inference_coordinator_id` identifies
the registered rank-zero worker for a client-side “Main” badge.

Management conflicts use `{"detail": {"code", "message", "retryable", "action"}}`.
Failed long operations place the same `ClusterError` under `operation.error`.
Authentication/validation errors also retain standard FastAPI status semantics.

| Code | Client action |
|---|---|
| `identity_mismatch`, `duplicate_device` | Recheck approved identity; do not create a duplicate from a new address |
| `management_busy`, `group_busy` | Wait for the operation, or drain/stop before changing membership |
| `unsupported_member_count`, `backend_unavailable` | Choose a supported model/count/backend |
| `ssh_transport_required` | Supply preconfigured SSH access independently of API credentials |
| `external_reconfiguration_required`, `launch_reconfiguration_required` | Reconfigure the external group or existing hostfile using the returned plan |
| `drain_timeout`, `retained_continuations` | Finish active streams or explicitly discard retained sessions |
| `restart_required`, `transition_failed` | Inspect the operation; replace failed ranks before retrying |
| `coordinator_restarted` | Reconcile workers and retry the interrupted operation with a new key |

## Application migration

| Application responsibility before | Supported server responsibility now |
|---|---|
| Rank request fan-out and stream relaying | Existing inference endpoints with `distributed: true` |
| Coordinated batch admission and cancellation | Shared distributed scheduler and native sessions |
| Parsing worker logs for status/memory | Cluster/nodes/operations endpoints and throttled events |
| Managing rank membership implicitly | Stable registrations, explicit selection and safe group transitions |
| Downloading/synchronizing weights itself | `models/prepare` and verified SSH/rsync staging |
| Resume implementation or regeneration | Existing Responses continuation contract across ranks |

Applications still own discovery, pairing/consent, device UI, and their own
external process supervision. SSH trust/access and backend/network preparation
remain deployment responsibilities. No Nativ fields or native onboarding code
are part of MLX-VLM.

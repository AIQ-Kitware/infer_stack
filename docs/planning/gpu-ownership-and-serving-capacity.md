# GPU ownership and serving capacity

Status: proposed October 8, 2026. Design and implementation sequence;
no new scheduler mode, Compose replication or autoscaler is implemented here.

## Problem and objective

Two client jobs can each reserve two GPUs while acquiring the same compatible
answerer and extractor. infer-stack coalesces their leases onto two engines;
the other two GPUs remain reserved but idle. More clients increase demand on
those engines without increasing serving capacity. A GPU at 100% utilization
does not establish that replication, more batching or tensor parallelism is
the best way to finish the workload sooner.

Reserve physical resources once per serving owner, make sharing explicit, and
add capacity only when measurement justifies it. Start with single-host
Compose, then carry the lifecycle contract into KubeAI. Preserve existing
catalogs, default shared-compatible acquisition and command-scoped cleanup.

This extends the [control-plane plan](distributed-control-plane-and-remote-server.md)
and its [authority inventory](control-plane-authority-inventory.md). It does
not require remote-server/HA work before local progress can be demonstrated.

## Current mechanisms to retain

- `EndpointRequest` separates structural compatibility from capacity;
  leases protect deployments, not an amount of HTTP throughput.
- Compose places new deployments within `allowed_gpus`. Existing live
  deployments can be reused outside a new caller's slice; that slice is an
  admission limit for new placements, not an exclusive client-routing scope.
- Compose has one container per deployment. `--dedicated` already requests
  separate deployments through `acquire`/`access` (or catalog sharing policy);
  `run` has no per-call dedicated flag today. Simultaneous same-name deployments need dynamic
  routing for unique services and upstreams. A shared alias is not a guarantee
  that a client will use only its own dedicated deployment.
- KubeAI already renders `runtime.min_replicas` / `max_replicas` into a Model
  CR, defaulting to 1/1. Kubernetes owns pod placement. Replicated residency
  exists; endpoint generation verification does not prove every replica works.
- SIGTERM unwinds a command lease. Releasing one shared client's lease must
  not destroy engines still protected by another client.

Useful code boundaries: `leasing/models.py`, `ledger.py`, `placement.py`,
`residency.py`, `compose.py`, `gateway.py`, and `backends/kubeai.py`.
CPU-client submission and DAG resource settings belong to scheduler integrations;
infer-stack must not embed a particular workload's model list or graph.

## Resource ownership: choose one mode for each workload

| Mode | Who reserves GPUs? | Client behavior | Intended use |
|---|---|---|---|
| Existing job-scoped mode | Each GPU client allocation | Acquires within its slice; compatible engines may be shared | Compatibility baseline |
| Explicit per-allocation serving | Each GPU allocation owns separate engines | Uses an allocation-scoped upstream set | First controlled Compose replication experiment |
| Shared serving pool | One serving session/allocation owns the pool | CPU clients borrow pool endpoints without duplicating GPU reservations | Repeated tasks and many short client jobs |
| Kubernetes serving | Serving pods request GPUs from Kubernetes | CPU clients access the serving API | Cluster operation |

Do not halve every client's GPU request to hide duplication: a cold-start
answerer plus extractor still needs its full placement footprint. Do not drop
client GRES without a live serving owner. When Slurm owns the physical node,
Compose must remain inside an explicit Slurm allocation; a host daemon must
not silently consume scheduler-owned cards. Never let Slurm and Kubernetes
independently allocate the same physical GPU pool.

A shared pool needs durable owner/allocation identity, expiry/heartbeat,
device scope and a generation fence. Borrowing clients protect serving demand
but cannot extend physical ownership past allocation revocation. On owner
loss: refuse new traffic, mark access unavailable, terminate owned engines and
recover through a new valid allocation. Scheduler completion alone is not
proof Docker engines have stopped. Client retries and operation replay must
not acquire duplicate owners. Use existing ledger/revision authority; do not
introduce another editable ownership file or rely on rendered Compose state.

## Delivery sequence

### 1. Measure and expose the mismatch

Add a joined diagnostic view: scheduler allocation, logical endpoint,
deployment/backend unit, device IDs or pod resource requests, lease count,
ready capacity and active/queued requests. Show allocated, resident and busy
resources separately. Missing metrics remain unknown. On Kubernetes, include
pending replicas and scheduling reasons; pending pods are not usable capacity.

Establish a repeatable benchmark with one versus two clients using identical
request/token distributions, then compare shared engines with two dedicated
engine pairs. Measure completed workload units/hour, token throughput, queue
wait, time to first token, p50/p95 latency, timeout/error rates, startup cost and
GPU-hours. Check response structure and task correctness under the same sampling
settings. Results must include model/runtime revisions and cold/warm-cache state.

For a four-GPU host, the controlled comparison is two shared one-GPU engines
versus two independent copies of each engine. Benchmark both stages: replicating
only an answerer may just move the bottleneck into the shared extractor.

Done when the diagnostic distinguishes two resident GPUs from four reserved
GPUs and the benchmark can attribute capacity changes to completed work.

### 2. Prove explicit ownership with existing Compose primitives

First implement an opt-in scheduler-integration experiment using dedicated
deployments inside disjoint allocations and dynamic routing. Supply stable
allocation-scoped access descriptors/upstream sets when isolation is required;
do not assume a global alias binds each job to its own deployment. Verify all
four units with real generation before admitting benchmark traffic.

Then prove a bounded shared serving session: reserve a GPU pool once, acquire
its engines, and run CPU-only DAG clients against its access descriptors.
Choose a concrete parent-allocation/job-step or serving-job dependency contract
in the scheduler integration. Clients start after readiness, stop on owner
loss, and preserve their own task cache identities. Neither option replaces
existing runners until demonstrated end to end.

Done when concurrent acquire/release/cancel leaves unrelated traffic intact,
engines cannot outlive GPU ownership, and rerunning a cached client acquires
no unnecessary resources. No live workload migration is part of this phase.

### 3. Compose: bounded replica groups

After the experiment establishes a benefit, introduce opt-in replica policy
separate from model/runtime compatibility. A logical deployment can own
multiple uniquely identified backend units, each with its own placement,
readiness, observed generation and upstream route. Keep default replica count
one and migrate residency deliberately: today's strict Compose observation
treats multiple containers for one deployment as a conflict.

Render one explicit service per replica with disjoint `device_ids`; blindly
scaling a service repeats its device reservation. Reconcile only changed units
with stable names, cache mounts and routes. Docker describes device selection
in its [GPU support documentation](https://docs.docker.com/compose/how-tos/gpu-support/).
Device selection alone does not establish exclusive host resource ownership.

New replicas join routing only after real generation succeeds. Scale-down
first removes admission to the selected upstream, drains active requests,
then stops that unit and releases its device claim. A failed or unschedulable
addition preserves existing ready capacity. Account for rolling-update overlap
inside the GPU budget; do not restart survivors to change replica count.

Start with explicit fixed replica counts and dynamic routing. Static routing
must either gain defined replica support or reject this mode clearly. Horizontal
copies leave tensor/pipeline parallelism inside each engine unchanged. Changing
parallelism is a separate measured runtime change, not a reaction to lease count.

### 4. KubeAI: reuse native replica scheduling

Use a Model's existing replica bounds rather than creating a Model per client.
Kubernetes/device-plugin resource requests reserve GPUs for serving pods;
CPU-only clients do not reserve the same GPUs again. For predictable tests,
set equal min/max counts and verify each replica independently. Account for
profile units, tensor/pipeline parallelism and all resident replicas when
reporting total GPU demand. Keep stock backend launch restrictions.

For elastic operation, capability-check the installed KubeAI CRD before adding
policy fields. Upstream documents `targetRequests`, `scaleDownDelaySeconds` and
replica bounds in [autoscaling configuration](https://www.kubeai.org/how-to/configure-autoscaling/),
and least-load/prefix-aware strategies in [load balancing](https://www.kubeai.org/concepts/load-balancing/).
These are upstream capabilities, not a claim that today's infer-stack forwards
every field. Let KubeAI own the replica decision inside declared bounds; do not
run a second infer-stack autoscaler against the same Model's replica field.

Verify pending-pod diagnostics, failed replacement, scale-down/drain behavior,
and lease release with multiple warm replicas. Extend the existing k3s e2e
surface, then perform real GPU checks on the target cluster. A CPU-backed
cluster verifies lifecycle/routing, not GPU capacity or speed.

### 5. Elastic Compose policy after fixed-capacity validation

Only then add opt-in expansion based on sustained queue/latency pressure,
fresh per-unit metrics and remaining owned devices. Bound replica count and
total GPU/VRAM budget; use separate scale-up/down thresholds and cooldowns
that reflect measured model startup time. Allocate spare cards to the pipeline
stage limiting completed work. Fairness across endpoint demand must keep one
busy model from consuming all capacity needed by another.

Lease count is a protection signal, not request pressure. Unknown/stale metrics
must not trigger speculative expansion or eviction. Preserve startup budgets,
warm caches, active requests and explicit dedicated-allocation contracts.

## Validation and rollout

1. Deterministic tests: sharing versus isolation, placement/ownership fences,
   repeated operations, readiness routing, bounded counts, rollback, reference
   counting and owner/client cancellation. Include restart recovery.
2. Real CPU Compose and k3s tests: distinct upstreams, correct model routing,
   traffic surviving replica addition/removal, no engine leak after cancellation.
3. Idle-host four-GPU acceptance: distinct physical assignments, all units
   generating, measured end-to-end throughput and correctness, unrelated work
   preserved, actual resource release after ownership ends.
4. Opt-in downstream DAG run without changing algorithm parameters or cached
   task identity. Record partial completion normally; resource orchestration
   must not introduce a new all-or-nothing result requirement.

Keep the old mode available during rollout. New policy defaults must wait for
the measured workload benefit and lifecycle tests, with explicit rollback to
one shared deployment. API/schema spelling, scale-down request accounting and
the scheduler serving-owner contract remain implementation decisions to settle
in the first experiment, not newly supported CLI promises.

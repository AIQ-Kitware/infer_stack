# Kubernetes service roadmap: from two-node proof to shared accelerator service

**Status:** current implementation plan · 2026-10-07

This document is the current executable plan for moving from the working
physical two-node K3s/KubeAI proof toward the goals in
[northstar.md](northstar.md).

**The north star owns intent. This file owns tactics.** An implementation agent
may revise phase boundaries, data structures, command spellings, or component
choices when it finds a better route, but it should explain any proposed change
that alters the north-star behavior.

This roadmap incorporates, rather than replaces:

- [backend-parity-roadmap.md](backend-parity-roadmap.md);
- [distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md);
- [control-plane-authority-inventory.md](control-plane-authority-inventory.md);
- [vram-aware-placement.md](vram-aware-placement.md).

---

## Current proof point

As of 2026-10-07, the important real-hardware facts are:

- `yardrat` is a K3s server/control-plane node and GPU worker;
- `namek.kitware.com` joined as a physical K3s agent/worker;
- both nodes became Ready and advertised NVIDIA resources through the device
  plugin/GFD stack;
- `namek.kitware.com` advertised one RTX 3090;
- `infer-stack kube node test namek.kitware.com --expected-gpus=1 --apply`
  scheduled a serving replica exactly on `namek`, verified the actual GPU UUID
  and model, and completed a real generation;
- the KubeAI Model CRD/operator path therefore works across two physical
  machines.

This proves the basic data plane and exact-node acceptance path. It does **not**
yet prove a production control/authority architecture.

The full `dev/handover/p5_two_hosts.sh` physical-host integration remains a
useful acceptance check because it additionally exercises the cluster gateway,
cross-node routing, NodePort access, and secret rotation.

### Footguns discovered during the proof

Keep these as regression targets rather than tribal knowledge:

1. **K3s installation can exceed 60 seconds.** The generic subprocess timeout
   killed a healthy worker install. Installer operations need an explicit long
   timeout.
2. **Canonical node names may be FQDNs.** `namek` registered as
   `namek.kitware.com`; exact-node operations must use/discover the canonical
   Kubernetes name.
3. **K3s's bundled `kubectl` may prefer `/etc/rancher/k3s/k3s.yaml`.** A
   normal `~/.kube/config` symlink was not enough on the server host. infer-stack
   needs a deliberate persisted cluster/kubeconfig target for its own commands;
   it should not silently weaken the root-owned K3s admin kubeconfig.
4. **Container image storage competes with kubelet ephemeral storage.** On
   `namek`, old Docker images/build cache consumed the root filesystem while
   K3s/containerd also used it. Pulling the ~14 GB vLLM image crossed K3s's
   configured `imagefs.available/nodefs.available` eviction threshold.
5. **Recovery has hysteresis.** K3s had `evictionHard=5%`,
   `evictionMinimumReclaim=10%`, and a 5-minute pressure transition period, so
   DiskPressure did not clear immediately after freeing space.
6. **A Pending pod can still reserve a GPU.** A stale model pod already bound
   to `namek` held its `nvidia.com/gpu: 1` allocation and made the next
   acceptance probe Unschedulable.
7. **Failed/evicted acceptance pods need first-class diagnostics and cleanup.**
   A serving pod evicted for ephemeral storage regressed to a misleading
   `waiting for scheduler`; cleanup later raced a slow pod deletion.

Do not solve these by documenting commands only. Where infer-stack can detect or
prevent the condition, the program should do so.

---

## Vocabulary and responsibility boundaries

Use these terms consistently in code, docs, and agent discussions.

### Kubernetes control plane

K3s/Kubernetes API servers and datastore. Owns Kubernetes node membership and
scheduling mechanics.

### infer-stack control plane / authority

Owns leases, desired endpoint/deployment state, catalog policy, operations,
user-visible status, and reconciliation. There must be one logical authority
per managed backend domain/namespace.

### Data plane

KubeAI model servers, vLLM/Ollama runtimes, LiteLLM/front door, and related
services that answer inference traffic.

### Cluster administrator

May admit/remove nodes and change cluster infrastructure/policy.

### Lease user

May request approved endpoints subject to policy. Initially this may be the same
human/credential as the administrator, but keep the domain distinction.

### Lease vs deployment vs replica

- **lease:** demand from a user/client;
- **deployment:** one compatible logical serving realization shared by leases;
- **replica:** one serving process/pod contributing capacity to a deployment.

Never implement replica count as `number of leases`.

---

## Target shape for the next useful system

The next target is deliberately simpler than full HA:

```text
                        user / agent
                         CLI / TUI
                             |
                   infer-stack authority
                   (one logical owner)
                             |
                       Kubernetes API
                             |
          +------------------+------------------+
          |                  |                  |
       worker             worker             worker
      yardrat            namek             4x A6000
          |                  |                  |
          +------- KubeAI Models / replicas ---+
                             |
                         stable gateway
```

Near term, the infer-stack authority may remain on `yardrat`. The architecture
must stop treating that colocation as permanent.

---

# Phase 0 — preserve and finish the physical two-node proof

**Goal:** turn the successful experiment into a repeatable regression boundary
before layering more features on top.

### Tasks

- Run and preserve the result of `dev/handover/p5_two_hosts.sh` across
  `yardrat` and `namek.kitware.com`.
- Keep the worker acceptance test able to identify:
  - scheduling failure reason;
  - failed/evicted pod reason/message;
  - actual node placement;
  - actual GPU UUID/model;
  - serving readiness;
  - real generation;
  - cleanup state.
- Ensure acceptance cleanup is idempotent and a failed prior run cannot silently
  consume GPU allocation for the next run.
- Add deterministic tests for:
  - pod eviction after it has already run;
  - DiskPressure / untolerated taint;
  - assigned Pending pod holding GPU capacity;
  - cleanup timeout where resources disappear at the timeout boundary.

### Done when

A failed physical worker test tells the operator the real Kubernetes reason
without a manual `kubectl describe`, and a second run is not blocked by stale
acceptance resources.

---

# Phase 1 — make worker onboarding an administrative product surface

**Goal:** adding tomorrow's 4x A6000 host should feel like a supported
infer-stack operation, not a reconstruction of today's debugging session.

The A6000 host should join as a **worker**, not as another Kubernetes server,
for this milestone. Do not combine worker onboarding with control-plane HA.

### 1.1 Fresh-host preflight

Before changing the host, report:

- canonical local hostname / proposed Kubernetes node name;
- connectivity to the selected Kubernetes API address;
- NVIDIA driver and `nvidia-smi` inventory;
- `nvidia-container-runtime` availability;
- candidate K3s version (must match the cluster policy);
- filesystem containing the proposed K3s/containerd data root;
- bytes and inodes free;
- Docker/container runtime disk usage when present;
- a warning when the expected image-pull/extraction headroom is too close to
  kubelet eviction/reclaim thresholds.

Do not hard-code one free-space number. Where possible inspect the configured
kubelet eviction thresholds and report both:

```text
hard pressure threshold
minimum-reclaim target
current free space
headroom after target
```

### 1.2 Explicit K3s data location

Support an administrator choosing the K3s data directory during onboarding.
The default may remain K3s's normal path, but machines with a large service/NVMe
filesystem should not require a reinstall to move containerd later.

The option must be visible in dry-run output and persisted in the systemd
service configuration.

### 1.3 Canonical identity

After join, discover and print the actual Kubernetes node object name. Future
commands should use or resolve that canonical name rather than assuming
`hostname -s`.

### 1.4 Cluster context persistence

infer-stack should have a persisted notion of the administrative kubeconfig or
cluster target it manages. Its KubeAI/TUI/kube operations should not depend on a
shell-specific `export KUBECONFIG=...` after bootstrap.

Do **not** silently:

- edit `.bashrc`;
- make `/etc/rancher/k3s/k3s.yaml` world-readable;
- overwrite an unrelated user's default Kubernetes context.

Prefer an explicit infer-stack setting/context whose path is created with
private permissions and passed to subprocesses it owns.

### 1.5 Acceptance levels

Keep the existing exact-device probe and add an explicit serving-GPU count so a
multi-GPU host can be tested as a multi-GPU serving host, not only as four GPUs
followed by a one-GPU smoke model.

Possible surface (spelling may change):

```text
infer-stack kube node test NODE --expected-gpus=4 --serve-gpus=4 --apply
```

The smoke model/runtime used for `--serve-gpus=N` must actually exercise N
GPUs. Do not claim a TP=4 acceptance from an inventory-only check.

### Tomorrow's 4x A6000 acceptance

On the physical A6000 host prove:

- [ ] clean preflight before installation;
- [ ] worker joins the existing `yardrat` cluster;
- [ ] canonical node name is reported;
- [ ] all four A6000 device UUIDs/models are visible;
- [ ] `nvidia.com/gpu=4` is allocatable;
- [ ] device plugin/GFD/runtime are healthy;
- [ ] exact-node one-GPU generation works;
- [ ] multi-GPU serving acceptance works when the test implementation is ready;
- [ ] disk pressure remains false while pulling/extracting the serving image;
- [ ] temporary acceptance resources are removed afterward.

---

# Phase 2 — administrator-defined machine classes and user placement constraints

**Goal:** users can request *where a deployment is eligible to run* without
infer-stack becoming a replacement scheduler.

### 2.1 Node classification

Use administrator-owned Kubernetes labels as the initial durable classification
surface. Reserve an infer-stack label namespace, for example:

```text
infer-stack.ai/pool=a6000
infer-stack.ai/class=4xa6000
infer-stack.ai/site=kitware-lab
```

Exact label vocabulary is revisable. Do not bake site-specific pool names into
Python code.

Add CLI/TUI administration for listing and applying the supported labels rather
than requiring routine raw `kubectl label` commands.

### 2.2 Catalog eligibility

Extend the endpoint placement schema beyond today's `min_vram_gib` and local
`gpu_indices` so a catalog can restrict Kubernetes eligibility, for example:

```yaml
placement:
  min_vram_gib: 40
  pool: a6000
```

or an equivalent normalized representation.

Catalog constraints are administrator policy and define the maximum allowed
placement domain.

### 2.3 Per-acquire narrowing

Allow a lease request to further narrow placement:

```text
infer-stack acquire ENDPOINT --node NODE
infer-stack acquire ENDPOINT --pool POOL
```

Exact spelling may change. Required semantics:

- a request cannot widen catalog eligibility;
- nonexistent/impossible constraints fail with a specific reason;
- temporarily busy eligible resources queue/wait rather than being described
  as impossible;
- hard placement constraints participate in deployment compatibility, so two
  incompatible requests do not coalesce onto one deployment.

### 2.4 Kubernetes remains the scheduler

Translate normalized placement constraints to selectors/affinity/resource
profiles. Do not implement a second global scheduler in infer-stack.

Use the scheduler's structured reason to distinguish:

```text
no eligible hardware exists
eligible hardware exists but is busy
node is tainted/draining
storage/runtime constraint blocks placement
```

### 2.5 Heterogeneous-node correctness

Do not assume one node-level product label completely describes a physically
heterogeneous node. Keep per-device inventory where infer-stack needs exact
facts, and define clearly which Kubernetes placement policies are expressible
at node granularity.

Homogeneous 4x A6000 hardware is a good next target, but the design must not
silently misrepresent machines such as `yardrat` with different GPU products.

---

# Phase 3 — expose demand-driven replica scaling

**Goal:** one logical endpoint can add serving capacity as demand increases,
without one replica per lease.

The current KubeAI renderer already emits `minReplicas`/`maxReplicas`; extend
this deliberately rather than building infer-stack replica scheduling.

### 3.1 Correct the current replica contract

- Permit a real `min_replicas: 0`; do not coerce zero back to one.
- Validate `min_replicas <= max_replicas`.
- Keep replica policy out of structural model identity unless changing it would
  make sharing unsafe; distinguish serving policy from model/runtime identity.

### 3.2 Autoscaling policy

Expose the KubeAI policy needed to scale from request pressure, including the
current operator's supported equivalent of:

- target concurrent/queued requests;
- scale-down delay;
- load-balancing policy;
- minimum warm replicas;
- maximum replicas.

Keep the catalog spelling backend-neutral where practical. The backend may map
it to KubeAI fields.

### 3.3 User-visible replica state

Status/TUI should distinguish:

```text
leases/demand
logical deployments
wanted replicas
ready replicas
starting replicas
request pressure / queue depth (when observable)
```

### 3.4 Tests

A deterministic single-node test must prove:

1. model starts with the configured minimum;
2. concurrent load causes additional replicas to be requested;
3. requests continue through scale-up;
4. replicas land only on eligible resources;
5. load removal eventually scales down according to policy;
6. active leases keep the deployment resident according to the selected
   residency policy.

Prefer a CPU/synthetic serving model for this control-loop test so it can run in
CI/VMs.

---

# Phase 4 — lifecycle/status UX and startup latency

**Goal:** an acquire that is not immediately ready explains why and gives the
best evidence-based expectation available.

### 4.1 Normalize operation phases

Introduce a backend-neutral operation/startup view whose Kubernetes
implementation can expose at least:

```text
READY
SCALING
WAITING_FOR_CAPACITY
SCHEDULING
PULLING_IMAGE
FETCHING_MODEL
LOADING_MODEL
STARTING
BLOCKED
FAILED
```

Do not derive all of these from one pod phase; combine scheduler events,
container states, KubeAI Model/replica state, and serving health as appropriate.

### 4.2 Preserve raw reason and next action

Every terminal/blocking failure should retain the useful backend reason and a
specific next action. Examples from the physical test:

- `ephemeral-storage` eviction should say which filesystem signal failed;
- `Insufficient nvidia.com/gpu` should identify the eligible node(s) and current
  allocation;
- a disk-pressure taint should be reported as disk pressure, not generic
  Unschedulable;
- a failed serving pod should not regress to `waiting for scheduler`.

### 4.3 Startup observations

Record timestamps for meaningful boundaries such as:

- lease accepted;
- scheduler placement;
- image pull start/end;
- container start;
- model ready;
- first successful generation.

Key historical distributions by useful identity such as:

```text
model/runtime revision
GPU class
cache state
```

Then surface measured ranges. Do not claim a prediction when the sample count or
state match is weak.

### 4.4 Reduce latency using measured bottlenecks

Candidate mechanisms:

- pre-pull standard serving images during/admin after node onboarding;
- cache selected model weights on selected pools;
- keep one warm replica for latency-sensitive endpoints;
- allow scale-to-zero for expensive infrequent endpoints;
- prefer already-cached eligible nodes when this does not violate fairness or
  hard placement constraints.

Instrument first; optimize based on the measured startup decomposition.

---

# Phase 5 — resource accounting and energy telemetry v1

**Goal:** make accelerator consumption and the best available energy estimate
visible in CLI/TUI/API without false precision.

This is a cross-cutting requirement from
[northstar.md](northstar.md#resource-accounting-and-energy-transparency). It
should begin before the remote/HA controller so the data model can migrate with
the rest of the authority later.

## 5.1 Define accounting facts before choosing telemetry plumbing

Persist or expose a backend-neutral usage record capable of representing:

- interval start/end;
- node identity;
- stable GPU identity/UUID when known;
- GPU product/class;
- deployment/model/endpoint identity;
- replica/pod identity;
- allocated GPU count;
- GPU-seconds;
- sampled/aggregated power when available;
- integrated energy in joules/Wh;
- measurement/estimate method;
- sample coverage and gaps;
- provenance/version needed to recompute estimates.

Avoid storing only a precomputed "kWh" scalar with no explanation.

## 5.2 Measurement ladder

Implement a clear hierarchy and label it in output:

### Level A — measured device power integration

When supported, sample GPU power telemetry (for example through an NVIDIA
telemetry source) and integrate watts over time:

```text
energy_joules ~= sum(power_watts * sample_interval_seconds)
```

Store enough metadata to identify sampling cadence and missing intervals.

### Level B — observed power proxy

When direct coverage for the workload is incomplete, use an explicitly named
observed device/workload-class power estimate derived from real telemetry.

### Level C — static device-class proxy

Fallback to:

```text
GPU-hours x explicit proxy watts
```

The proxy must be a recorded input, not an unexplained constant hidden in code.

### Level D — resource time only

If no responsible power proxy exists, show GPU-hours and say energy is not yet
estimated.

## 5.3 Start with deployment-level truth

Kubernetes allocates whole GPUs today, so deployment/replica-level accounting
is the first defensible scope.

Do **not** duplicate the full deployment energy onto every lease sharing it.
Initial user output may legitimately say:

```text
deployment energy measured: 0.42 kWh
lease-attributed energy: unavailable (shared deployment)
```

Later attribution may use gateway request windows, token counts, or another
published method.

## 5.4 Separate reservation from activity

Track and display at least:

- GPU allocation/residency time;
- measured energy over that residency;
- request/token activity where observable.

This allows the system to make warm-idle cost visible rather than hiding it in
request averages.

If feasible from the telemetry, later split startup / active serving / warm-idle
energy. Do not block v1 on perfect phase attribution.

## 5.5 User surfaces

Add a read-only surface such as `infer-stack usage` (name revisable) and TUI/API
fields that can answer:

```text
endpoint/deployment
GPU product x count
GPU-hours
energy Wh/kWh
measured vs estimated
method/source
coverage/confidence
observation window
```

Support useful aggregation by endpoint/model/node/time window as the data model
allows.

Do not mix carbon into the primary energy value. A future carbon estimate should
multiply energy by an explicitly sourced carbon-intensity series/assumption.

## 5.6 Deterministic tests

Energy/accounting tests must not require physical GPUs. Feed synthetic power
samples and prove:

- unit conversions;
- numerical integration;
- missing-sample behavior;
- no double counting across adjacent windows;
- aggregation across replicas;
- shared-lease output does not duplicate total energy;
- fallback/proxy method is labeled;
- changing a proxy can recompute estimates from retained raw resource facts.

A later physical acceptance test should compare the collector's device IDs and
power readings against `nvidia-smi`/the chosen telemetry source on the A6000
host.

---

# Phase 6 — make infer-stack a remote single-authority service

**Goal:** the CLI/TUI can run on an unrelated workstation without copying the
SQLite ledger or granting the user Kubernetes admin credentials.

This is the concrete next slice of
[distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md).

### 6.1 Service boundary first

Introduce one domain service used by both:

```text
local CLI/TUI -> in-process service -> SQLite/backend
remote client -> typed API -> same service -> SQLite/backend
```

Preserve local standalone operation.

### 6.2 One remote server before HA

Run one canonical infer-stack server on a chosen management host. It owns:

- the ledger/state store;
- catalog revision/input;
- reconciliation;
- Kubernetes credentials;
- operation/status stream;
- accounting records until they move to the durable distributed store.

Clients should not need kubeconfig.

### 6.3 Separate API capabilities now, harden identity later

Define actions/resources so the service can distinguish:

- node/cluster administration;
- catalog/policy administration;
- lease-user operations;
- read-only observation/accounting.

On the trusted LAN, the first deployment may use one shared credential or
simple scoped tokens. Do not postpone the API boundary by building a full OIDC
or quota system first.

### 6.4 TUI becomes a client

The TUI should be able to connect to the canonical infer-stack service from
`yardrat`, `namek`, or an unrelated workstation and see the same authority.

Do not achieve this by copying the SQLite database/data root to each client.

---

# Phase 7 — make the Kubernetes control plane replaceable

**Goal:** `yardrat` stops being a permanent single point of Kubernetes control
without conflating this with infer-stack controller HA.

Do this after the worker/resource model is stable enough that control-plane
migration is not being debugged at the same time as basic scheduling.

### 7.1 Stable Kubernetes API/registration address

Introduce a stable cluster-control name/address before adding redundant servers.
Workers and administrative kubeconfigs should target that stable endpoint rather
than treating `yardrat` as permanent identity.

### 7.2 Move from single-server datastore to HA-capable K3s mode

Plan and test the K3s-supported migration from the current single server to an
HA-capable datastore/control-plane topology. At implementation time, re-check
the exact procedure against the pinned K3s version.

Prefer an odd number of control-plane/datastore members for quorum. Three is the
normal small-cluster target.

### 7.3 Add new server(s), then remove yardrat if desired

Acceptance must prove:

- existing workers remain registered;
- KubeAI resources remain intact;
- healthy inference survives one control-plane server loss;
- a fresh acquire/release still works after failover;
- `yardrat` can be drained/removed as a server without redefining the logical
  cluster.

The infer-stack authority may still be a single process during this phase.

---

# Phase 8 — durable/HA infer-stack authority and stronger roles

**Goal:** remove the infer-stack controller host as a single point of mutation
while preserving one logical authority.

Follow the detailed design in
[distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md):

- PostgreSQL (or another deliberately chosen transactional shared state store)
  instead of network-shared SQLite;
- idempotent operation identities;
- per-object desired generations/fencing;
- active-active API replicas;
- a fenced reconciliation leader initially;
- durable audit/accounting state;
- controller failover tests;
- backup/restore.

This is also the right stage to make administrator/user identity and RBAC
operationally real if the trusted-LAN shared-credential phase has outlived its
usefulness.

Per-user quotas remain optional until there is a concrete need.

---

# Cross-cutting single-node test strategy

Every implementation phase should ask: **what part of this can be proven on one
machine before touching the physical cluster?**

## Layer A — pure deterministic tests

Use fake scheduler/backend/telemetry inputs for:

- lease compatibility/coalescing;
- node/pool placement normalization;
- autoscaling render policy;
- lifecycle status mapping;
- accounting/energy integration;
- authority/RBAC decisions;
- idempotency and cleanup.

## Layer B — real single-node K3s/KubeAI

Use a disposable local/VM cluster to exercise:

- CRD/operator reconciliation;
- gateway/service routing;
- scale-up/down behavior;
- node labels/selectors;
- controller restart;
- cleanup;
- CPU-backed serving where GPU semantics are not essential.

## Layer C — simulated second node

Keep/improve the existing containerized K3s-agent mechanism for:

- exact-node selectors;
- pools/classes;
- cross-node service networking;
- multi-node scheduling outcomes;
- taints/drain behavior.

## Layer D — deterministic failure injection

Reproduce today's real failures on one machine:

- DiskPressure/ephemeral-storage eviction;
- `Insufficient nvidia.com/gpu`;
- stale assigned Pending pod;
- failed image pull;
- failed model container;
- slow deletion/cleanup race;
- controller process restart;
- unavailable Kubernetes API.

Do not require filling a real root disk to test the controller's response;
construct backend observations/events that reproduce the state machine.

## Layer E — thin hardware acceptance

Real machines should prove only hardware-specific claims:

- actual NVIDIA runtime/device identity;
- real multi-GPU serving;
- real model generation;
- telemetry power source identity/values;
- physical cross-host networking.

---

# Overnight implementation tranche

An agent working without access to the physical GPUs should prioritize changes
that improve tomorrow's A6000 test and are mechanically testable locally.

Work in this order unless a discovered dependency justifies changing it.

## O1. Preserve current behavior and read the plans

Before editing:

- read [northstar.md](northstar.md);
- read this file;
- read [distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md);
- read [backend-parity-roadmap.md](backend-parity-roadmap.md);
- inspect the current KubeAI renderer, worker acceptance path, catalog placement
  validation, controller/ledger, and test helpers.

Do not implement the stretch goals merely because they are mentioned in the
north star.

## O2. Onboarding/preflight hardening

Implement as much of Phase 1 as can be tested without root/K3s mutation:

- structured preflight facts;
- disk/inode/headroom diagnosis;
- configurable K3s data-dir plan/rendering;
- canonical node-name discovery/reporting;
- persisted infer-stack kubeconfig/cluster target design and tests;
- no shell-startup-file edits.

If changing K3s installation behavior cannot be verified safely in the agent
environment, isolate the logic and provide a handover command/test for tomorrow.

## O3. Placement vocabulary

Implement Phase 2's normalized exact-node and pool constraint model with fake
KubeAI tests. Preserve existing catalogs when the new fields are absent.

Do not build a scheduler.

## O4. Replica/autoscaling renderer

Fix `min_replicas=0`, add validated scaling-policy fields supported by the
current KubeAI version, and add deterministic render/status tests.

Do not claim load-based scale-up is physically verified unless the test actually
generates load against a real KubeAI installation.

## O5. Resource-accounting foundation

If time permits, implement the backend-neutral accounting data model plus pure
synthetic energy-integration tests from Phase 5. It is acceptable to leave the
real NVIDIA collector as a handover if the agent has no GPU.

The first implementation must already encode measured-vs-estimated provenance;
do not add a bare unlabeled `energy_kwh` value that will be difficult to fix
later.

## O6. Update handover

Leave a concise handover containing:

- commits made;
- tests run and exact results;
- changes that require root/K3s/GPU validation;
- exact commands for the 4x A6000 acceptance;
- any plan revision proposed and why.

---

# Explicit short-term non-goals

Do not spend the next tranche implementing:

- unknown/public node donation;
- Internet/WAN federation;
- quota/billing machinery;
- hardened hostile multi-tenancy;
- HA infer-stack controller replicas before the single remote service boundary;
- PostgreSQL solely to say the system is distributed;
- a custom GPU scheduler;
- perfect request-level energy attribution;
- carbon accounting before energy accounting is credible.

---

# Questions the implementation should keep visible

These do not need to block the next phases, but code should avoid making them
hard to answer later.

1. What exactly makes two constrained leases compatible enough to coalesce?
2. Should node pool selection be hard-only initially, or do we also need soft
   preferences/cache locality soon?
3. What is the stable user-facing endpoint when replicas scale and controllers
   move?
4. Which request-pressure signal gives KubeAI the best scaling behavior for LLM
   traffic?
5. What startup boundaries can be observed reliably across runtimes?
6. Where should durable usage/accounting facts live before and after the remote
   state-store migration?
7. Which energy telemetry source gives sufficiently stable GPU UUID/power data
   across the hardware we operate?
8. How should deployment-level energy eventually be apportioned across
   concurrent requests/leases without double counting?
9. Which state must migrate to move the single infer-stack authority to another
   host before PostgreSQL/HA exists?
10. What stable Kubernetes API address should be used before adding multiple K3s
    servers?

# infer-stack north star: a shared accelerator service

**Status:** product intent · 2026-10-07

This document describes **what infer-stack should make possible and why**. It
is intentionally less specific about implementation than the other planning
documents.

Implementation plans may change as better designs become available. They should
not silently change the goals in this document. If an implementation plan and
this document disagree, either revise the plan or explicitly revise this north
star with an operator decision.

Related implementation documents:

- [kubernetes-service-roadmap.md](kubernetes-service-roadmap.md) — current
  executable roadmap from the working two-node K3s/KubeAI proof;
- [distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md)
  — detailed remote/HA controller design;
- [backend-parity-roadmap.md](backend-parity-roadmap.md) — backend contract and
  KubeAI parity history;
- [vram-aware-placement.md](vram-aware-placement.md) — current resource-fit
  model.

---

## Motivating idea

A collection of accelerator-equipped machines should be usable as one managed
resource pool instead of a set of hosts that people manually SSH into and
configure independently.

An administrator should be able to create a cluster, admit known machines into
it, describe useful machine classes or pools, and see which resources are
healthy and available. A user or automated agent should be able to ask for a
named inference endpoint without needing to know how to launch a container,
which GPU is free, which machine has the right hardware, or how the serving
runtime is wired together.

Conceptually:

```text
administrator
    |
    | creates/manages a resource pool
    v
+--------------------------------------------------+
| infer-stack service                              |
|                                                  |
|  approved endpoint catalog                      |
|  lease / demand authority                       |
|  placement policy                               |
|  user-visible status                            |
|  resource + energy accounting                   |
+--------------------------+-----------------------+
                           |
                           | desired serving intent
                           v
                cluster scheduler / runtime
                  /          |          \
                 /           |           \
            GPU host     GPU host     GPU host

user / agent
    |
    +-- acquire endpoint --------------------------^
    +-- use stable endpoint
    +-- release lease
    +-- understand wait state, resource use, energy
```

The resource pool may begin as a trusted LAN inside one organization. The
architecture should not make that trust assumption impossible to relax later,
but solving an open public compute marketplace is not required to make the
internal service useful.

---

## The experience we want

### For a cluster administrator

An administrator should be able to:

- bootstrap one usable cluster without manually assembling low-level commands;
- add a machine with a small, reviewable onboarding procedure;
- see the canonical node identity, GPU inventory, runtime health, disk/cache
  health, and scheduling eligibility;
- classify machines into useful pools such as `a6000`, `interactive`, or
  `large-vram` without hard-coding those names into application logic;
- drain, remove, repair, or replace machines without losing the logical
  infer-stack service;
- eventually run multiple redundant Kubernetes control-plane machines and move
  away from the machine that originally bootstrapped the cluster;
- eventually move or replicate the infer-stack controller itself without
  creating a second competing authority;
- define the endpoint catalog and the maximum resources/placement choices users
  are allowed to request.

The administrator should not need to manually edit Kubernetes objects for
normal infer-stack operations.

### For an endpoint user

A user or automated agent should be able to ask for an approved logical
endpoint:

```text
"give me hyperqwen"
```

and receive a stable endpoint contract. If the model is already warm, the
request should become usable quickly. If it is cold or capacity is occupied,
the system should clearly explain what it is waiting for.

A user should not have to know:

- which physical machine serves the model;
- whether the serving implementation is one replica or several;
- the Kubernetes resource names;
- how to run `kubectl`;
- how the model image, weights, runtime class, or gateway are configured.

Advanced users should be able to **narrow** placement when there is a reason:
for example, request a specific machine, an administrator-defined machine pool,
or a hardware class. A user request must not widen the endpoint's
administrator-approved eligibility or bypass policy.

### For an operator watching the system

The CLI/TUI should answer, without requiring a Kubernetes debugging session:

- What is running?
- Why is it running?
- Who or what currently holds demand for it?
- Where did it land?
- How many replicas are ready, starting, or wanted?
- Is the request waiting for capacity, image download, model download, model
  load, health checks, or another condition?
- What can the operator do next if it is blocked?
- How much accelerator capacity has this model consumed?
- How much energy has it used, or what is the best available proxy?

---

## Separate the control planes

"Control node" is too ambiguous for future design discussions. There are at
least two control planes and a data plane:

### Kubernetes control plane

Owns cluster membership, Kubernetes API state, scheduling, kubelet coordination,
and the Kubernetes datastore.

Its physical machines should eventually be replaceable and redundant.

### infer-stack control plane

Owns endpoint demand/leases, approved catalog intent, operations, policy,
user-visible state, and reconciliation into the serving backend.

It should have **one logical authority** even if several controller processes
serve it later.

### Inference data plane

The model servers, gateways, caches, and routes that answer inference traffic.
Already-running inference should not depend on an infer-stack controller process
being continuously alive.

These concerns may be colocated on one machine in a small installation. That is
an implementation convenience, not an identity relationship.

---

## Leases express demand; replicas express serving capacity

A lease means there is legitimate demand for an endpoint to remain available.
It should not imply one serving replica per lease.

For example:

```text
10 compatible leases for model X
        |
        v
1 logical deployment for model X
        |
        v
1 .. N serving replicas, chosen from actual load and policy
```

This distinction is required for efficient sharing. A lightly used model can
serve many lease holders from one warm replica. A heavily used model can scale
to several replicas without inventing fake leases.

The system should make the relationship visible: lease demand, desired
residency, desired replica count, ready replica count, and actual request load
are different facts.

---

## Placement is a constraint, not a replacement scheduler

infer-stack should express **eligibility and preference**. The cluster scheduler
should decide the concrete placement among eligible resources.

Useful placement requests include:

- enough GPUs and enough VRAM;
- exact machine;
- administrator-defined machine pool/class;
- accelerator capabilities needed by a runtime;
- eventually soft preferences such as cache locality or energy efficiency.

The catalog defines the maximum allowed placement domain. Per-acquire placement
may narrow that domain but must not widen it.

Placement constraints are part of whether two leases can safely share one
logical deployment. Requests with incompatible hard constraints must not be
silently coalesced.

---

## Fast when possible, explicit when not

A stable endpoint should hide deployment mechanics, but it should not hide
waiting.

The system should distinguish states such as:

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

The exact vocabulary may evolve. The principle is that a user should know
**what class of delay is occurring**.

Do not manufacture precise ETAs without evidence. Prefer measured historical
ranges and concrete cache/capacity facts, for example:

```text
model is cold
image is cached
weights are cached
eligible free GPUs: 0 / 4
recent cold starts on this GPU class: p50 43 s, p90 68 s
```

Startup speed should improve over time through evidence-based mechanisms such
as image pre-pulls, model caching, warm-replica policy, cache-aware placement,
and measured startup histories.

---

## Resource accounting and energy transparency

Resource use should be a first-class observable property of inference, not an
afterthought.

At minimum, infer-stack should be able to report durable accounting facts such
as:

- accelerator type and count;
- GPU-seconds / GPU-hours allocated;
- deployment residency time;
- request count and, when available, input/output token counts;
- replica count over time;
- cache/startup events that materially affect resource use.

### Energy is a user-visible goal

For each model/deployment, and eventually for useful user/lease/request scopes,
infer-stack should show either **measured energy** or the **best defensible
proxy available**.

The interface must distinguish the two. It must never present a rough estimate
with measurement-level precision.

A useful result has provenance, for example:

```text
energy: 0.184 kWh
method: measured GPU power integration
source: NVML/DCGM samples from GPU UUID ...
coverage: 99.2% of deployment runtime
```

or:

```text
energy: ~0.21 kWh
method: estimated from 0.72 GPU-hours x 290 W proxy
source: device-class proxy
confidence: low
```

The implementation may improve through a ladder of evidence:

1. direct power telemetry integrated over time when available;
2. observed device-class power from nearby periods/workloads;
3. GPU-time multiplied by an explicit power proxy;
4. raw resource time only when no responsible energy estimate is available.

Raw accounting facts should be retained so estimates can be recomputed later as
better models become available.

### Attribution must be honest

A deployment may serve several leases and several concurrent requests. The same
joule must not be counted in full for every lease.

It is acceptable for early versions to expose accurate deployment-level energy
while saying that per-user attribution is unavailable. Later attribution can
use request overlap, token counts, or another declared method, but the method
must be visible.

Idle/warm residency is also real resource use. The UI should be able to
separate, where practical:

- startup energy;
- active serving energy;
- warm-idle energy;
- estimated/request-attributed energy.

Carbon emissions are a separate derived quantity. If added later, they should
be computed from energy plus an explicit carbon-intensity source/time/location
assumption rather than being conflated with energy measurement.

---

## Remote use should not require Kubernetes credentials

Kubernetes can already be controlled remotely with a kubeconfig, but normal
infer-stack users should not need cluster-admin credentials.

The long-term user shape is:

```text
CLI / TUI / agent
        |
        v
infer-stack service API
        |
        v
cluster API / serving backend
```

This allows the TUI to run on an unrelated workstation while still talking to
one canonical infer-stack authority.

The short-term trusted-LAN deployment may use one shared administrative
identity. The API/resource model should nevertheless keep these concepts
separate from the beginning:

- **cluster administrator:** admits/removes nodes and manages cluster
  infrastructure;
- **infer-stack administrator/operator:** manages catalog/policy and privileged
  operations;
- **lease user:** acquires approved endpoints and releases their leases.

Authentication, quotas, and hardened multi-tenant isolation can arrive later
without requiring the core domain model to be rewritten.

---

## Control-plane location should not define the service

The cluster should survive replacing the machine that first created it.
Similarly, moving the infer-stack authority should not require users to learn a
new serving topology or cause healthy inference to be recreated merely because
the controller moved.

The intended progression is:

- one Kubernetes server and one infer-stack authority are acceptable initially;
- introduce stable service/control addresses before relying on physical host
  names as permanent identities;
- support redundant Kubernetes control-plane machines;
- support relocating the single infer-stack authority;
- only then add multiple infer-stack controller replicas against one shared
  authoritative state.

High availability must not be implemented by running independent infer-stack
ledgers against the same namespace.

---

## Testability is a product requirement

Most correctness failures should be discoverable without assembling several
physical GPU machines.

The development ladder should include:

1. deterministic unit and fake-backend tests;
2. real single-node Kubernetes/KubeAI tests, including CPU-backed serving where
   sufficient to test control semantics;
3. simulated multi-node scheduling/network tests on one physical machine or VM;
4. fault-injection tests for eviction, disk pressure, taints, failed image
   pulls, stale resources, controller restarts, and cleanup races;
5. thin real-GPU acceptance tests for facts that cannot be simulated
   responsibly.

A physical cluster should validate hardware/runtime assumptions, not be the
first place the controller state machine is exercised.

---

## Near-term trust boundary

For the next development stage it is acceptable to assume:

- machines are known and intentionally admitted by an administrator;
- machines share a trusted company LAN;
- users are trusted collaborators;
- one credential may initially cover both operator and lease-user actions.

Even in this mode, normal lease requests should refer to administrator-approved
catalog endpoints rather than arbitrary images, commands, mounts, or privileged
containers. That keeps a later security boundary possible.

---

## Deliberate short-term non-goals

These ideas are compatible with the north star but should not drive the next
implementation tranche:

- accepting arbitrary unknown machines that volunteer compute;
- Internet-scale or multi-region federation;
- a public distributed compute marketplace;
- per-user quotas, billing, or chargeback;
- hardened hostile multi-tenancy;
- replacing Kubernetes scheduling with a custom infer-stack scheduler;
- perfect per-request energy attribution;
- precise carbon accounting without an explicit energy and carbon-intensity
  basis.

A future opt-in network where independent organizations donate machines is an
interesting extension. It requires a substantially stronger trust, enrollment,
isolation, accounting, and abuse model than the trusted-LAN system and should be
built on top of—not mixed into—the first cluster service.

---

## Decisions intentionally left open

This document does not require forever using:

- K3s rather than another conforming Kubernetes distribution;
- KubeAI rather than another serving operator;
- LiteLLM rather than another stable front door;
- SQLite or PostgreSQL specifically;
- a particular authentication provider;
- NVML/DCGM or any one energy telemetry stack;
- one particular autoscaling algorithm.

Those are implementation choices. The durable requirements are the authority
boundaries, lease semantics, resource-pool experience, placement behavior,
status transparency, accounting, remote controllability, and testability above.

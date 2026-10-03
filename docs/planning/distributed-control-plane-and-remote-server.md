# Distributed control plane, remote server, and HA controller plan

**Status:** proposed 2026-10-03 · design only · no implementation in this PR

## Objective

> **Evolve infer-stack from a host-local controller into a model-serving control
> plane that can be operated remotely and, when needed, run with multiple
> redundant controller processes without creating multiple competing
> authorities.**

The important distinction is between a **logical control plane** and a
**control-plane machine**.

The target invariant is not "there must be exactly one machine running
infer-stack." It is:

> **There is exactly one authoritative infer-stack control-plane state for a
> managed backend domain, and one or more controller replicas may serve and
> reconcile that state. At least one healthy controller is required for
> mutations and reconciliation; existing inference should continue when all
> controllers are temporarily unavailable.**

This plan also introduces a remote control API and a node-agent model for
Compose hosts. Security is a first-class requirement: a remotely reachable
infer-stack server must not become a generic remote shell or an unaudited way to
launch arbitrary privileged containers.

The plan deliberately preserves the useful existing local workflow. A single
workstation should still be able to run infer-stack with no external database,
no distributed-systems setup, and no server daemon if the operator does not
need those features.

---

## Why this direction

Several current features already point toward a real control plane:

- leases represent user demand rather than a single global "active model";
- the controller reconciles desired deployments rather than merely executing a
  one-shot render;
- dynamic LiteLLM routing can add and remove individual upstream deployments
  without recreating the gateway;
- KubeAI delegates cluster placement to Kubernetes while keeping infer-stack as
  the lease/catalog/front-door coordinator;
- multiple clients can already share one ledger when they use the same
  authority and data root.

The current limitation is that the authoritative state is still fundamentally
local. Compose uses a local SQLite ledger and rendered files, and the supported
operating model is one controller authority per Docker host. KubeAI similarly
assumes one infer-stack authority per managed namespace. Running independent
controllers with independent data roots against the same backend is unsafe.

That limitation becomes increasingly awkward for the intended uses:

1. a GPU server should be controllable from another machine without SSH-driven
   imperative scripts;
2. a cluster should not depend on one particular workstation being the only
   place where administrative commands can run;
3. multiple users, agents, CI jobs, or higher-level schedulers should be able to
   acquire and release endpoints through one stable interface;
4. a control-plane process restart should not interrupt model traffic;
5. eventually, loss of one controller machine should not stop new control
   operations if another controller replica is healthy.

This is a control-plane problem, not merely a resource-namespacing problem.

---

## Terminology

This document uses the following terms consistently.

### Logical control plane

The single authoritative infer-stack state for a managed domain: leases,
desired deployments, endpoint contracts, node facts, operations, catalog
revision, and reconciliation state.

Multiple controller processes may serve one logical control plane. Two
independent databases controlling the same Docker/Kubernetes resources are
**two control planes** and remain unsupported.

### Controller

An infer-stack process that exposes the control API and may participate in
reconciliation.

Controller API replicas may be active-active. Reconciliation should initially
be single-leader, with failover.

### Reconciler

The component that converts authoritative desired state into backend actions
and records observed/converged state.

### State store

The transactional source of truth for the logical control plane.

- Local/standalone mode: SQLite remains appropriate.
- Server/HA mode: PostgreSQL is the proposed store.

### Node agent

A small infer-stack-managed process on a Compose GPU host. It reports local
facts and applies controller-authorized desired state using the local Docker
runtime.

The controller should not require a remotely exposed Docker socket.

### Data plane

The services that answer inference traffic: LiteLLM, vLLM, Ollama, KubeAI
model servers, and related serving components.

### Control plane

The infer-stack API, state store, reconciliation logic, controller replicas,
and node agents.

The data plane must not require the control plane to be continuously reachable
in order to keep answering already-routed inference requests.

---

## Architectural principles and invariants

These are stronger than implementation suggestions; later designs should
preserve them unless this document is explicitly superseded.

### 1. One logical authority, optionally many controller replicas

There may be N controller processes, but they all operate on the same
authoritative state.

Do not implement HA by copying SQLite files, putting SQLite on a network
filesystem, or allowing several independent ledgers to "eventually agree."

### 2. Existing inference survives control-plane loss

If every controller crashes or is restarted:

- running vLLM/Ollama/KubeAI model servers continue;
- LiteLLM continues routing through its already-published routes;
- clients using existing endpoints continue to generate;
- durable lease state remains in the state store.

What stops until a controller returns:

- new acquires;
- releases and evictions;
- TTL garbage collection;
- desired-state reconciliation;
- route changes;
- node onboarding and administrative mutations.

This failure mode is intentional.

### 3. The state store is authoritative; rendered artifacts are derived

Rendered Compose YAML, env files, cached inventories, and local realization
files are useful implementation artifacts, not the distributed source of truth.

A controller must be able to reconstruct desired backend state from the
authoritative store plus catalog/config inputs.

### 4. Mutations are transactional and idempotent

An API client must be able to retry an operation after a timeout without
accidentally acquiring two leases or launching two unintended deployments.

Mutating API operations should accept or derive an idempotency key. Durable
operation records should expose whether a request was accepted, converging,
succeeded, failed, or superseded.

### 5. Reconciliation is fenced by generation

A distributed lock alone is insufficient.

Every mutable desired-state object that can trigger backend actions needs a
monotonic desired generation (or equivalent fencing token). Backend
publication must never allow an older reconciler attempt to overwrite a newer
desired state.

Conceptually:

```text
desired generation 41
    -> controller A starts work

desired generation 42
    -> controller B becomes leader and converges 42

controller A resumes
    -> its generation-41 publication/action is rejected or ignored
```

Node agents must similarly reject stale commands once a newer generation has
been accepted.

### 6. Remote control is typed, not shell-shaped

The remote server exposes operations such as "acquire endpoint X" and "release
lease Y." It does not expose "run this arbitrary shell command as the infer-stack
service account."

A normal user must not be able to turn a model acquire into arbitrary Docker
root-equivalent execution through custom images, host mounts, commands, or
environment injection.

### 7. Catalog use and catalog mutation are separate permissions

Acquiring an administrator-approved catalog endpoint is a routine operation.
Changing an endpoint's image, mounts, command, runtime security context, or
model source is a higher-trust administrative operation.

### 8. Backends keep their proper responsibilities

infer-stack remains a broker/coordinator.

- Compose: infer-stack may do simple host-local placement because Docker does
  not provide a cluster scheduler.
- Kubernetes/KubeAI: infer-stack should delegate node/GPU placement and pod
  lifecycle to Kubernetes rather than building a competing scheduler.
- Slurm or another batch scheduler remains a separate scheduler integration,
  not something infer-stack reimplements.

---

## Target architecture

A server/HA installation should look approximately like this:

```text
                   infer-stack clients
           CLI / TUI / CI / pi / kwdagger
                         |
                      HTTPS
                         |
              +----------------------+
              | infer-stack API      |
              | controller replicas  |
              |      1 .. N          |
              +----------------------+
                   |            |
             authoritative      | route publication
             state store        v
              PostgreSQL     LiteLLM
                   |
             reconciler leader
               /          \
              /            \
     Compose node agents    Kubernetes / KubeAI
        /       \                 |
   Docker/GPU  Docker/GPU        pods
       hosts      hosts
```

The API tier may be active-active. The initial HA reconciliation model should
be one elected reconciler at a time.

A single-node installation remains:

```text
CLI -> controller library -> SQLite -> local Docker
```

or, if server mode is explicitly enabled:

```text
CLI -> localhost HTTPS/API -> controller -> SQLite -> local Docker
```

The server architecture must not force every workstation user to deploy
PostgreSQL.

---

## State-store design

### Local mode: keep SQLite

SQLite is a good fit when:

- one logical controller owns the backend;
- the database is on local storage;
- the operator wants the smallest possible dependency set.

Do not remove this mode merely to make the distributed design cleaner.

### Server / HA mode: PostgreSQL

PostgreSQL should become the authoritative infer-stack state store for remote
multi-client and HA deployments.

Do **not** put the existing SQLite ledger on NFS/SMB/shared storage as the HA
design.

PostgreSQL provides the primitives infer-stack needs:

- concurrent transactional writers;
- row-level locking;
- compare-and-swap style generation updates;
- durable operations;
- controller heartbeats;
- advisory locks or lease rows for leader election;
- schema migrations;
- audit/event storage;
- mature backup and HA options.

The application should retain a state-store abstraction so local SQLite and
server PostgreSQL implement the same domain operations rather than maintaining
two separate control-plane semantics.

### Relationship to LiteLLM's PostgreSQL

Dynamic LiteLLM routing already introduces PostgreSQL in the Compose gateway
configuration.

It may be operationally convenient to run infer-stack control state and
LiteLLM's model store in the same PostgreSQL *service*, but they should be
separate databases/schemas and separate credentials.

Reasons:

- LiteLLM's database schema is not infer-stack's control-plane API;
- infer-stack must not depend on undocumented LiteLLM tables;
- either component should be independently migratable;
- privilege boundaries should remain narrow;
- a LiteLLM upgrade must not become an infer-stack state migration.

Whether one physical PostgreSQL service or two is the deployment default is an
open operational question, not a semantic coupling.

---

## Conceptual durable state

Exact schemas are deferred, but server mode needs first-class records for at
least:

### Leases

- lease id;
- owner / authenticated principal;
- requested endpoint aliases;
- creation/expiry/release timestamps;
- state;
- idempotency key;
- catalog revision used for resolution.

### Deployments

- stable deployment id;
- engine/backend;
- compatibility identity;
- desired generation;
- observed generation;
- desired lifecycle state;
- observed lifecycle/readiness state;
- assigned node/resource identity where infer-stack owns placement;
- endpoint aliases/contracts served by the deployment.

### Nodes

For Compose node-agent hosts:

- stable node id;
- agent identity;
- heartbeat;
- schedulable/draining/offline state;
- GPU/device inventory;
- runtime capabilities;
- currently applied generations;
- cache/disk pressure summary.

Kubernetes nodes remain Kubernetes objects; infer-stack may cache observations
but should not duplicate Kubernetes as the scheduling authority.

### Operations

Mutations that may outlive an HTTP request need durable operation records:

- operation id;
- type;
- caller;
- idempotency key;
- requested object;
- accepted generation;
- state: pending/running/succeeded/failed/superseded;
- structured error;
- timestamps.

### Controller heartbeats / leadership

Track enough information to answer:

- which controller replicas are alive;
- which replica currently holds reconciliation leadership;
- current fencing epoch/token;
- when leadership last changed.

### Catalog revisions

Runtime state should record the catalog/config revision that produced it.

The catalog may remain file/Git based, but every deployment should be
attributable to an immutable content digest or Git commit so remote operations
remain auditable and reproducible.

### Audit events

Security-sensitive changes should create append-only audit events:

- principal;
- action;
- object;
- result;
- source address/client metadata where appropriate;
- catalog revision;
- timestamp.

---

## Transaction and concurrency model

### Acquire

An acquire should be one logical transaction from the client's perspective:

1. authenticate/authorize caller;
2. resolve endpoint contract against an immutable catalog revision;
3. check quota/policy;
4. select an existing compatible deployment or plan a new desired deployment;
5. create the lease;
6. increment/record demand;
7. advance desired generation if realization changes;
8. commit;
9. return lease + operation identities;
10. reconciliation proceeds.

A retry with the same idempotency key returns the same accepted operation/lease.

### Release

A release transaction:

1. validates caller ownership or elevated permission;
2. marks the lease released;
3. recomputes deployment demand;
4. advances desired generation if reclaim policy changes desired realization;
5. commits.

Backend teardown happens during reconciliation, not by partially mutating the
database around an imperative Docker command.

### Placement serialization

For Compose, two concurrent acquires must not both reserve the same exclusive
GPU.

The simplest initial implementation can serialize the placement/admission
transaction for a logical Compose scheduling domain using a PostgreSQL lock or
locked allocation rows.

Do not prematurely build a distributed bin-packing service. The host counts are
small; correctness matters more than high scheduler throughput.

---

## Reconciliation and HA

### Initial model: active-active API, single reconciliation leader

Run any number of API replicas:

```text
controller A: API + current reconciler leader
controller B: API
controller C: API
```

All may accept transactional API operations because PostgreSQL is authoritative.

Only one replica performs global reconciliation initially.

If A dies, B or C acquires leadership and resumes from durable desired/observed
state.

This is simpler and safer than immediately sharding reconciliation.

### Leader election

PostgreSQL can provide the first implementation through a connection-scoped
advisory lock or an explicit lease/heartbeat row.

Leader election must also allocate a monotonically increasing **leadership
epoch/fencing token**.

The lock answers "who should act now." The fencing token handles actions that
were already in flight when leadership changed.

### Per-object desired generations

Each deployment/route/node desired-state mutation advances a generation.

Controller and agent behavior must be monotonic:

```text
desired_generation = 105
observed_generation = 104
    -> reconciliation needed

observed_generation = 105
    -> converged

attempted publication generation = 103
    -> reject/ignore as stale
```

The backend adapters should record which generation they attempted and verify
the desired generation again before committing publication state.

### Future sharding

If one reconciler becomes a throughput bottleneck, reconciliation can later be
sharded by resource/domain using per-node or per-deployment locks.

That is explicitly **not required for the first HA release**.

---

## Dynamic LiteLLM routing

Dynamic routing fits the target architecture better than a controller-generated
static gateway superset:

- each deployment has an independent upstream identity;
- same-model dedicated replicas are representable;
- adding/removing deployments does not require recreating LiteLLM;
- route mutation becomes a reconciled data-plane operation;
- remote API/controller restarts need not restart the gateway.

The intended long-term direction is therefore:

> **Dynamic routing should become the normal server-mode route-management
> mechanism once its recovery and migration behavior meets the reliability
> bar.**

This plan does not by itself flip the existing local default. Local static mode
may remain useful as a minimum-dependency compatibility mode.

The important invariant is that controller loss does not remove existing
routes. Route retirement and upstream teardown must retain their existing safe
ordering: stop routing to an upstream before destroying the upstream.

Route semantic changes must remain generation-aware. A stale controller must
not republish an older route after a newer deployment contract has won.

---

## Remote control API

Introduce an `infer-stack server` (exact CLI spelling can be refined during
implementation).

The API should model infer-stack domain operations, not shell commands.

A possible resource shape:

```text
GET    /v1/status
GET    /v1/catalog
GET    /v1/nodes
GET    /v1/nodes/{id}
GET    /v1/leases
POST   /v1/leases
GET    /v1/leases/{id}
DELETE /v1/leases/{id}
GET    /v1/deployments
GET    /v1/deployments/{id}
POST   /v1/deployments/{id}/evict
GET    /v1/operations/{id}
GET    /v1/events
GET    /v1/audit
```

This is illustrative, not a frozen REST contract.

### Long-running operations

Acquiring a large model may take minutes. Do not hold the semantics of the
operation inside one HTTP connection.

A mutation should quickly return something like:

```json
{
  "lease_id": "lease-...",
  "operation_id": "op-...",
  "accepted_generation": 42
}
```

The CLI may then wait by polling or subscribing to operation/events, preserving
today's convenient blocking UX.

### Event streaming

The server needs a structured event stream for:

- operation progress;
- deployment readiness;
- node state changes;
- logs/status summaries.

Start with SSE or another simple server-to-client stream unless bidirectional
requirements justify WebSockets/gRPC.

Do not make log streaming the transport for state changes; logs are diagnostic,
operations/events are protocol.

### Local and remote CLI

The existing CLI should preserve the same conceptual verbs.

Local mode:

```text
infer-stack acquire qwen
    -> local controller/domain code
```

Remote mode:

```text
infer-stack acquire qwen
    -> authenticated server API
```

Avoid maintaining separate semantics in "CLI implementation" and "server
implementation." Extract/use one domain service layer, with the CLI becoming a
local or remote client.

---

## Security model

Remote infer-stack control is high impact. A compromise can consume expensive
accelerators, expose model credentials, alter routes, and—if container
configuration is insufficiently constrained—become host-root-equivalent.

Security requirements therefore shape the API from the beginning.

### Safe defaults

Server mode should default to loopback only:

```text
127.0.0.1
::1
```

A user must explicitly configure a remotely reachable listener.

Do not inherit the current optional reverse proxy's "trusted network, no
TLS/auth" posture for the control API.

### TLS

Remote control traffic requires TLS.

Reasonable first deployment choices:

- mTLS for controller-to-agent machine identity;
- mTLS or bearer/OIDC credentials for administrative clients.

Plain HTTP may be allowed only on loopback or a deliberately configured local
Unix socket.

### Human/client authentication

Longer term, OIDC is preferable for human/CI identities because it avoids
inventing account lifecycle management.

A simpler initial token/mTLS mechanism may be acceptable if:

- tokens are scoped;
- rotation is supported;
- secrets are never returned after creation;
- credentials are stored privately;
- audit records identify the principal.

### Authorization roles

At minimum distinguish:

**viewer**
- status;
- nodes;
- deployments;
- read-only logs/events.

**user**
- viewer permissions;
- acquire approved catalog endpoints;
- release own leases;
- wait/test/access own endpoints within policy.

**operator**
- release/evict others' workloads;
- drain/enable nodes;
- perform runtime operational actions.

**admin**
- mutate catalog/server policy;
- manage credentials;
- change backend/control-plane settings;
- perform privileged maintenance.

Exact RBAC should be resource/action based internally even if the first UI
exposes these four roles.

### Resource policy and quotas

Server mode should be able to constrain:

- endpoints a principal/project may acquire;
- total GPUs;
- replica count;
- TTL;
- dedicated deployments;
- priority classes if introduced later.

This is both security and denial-of-service protection.

### No arbitrary remote execution

Normal remote users must not be able to supply:

- arbitrary container images;
- arbitrary container commands;
- arbitrary host bind mounts;
- Docker socket mounts;
- privileged mode;
- arbitrary device mappings;
- arbitrary host environment injection.

Those capabilities make the API equivalent to remote root on a Docker host.

If administrative custom-runtime support is retained, it must be explicitly
privileged, policy checked, and auditable.

### Catalog trust boundary

Separate:

```text
"use endpoint already approved in catalog"
```

from:

```text
"change what that endpoint launches"
```

A Git-backed catalog is attractive because review/merge provides a natural
administrative control and an immutable revision.

Server-side catalog mutation may still exist, but it requires admin permission
and audit events.

### Secrets

The control API should prefer secret references over returning secret values.

Node agents receive only credentials required for deployments assigned to that
node, ideally scoped and short-lived where upstream systems permit.

Never make a "dump environment" endpoint part of the remote API.

### Network policy

Recommended topology:

- public/user-facing inference endpoint: LiteLLM/data-plane address;
- control API: private management network/VPN or strongly authenticated TLS;
- PostgreSQL: reachable only by controller/LiteLLM services that require it;
- node agents: preferably initiate outbound authenticated connections;
- Docker daemon: never exposed over the network for infer-stack.

### Auditability

Every privileged mutation should be attributable.

"Who launched this model/image and under which catalog revision?" must have a
durable answer.

---

## Compose node agents

Remote Compose control should not be implemented as controller SSH commands or
a network-exposed Docker socket.

Introduce an `infer-stack agent` process on each managed Compose host.

### Agent responsibilities

The agent owns host-local operations:

- report GPU inventory and stable device identities;
- report Docker/runtime capabilities;
- report local disk/cache pressure;
- realize assigned deployment generations;
- stop/replace managed deployments;
- run local readiness/health probes as requested;
- stream structured status and selected logs;
- report observed generation and error state.

### Agent non-responsibilities

The agent should not:

- make global placement decisions;
- allocate GPUs independently of controller authority;
- mutate catalog definitions;
- invent leases;
- route around authorization policy.

### Connection direction

Prefer an agent-initiated persistent authenticated connection:

```text
GPU node agent ---> control plane
```

Benefits:

- no inbound management port required on every GPU host;
- simpler NAT/firewall operation;
- centralized certificate/policy handling;
- controller messages travel over an already authenticated channel.

A pull/long-poll implementation is acceptable initially if a persistent channel
is premature.

### Agent identity

Each node agent gets a stable node identity and machine credential.

Enrollment must be explicit. A newly connected credential must not be allowed
to silently impersonate an existing node.

### Fencing at the agent

Commands include desired generation/fencing metadata.

Once the agent has accepted generation 42 for deployment X, a delayed
generation-41 command is rejected.

This protects against a stale controller continuing after leader failover.

### Compose remains host-local

This does **not** turn Compose into a new Kubernetes replacement.

The controller may choose among known Compose hosts using infer-stack's simple
resource model, but each agent still realizes ordinary local Docker resources.
Complex cluster scheduling remains Kubernetes territory.

---

## Kubernetes / KubeAI behavior

Kubernetes already has:

- node agents (kubelet);
- resource scheduling;
- restart/health controllers;
- desired-state APIs;
- identity/RBAC mechanisms.

Therefore normal KubeAI serving should not require infer-stack's Compose node
agent.

The infer-stack controller talks to the Kubernetes API/KubeAI resources and
maintains lease/front-door semantics.

A future optional host bootstrap agent could help with node onboarding, but it
must not become a second scheduler competing with Kubernetes.

For an HA infer-stack control plane, Kubernetes resource ownership also needs a
stable logical control-plane identity so a new controller replica continues the
same ownership rather than appearing to be an independent controller.

---

## Failure behavior

The system should specify failure semantics before implementation.

### All controller replicas down

Expected:

- existing inference continues;
- LiteLLM and model servers remain;
- no TTL GC;
- no new mutations;
- agents retain last accepted desired state and do not improvise.

Recovery:

- a controller reconnects to the state store;
- acquires reconciliation leadership;
- observes backends/agents;
- converges from durable desired state.

### State store unavailable

Controllers fail closed for mutations.

Do not accept acquires into process memory and hope to persist them later.

Existing inference continues.

Read-only cached status may be exposed with a clear stale/unavailable marker,
but must not masquerade as authoritative current state.

### Controller loses leadership mid-operation

A new leader receives a higher fencing epoch.

Backend/agent publication from the stale epoch cannot overwrite newer state.

Operations either resume idempotently or become superseded/failed with enough
state for the client to retry.

### Node agent disconnected

The controller marks the node unknown/unreachable after a timeout.

Do not immediately assume GPUs are free and schedule the same exclusive
resources elsewhere unless the new target is a different physical host.

Running workloads on the disconnected host may still serve traffic.

### Node rejoins

Agent reports its actual managed resources and applied generations.

Controller reconciles rather than blindly recreating everything.

Unknown/unmanaged containers remain outside infer-stack ownership.

### LiteLLM unavailable

Model servers may remain healthy but the normal front door is unavailable.

Controller operations should distinguish route/gateway failure from engine
failure.

Do not tear down healthy engines simply because the gateway is temporarily
unreachable unless policy explicitly requires it.

### PostgreSQL used by LiteLLM and infer-stack is unavailable

Existing route state in a running LiteLLM process and existing engines may
continue serving depending on LiteLLM behavior, but control mutations must stop.

This is one reason to treat database HA/backups as an operational concern for
server mode.

---

## Dynamic routing as the server-mode normal path

Static superset routing remains useful for today's lightweight local mode, but
it is not the best primitive for a distributed controller:

- per-deployment replicas are harder to represent;
- route lifecycle is tied to generated config;
- same-model dedicated replicas need independent upstream identities;
- remote reconciliation benefits from mutable route objects.

Therefore server mode should assume dynamic routing unless future LiteLLM
limitations prove otherwise.

Before changing the global default, validate:

1. clean bootstrap with an empty database;
2. restart recovery;
3. database backup/restore;
4. LiteLLM upgrade/migration behavior;
5. route drift repair;
6. same-model replica behavior;
7. failure during route replacement;
8. safe route retirement before upstream teardown;
9. no gateway-wide interruption on ordinary acquire/release.

Local static mode can remain as a no-Postgres fallback.

---

## Control-plane observability

Add explicit controller/state-store visibility rather than hiding distributed
behavior behind generic `status`.

A future command/API should expose something like:

```text
NAME       HEALTHY   API      RECONCILER   EPOCH   LAST_SEEN
ctrl-a     yes       active   leader       18      1s
ctrl-b     yes       active   standby      18      1s
ctrl-c     yes       active   standby      18      2s
```

Also expose:

- authoritative state-store health;
- reconciliation backlog;
- oldest unconverged generation;
- connected/offline node agents;
- route publication health;
- controller build/schema versions.

Do not infer "healthy control plane" solely from process liveness.

---

## Rolling upgrades and schema compatibility

HA only helps if replicas can be upgraded safely.

Server mode needs explicit schema-version rules:

- database migrations are versioned;
- older binaries refuse to run against an incompatible newer schema;
- rolling-compatible releases document the supported mixed-version window;
- only one migration actor runs at a time;
- migrations do not depend on every controller being stopped unless required.

Agent protocol messages also need a version/capability handshake so a controller
does not send unsupported commands to an older node agent.

The first implementation may require all controllers to run the same version;
that is acceptable if enforced clearly.

---

## Backup and disaster recovery

PostgreSQL makes the state durable but not magically safe.

Server-mode operations documentation should define backup/restore for:

- infer-stack control database;
- LiteLLM route database if separate;
- catalog/config source;
- CA/certificates or other machine credentials;
- secrets required to recreate routes/deployments.

A restored controller must reconcile with actual running resources rather than
assuming the backup's observed state is still true.

Observed/backend state is always re-discovered after restore.

---

## Migration from today's single-controller installation

Migration must be explicit and reversible.

### Standalone remains unchanged

Existing users can continue:

```text
CLI -> SQLite -> local Compose
```

No server or PostgreSQL requirement.

### Single remote server

First distributed step:

```text
remote CLI -> infer-stack server -> PostgreSQL -> local backend
```

Only one controller replica is required. This proves API/state semantics before
adding failover.

### Add node agents

Move Compose execution behind node agents:

```text
server -> state -> agent on GPU host -> Docker
```

The original server host may also run an agent.

### Add controller replicas

Once every mutation is transactional and reconciliation is generation-fenced,
add multiple API replicas and leader failover.

Do not add replicas before eliminating process-local authority from mutation
paths.

---

## Proposed implementation phases

### Phase 0 — finish the local controller invariants

Before remote/HA work:

- keep acquire/release transactional;
- keep route retirement ordered before upstream teardown;
- ensure dynamic route reconciliation is idempotent;
- ensure all desired mutations have stable identities;
- identify all remaining process-local or rendered-file authority.

Deliverable: a written inventory of state that must move behind the state-store
interface.

### Phase 1 — controller service boundary

Introduce a domain service/API boundary while preserving SQLite and local
execution.

Goals:

- CLI can call the same domain service directly;
- optional localhost `infer-stack server` exposes typed operations;
- long-running operations have durable/local operation identities;
- no arbitrary command endpoint.

This phase should be deployable on one workstation without PostgreSQL.

### Phase 2 — PostgreSQL state-store implementation

Add PostgreSQL as an alternative state store.

Acceptance:

- lease/catalog/deployment semantics match SQLite;
- concurrent acquire/release tests exercise real transactional contention;
- idempotency keys work;
- migrations are automated/tested;
- server can restart and reconstruct desired state.

At this stage one server process is still the supported reconciler.

### Phase 3 — secure remote mode

Make remote binding a supported configuration.

Required before calling it supported:

- TLS;
- authenticated principals;
- RBAC;
- private secret handling;
- audit events;
- rate/quota hooks;
- explicit remote-listener configuration;
- no unauthenticated 0.0.0.0 default.

### Phase 4 — Compose node agent

Implement agent enrollment, heartbeat, inventory, desired-generation apply, and
status reporting.

Acceptance:

- controller does not need the remote Docker socket;
- controller can acquire an endpoint on a remote Compose host;
- node disconnect/reconnect is safe;
- stale generation is rejected;
- unmanaged containers are untouched.

### Phase 5 — HA controller replicas

Add:

- active-active API replicas;
- reconciliation leader election;
- monotonically increasing leadership epoch;
- per-resource desired-generation fencing;
- controller heartbeat/status;
- failover tests that kill the current leader mid-operation.

Acceptance:

- killing the leader does not interrupt existing inference;
- another replica resumes reconciliation;
- stale leader actions cannot overwrite newer state;
- clients can retry timed-out mutations idempotently.

### Phase 6 — operational hardening

Add/document:

- database backup/restore;
- certificate rotation;
- controller/agent rolling upgrades;
- metrics and alerts;
- disaster-recovery acceptance tests;
- optional external secret manager integration.

---

## Required test strategy

Distributed behavior must be tested mechanistically rather than through timing
hope.

### State-store contract tests

Run the same suite against SQLite and PostgreSQL where semantics should match:

- acquire;
- release;
- TTL;
- coalescing;
- dedicated deployment;
- idempotent retries;
- concurrent writers;
- recovery after process restart.

### Reconciliation generation tests

Simulate:

1. leader A reads generation 10;
2. desired state advances to 11;
3. leader B converges 11;
4. A attempts to publish 10;
5. publication is rejected/no-op.

Cover routes, deployments, and agent commands.

### Failover tests

Kill the leader at each boundary:

- before backend action;
- after backend action but before observed-state write;
- during route publication;
- during release/eviction;
- during node-agent command.

Every operation must converge or end in an explicit retryable/terminal state.

### Security tests

Verify that a normal user cannot:

- mutate catalog;
- acquire a disallowed endpoint;
- exceed GPU quota;
- release another user's lease;
- specify arbitrary image/command/mount;
- retrieve secrets;
- impersonate a node agent.

### Data-plane independence test

Start endpoint and route, then stop every controller process.

A real generation through LiteLLM must still succeed.

Restart controller and verify it reconstructs/converges without recreating
healthy serving processes unnecessarily.

### Agent partition tests

Disconnect agent while its workload is running.

Verify:

- controller reports unknown/offline;
- it does not double-allocate that host's GPUs;
- inference may continue;
- reconnect reconciles actual state safely.

---

## Non-goals

This plan does **not** propose:

- writing a replacement for Kubernetes;
- implementing Raft/etcd inside infer-stack;
- building a generic remote shell;
- exposing Docker's TCP socket;
- allowing normal remote users arbitrary containers or host mounts;
- solving global multi-region scheduling;
- transparently merging two independent infer-stack databases;
- making SQLite a network-shared HA database;
- replacing Git or another configuration review process with an unreviewed API;
- requiring HA for ordinary workstation use.

---

## Open design questions

These need explicit decisions during implementation.

### API transport

REST + SSE is likely sufficient and easy to inspect. gRPC may be attractive for
the node-agent stream. Avoid choosing a transport before defining domain
semantics.

### Agent connection

Options:

- outbound WebSocket/gRPC stream;
- long poll;
- message broker.

Prefer the smallest design that preserves authentication, ordering, fencing,
and reconnect behavior.

### Authentication provider

Possible progression:

1. local Unix/loopback trust;
2. mTLS/API tokens for early remote deployments;
3. OIDC for human/CI identities.

Do not invent a full user/password database unless there is a compelling need.

### Catalog authority

Options:

- Git/file remains authoritative; server loads approved revision;
- server stores catalog revisions;
- hybrid: Git publishes signed/hashed revisions into control state.

Whichever wins, runtime objects must record immutable revision identity.

### One PostgreSQL service or two

Operationally one service is simpler; separate services reduce correlated
failure and privilege coupling.

Semantically, infer-stack and LiteLLM must still use independent
schemas/databases and credentials.

### Controller placement

Controller replicas may run:

- on dedicated management hosts;
- on Kubernetes;
- on GPU-serving hosts.

The design should not require a GPU host, and losing a GPU worker must not imply
losing the whole control plane.

### Server-mode dynamic routing default

The proposed answer is yes. The exact migration/default policy for existing
local installs needs a separate compatibility decision.

---

## Decisions this plan recommends now

Unless future evidence contradicts them:

1. **Model the system as one logical control plane with N controller replicas,
   not N independent control planes.**
2. **Keep SQLite for standalone mode; use PostgreSQL for remote/HA mode.**
3. **Do not use network-shared SQLite.**
4. **Start HA with active-active API replicas and one elected reconciler.**
5. **Use generation/fencing semantics in addition to leader locking.**
6. **Make existing inference independent of controller availability.**
7. **Use node agents for remote Compose hosts; never expose Docker remotely.**
8. **Do not require an infer-stack node agent for ordinary Kubernetes/KubeAI
   scheduling.**
9. **Make remote operations typed and catalog-driven; no general remote exec.**
10. **Default remote listeners to loopback and require strong authentication/TLS
    before external exposure.**
11. **Separate endpoint use permission from catalog mutation permission.**
12. **Treat dynamic LiteLLM routing as the intended normal mechanism for server
    mode once its recovery behavior is fully hardened.**
13. **Add controller/agent/audit observability before claiming HA support.**
14. **Build in stages: service boundary -> PostgreSQL -> secure remote mode ->
    node agents -> HA.**

---

## Success criteria

The architectural work is complete when all of the following are true:

- a user can point the ordinary infer-stack CLI at a remote control server and
  use the same acquire/release/status workflow;
- multiple authenticated clients can mutate one authoritative state safely;
- a Compose GPU host can be controlled through an authenticated node agent
  without exposing Docker;
- Kubernetes remains the scheduler for KubeAI workloads;
- at least two controller replicas can run concurrently;
- loss of the reconciliation leader causes automatic failover;
- a stale leader cannot overwrite a newer desired generation;
- loss/restart of every controller does not interrupt an already-running
  generation through LiteLLM;
- remote users cannot turn normal endpoint acquisition into arbitrary host code
  execution;
- every privileged mutation is attributable in an audit log;
- standalone SQLite/local mode remains simple and supported.

This provides a path from today's single-owner limitation to a real
model-serving control plane without sacrificing the low-friction workstation
workflow that makes infer-stack useful now.

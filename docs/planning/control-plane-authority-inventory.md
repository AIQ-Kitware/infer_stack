# Control-plane authority inventory

**Status:** current standalone implementation inventory, 2026-10-03

This document is the Phase 0 inventory for
[distributed-control-plane-and-remote-server.md](distributed-control-plane-and-remote-server.md).
It records which state is authoritative *today*, which files are merely
realizations/caches, and which local authorities have to move behind a durable
state-store or secret-store boundary before remote/multi-controller mode is
safe.

The point is not to make every datum live in one object. Some separations are
intentional and load-bearing. The rule is narrower:

> One semantic fact should have one authority. Other representations must be
> derived from it, observations of it, or explicitly different facts.

## Current authority map

| Concept | Current authority | Derived / observed copies | Server/HA direction |
| --- | --- | --- | --- |
| User catalog intent | `catalog.yaml` / explicitly supplied catalog sources | `Catalog`, `CatalogUnion`, TUI/CLI views | Keep catalog authority policy explicit; runtime records use an immutable catalog revision/digest |
| Catalog parsing / endpoint meaning | `leasing/catalog.py` | CLI/TUI presentation | Keep one domain parser/resolver; remote API must call the same semantics |
| Catalog file mutation | `leasing/catalog_edit.py` | CLI and TUI are clients | Server-side mutation, if enabled, becomes an admin operation against versioned catalog state |
| User settings intent | `settings.yaml` plus explicit invocation overrides | invocation profile | Keep local file mode; server mode stores/version-controls approved settings |
| Published render inputs | recovery profile stored in the ledger | backend objects configured by `use_profile()` | Durable control-plane state keyed by immutable config/catalog revision |
| Leases, demand, deployment records, allocations | SQLite ledger (`SqliteStore` / `Ledger`) | CLI/TUI/status views | State-store interface; PostgreSQL for server/HA |
| Desired-state change epoch | ledger `desired_gen` plus admission-state version | publication/reconcile callers | Evolve to per-object desired generations; do not infer generations from files |
| Standalone publication intent / crash recovery | ledger publication marker | rendered-file digest / controller transient state | Durable operation + per-object generation/fencing; publication marker is not itself an HA fence |
| Runtime existence/readiness | backend observation (Docker/Kubernetes/LiteLLM) | ledger observed summaries and UI | Observed-generation/state records updated by reconciler/agents; backend remains source for live reality |
| Gateway secret values | managed state-dir `.env` | Kubernetes Secret / container env | Dedicated secret storage/reference mechanism; never copy secrets into catalog/state event payloads |
| Remembered non-catalog gateway routes | `litellm_registry.json` | rendered route tables, live LiteLLM DB | Move durable route intent behind state store before multi-controller route reconciliation |
| Current dynamic route realization | LiteLLM model store/API | `litellm_routes.json` desired render | Desired route objects in control state; LiteLLM remains observed data-plane realization |
| Compose/KubeAI rendered manifests | **not authoritative** | `docker-compose.yml`, `models.yaml`, gateway manifests | Reconstruct from durable desired state + immutable revision |
| Render bookkeeping sidecars | **not authoritative for residency** | `leasing-*-state.json`, gateway state files | Replace needed fields with durable observed/applied-generation records; otherwise keep as local cache |
| GPU/runtime inventory | live host/Kubernetes observation | short-lived caches | Node-agent/node records with freshness timestamps; Kubernetes stays scheduling authority for KubeAI |
| VRAM measurements | `measurements.json` for locally learned sizing facts | planner/suggestion inputs | Optional durable measurement facts scoped to hardware/model/runtime identity; not desired-state authority |

## Deliberate separations that are not duplicate authorities

### User configuration vs recovery profile

`settings.yaml` and current catalog files describe what the user wants *now*.
The recovery profile freezes the non-ledger inputs that produced the currently
published standalone runtime epoch. Keeping those separate prevents a later
command, package upgrade, or edited catalog from silently re-rendering live
services with different inputs after a crash.

The distributed design should preserve the distinction as immutable
configuration/catalog revision identity attached to desired state. It should
not make the recovery profile another independently editable configuration.

### Desired state vs runtime residency

The ledger says what should exist. Docker/Kubernetes says what actually exists.
Reconciliation requires both. A rendered sidecar is neither one: it records
what infer-stack last rendered and may be stale after crashes or manual/runtime
changes.

### Structural identity vs capacity

A deployment's structural compatibility and its capacity are separate facts.
For example, a 262K process can satisfy a compatible 65K endpoint without
changing model identity. Collapsing those concepts would break safe
subsumption/coalescing.

### Endpoint contract vs upstream served name

The endpoint alias is the client contract. `served_name` is the upstream model
name used to coalesce/request a runtime. They may be equal, but neither is a
second spelling of the other. `public_name` is only a legacy spelling of
`served_name` and should be accepted only at compatibility boundaries.

### Published catalog routes vs remembered route registry

Published catalog routes are derived from the active catalog revision.
Remembered registry rows preserve routes whose deployment is no longer defined
by a published catalog (including historical/ad-hoc routes). They intentionally
have different provenance and precedence. The problem for HA is not that both
exist; it is that the latter is currently authoritative in a local JSON file.

## Local authorities that block multi-controller mode

These are the important Phase 0/1 migration targets.

### 1. SQLite is the single standalone state authority

This is correct for local mode. Do not make it multi-controller by placing the
file on shared storage. The future state-store boundary should expose domain
operations (lease mutation, demand/allocation changes, profile/revision state,
generation updates, operations), not generic SQL calls from controller code.

A store interface should be introduced when there are two implementations to
support (SQLite and PostgreSQL), not as a speculative wrapper around every
current method.

### 2. Process/file locks are standalone serialization, not distributed fencing

The publication lock and backend converge flock correctly serialize local
processes sharing a data root. They cannot establish leadership across
machines. Server/HA mode needs transactional generation checks plus a leadership
fencing epoch. File locks remain useful inside a node agent for host-local
realization.

### 3. Generation state must be strengthened, not removed

The current ledger-wide `desired_gen` records a monotonic desired-state epoch;
the publication marker records richer standalone render/apply crash-recovery
intent. Neither should be mistaken for a rendered-file version.

The HA design needs per-object desired/observed generations and a leadership
fencing epoch. Migration should therefore move from the coarse generation to
more precise generations while preserving monotonicity. It should **not** delete
generation state on the grounds that the standalone controller currently waits
on the publication marker instead.

### 4. The gateway route registry is durable local control state

`litellm_registry.json` is not just a cache: rows can preserve route meaning
that no currently published catalog defines. A second controller with a
different local copy could make a different routing decision.

Before multiple controllers reconcile routes, registry entries (or their
successor route objects) must be transactional state-store records with
identity, desired generation, provenance, and catalog revision where
applicable. `litellm_routes.json` remains a render artifact and the LiteLLM DB
remains a data-plane realization to observe/reconcile.

### 5. Managed secrets are durable local state

The gateway `.env` is intentionally the local authority for generated master
keys, DB credentials, WebUI secrets, and referenced external-route keys. It is
not safe to move these values into ordinary operation records or catalog JSON.
Remote mode needs a secret-store/reference boundary and scoped access; node and
gateway realization should receive only the secrets they require.

### 6. Backend sidecars contain useful recovery bookkeeping, not live truth

Compose/KubeAI state JSON records hashes, names, request mappings, degraded
render outcomes, and similar realization metadata. Those files must never win
over live backend observation for residency. Any field needed to fail over to a
different reconciler or node agent should become a durable observed/applied
state field; the remaining sidecar can stay a reconstructable local cache.

## Authority boundaries tightened by the Phase 0 cleanup

The accompanying cleanup deliberately makes today's boundaries easier to carry
into the server design without introducing the server architecture early:

- `leasing/catalog.py` remains the one definition of catalog semantics.
- `leasing/catalog_edit.py` is the one editable-YAML persistence path; CLI and
  TUI no longer have independent writers.
- `served_name` is the one canonical writable spelling; `public_name` is
  read-only compatibility input.
- `ResolvedEndpoint` owns endpoint-target resolution while `EndpointRequest`
  owns managed deployment demand. Repository code crosses that boundary
  explicitly through `to_request()`; the older read-only properties remain only
  as derived Python-API compatibility views.
- `Gateway` / `ClusterGateway` own front-door keys and route-registry
  operations. `ComposeBackend.front_door()` returns that authority instead of
  re-exporting a second operational gateway API.
- vLLM/Ollama structural identity is defined by the executable structural
  builders, not mirrored field-name constants.
- real-generation readiness is unconditional controller policy; the deprecated
  CLI compatibility flag no longer becomes inert backend state.

These changes are intentionally modest. They remove false authorities without
adding a generic service locator, repository layer, command bus, or backend
factory hierarchy.

### Deliberately deferred normalization

Two areas still contain repeated *representations* but are poor candidates for
a broad refactor before the server-state schema is designed:

- `EndpointRequest` derives structural identity, capacity, render spec, and
  per-alias served contracts from the same catalog endpoint. Those views have
  different semantics and persistence consequences. A future normalized
  endpoint object may usefully derive them, but changing that data model now
  would churn ledger/profile serialization immediately before per-object
  generations and catalog revision identity are introduced.
- CLI status and the TUI both interpret backend observations. The generic
  deployment condition should continue converging on `Controller.observe_state`,
  but Kubernetes Model declaration, pod residency, replica readiness, and the
  TUI's visibility-gated cluster polling are not one fact. Do not hide those
  distinctions behind a generic view-model hierarchy merely to remove branches.

## Phase 0 exit checklist

Before beginning the remote server or PostgreSQL work, verify that:

- [ ] every persistent local file under the leasing state directory is listed
      above as authority, secret state, observation/cache, or render artifact;
- [ ] no CLI or TUI path implements catalog semantics independently of the
      catalog domain/persistence helpers;
- [ ] runtime status does not treat a rendered sidecar as residency truth;
- [ ] dynamic route add/replace/delete remains idempotent and route retirement
      precedes upstream teardown;
- [ ] every desired-state mutation that will become remotely callable has a
      stable mutation/object identity suitable for idempotency records;
- [ ] the future state-store interface is specified in domain operations before
      a PostgreSQL implementation is added;
- [ ] per-object generation/fencing semantics are designed before more than one
      reconciler can act;
- [ ] catalog revision and secret-reference behavior are explicit in the server
      API design.

# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 4**
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** **plan for review. No code written.**
- **History:** revision 1 `5244229` · revision 2 `7be95ba` (addendum `90d4ca4`) ·
  revision 3 `3aeb5c0`. Each revision's §0 records the review round it answers;
  this one answers the review of revision 3.
- **Code baseline:** `dev/0.7.1` at `e751676`. Leasing code is unchanged since; line
  references were re-checked for this revision.
- **Evidence:**
  - [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
    (C1-C7, R1-R5, A-G, and the round-4 generation race);
  - [`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`](investigation-gateway-stale-upstream-after-recreate-2026-09-16.md)
    (M1-M4, **now reproduced on the host**).

---

## 0. Changes from revision 3

The reviewer accepted §4.5's admission boundary and `current` as the sole
authority, and judged P1 design-ready. It did **not** accept P2, because the
publication protocol covered only admission, while many other mutators change
desired state outside any protocol. Every factual claim below was re-verified
against the code.

| # | finding | re-verified | disposition |
|---|---|---|---|
| 1 | Crash-safe activation covered only admission. `release`, `evict`, `gc`, rollback and `sweep` all commit before any render (`controller.py:791-792`, `810`, `831`, `628-644`; `ledger.py:249`). The CLI release-all writes the ledger directly (`commands_leasing.py:1025-1033`). Eight CLI commands call a mutating `sweep()` (`943`, `1025`, `1079`, `1241`, `1362`, `1767`, `2121`, `2243`). The TUI sweeps and releases directly (`tui.py:1100`, `1187`, `2465`, `2476`, `2480`). | yes | **Accepted, with a different mechanism for non-admission mutations** (§4.2): see "Two mutation classes" below. Read-only paths stop mutating (I17); every mutator goes through the controller (I16) |
| 2 | §4.8 was wrong about KubeAI: it writes a mutable models file and sidecar at render (`backends/kubeai.py:353-354`) and reads both back at apply to prune (`383-384`), the same race as Compose. | yes | **Accepted.** KubeAI gets immutable generation bundles; only NullBackend needs none (§4.8) |
| 3 | Migration must rebase `applied_gen`. A pre-existing `meta.applied_gen` of, say, 67 would make every waiter on G=1 believe it is applied. | yes (`store.py:208-229`) | **Accepted.** Atomic rebase specified (§5) |
| 4 | D14's bump rule was too broad: `set_applied_generation` is a mutation made outside `_global_lock` during apply (`controller.py:516-523`), so it could invalidate an admission waiting at approval. | yes | **Accepted.** `admission_state_version` bumps only on changes that can affect the admission/render snapshot (§4.5) |
| 5 | The route registry must be generation-relative: `prepare(G+1)` must build on `current`'s registry, not the stable runtime copy (`_load_route_registry`, `_update_route_registry`). `routes prune` and `routes seed` write the registry directly (`commands_leasing.py:2266-2278`, `2336-2345`). | yes | **Accepted.** The base is `current`'s snapshot; prune and seed publish a generation through the controller (§4.12) |
| 6 | Dynamic route reconciliation is best-effort: it returns if the gateway is unreachable and logs failed POSTs (`compose.py:2115-2127`, `2185-2209`). G must not be marked applied if it failed. | yes | **Accepted.** Reconciliation returns success, the final route set is verified, and failure leaves G unapplied (I18) |
| 7 | D12: stable per-service IPs is the right direction, but needs a design layer. There is no explicit network today (`render_compose` returns only `{'name', 'services'}`, `compose.py:1275-1277`). A low DNS TTL is a mitigation, not I14. The upstream-direct check needs an execution location, because models publish no host port behind the gateway (`compose.py:361-366`). | yes | **Accepted.** An explicit network with IPAM, an append-only address table, an operator-run migration of a running stack, and an in-network probe (§4.7). Host reproduction since confirms the reasoning (below) |
| 8 | A missing or corrupt **activated** bundle must fail closed. Today an unreadable compose file is tolerated (`compose.py:2038-2049`). | yes | **Accepted** (I9) |
| 9 | §4.6 computed `departing` without excluding DEGRADED deployments, contradicting its own "never removed" rule. | — | **Accepted.** Fixed in the pseudocode |
| D13 | Agreed: re-admit. Make the fast/slow split explicit, and let an EXPIRED lease lose the race. | — | **Accepted** (§4.10) |
| D15 | Agreed: release-only repair. | — | **Resolved** |
| P1 | Design-ready; run V1, V2 and V8 before calling it complete. | — | Recorded (§8) |
| P2 | Not ready. The reviewer proposed splitting it into P2a (pure overlays for every mutator, plus durable stage and atomic commit) and P2b (bundles). | — | **Split accepted; overlay-for-every-mutator not adopted** (see below and D16) |

### Two mutation classes, and why only one needs a preview (D16)

The reviewer's rule offers two options: each mutation either commits an activation
atomically with itself, *or* leaves a durable "unpublished state" marker from which
recovery builds the next generation. This revision uses **both, split by mutation
class**:

- **Admission-class mutations** (acquire, renew re-admission) can **fail**: a
  candidate may be unplaceable or unrenderable. They must preview before
  committing (§4.5), so a failed attempt commits nothing.
- **Removal-class mutations** (release, sweep, evict, gc, rollback) only **remove**
  demand, and removing demand cannot make a publishable desired state
  unpublishable:
  - with fewer required deployments, every remaining hard allocation stays valid;
  - collisions resolve to the oldest name-holder, so removing that holder only makes
    the next one renderable.

  These commit first, setting a durable **publication-pending** marker in the same
  transaction, and recovery publishes from the current ledger.

The consequence for ordering: **P2 does not depend on the overlay machinery.**
- **P2a** is the pending marker for every mutator, plus routing every caller through
  the controller.
- **P2b** is the immutable bundles.
- Until P5, today's acquire keeps committing first and simply sets the marker too.
  That is no worse than today, and I1's "no ACTIVE lease for a failed attempt" holds
  from P5 on.

The reviewer should confirm the monotonicity argument (D16).

### Additions by this author in this round

- **The misroute was reproduced on the host, and its mechanism is now observed.**
  Two stop-policy models were run through the gateway at default settings. A was
  removed; B was created and got A's old IP (`172.18.0.3`); A was recreated at
  `.0.4`.
  - All **330** gateway answers for A over ~11 minutes were
    ``404 The model `A` does not exist``.
  - During **186** of those, a fresh lookup from inside the gateway resolved A to
    `.0.4`, and A served its model correctly there.
  - During the other **144**, A had been removed and its name no longer resolved,
    and the gateway **still** returned B's 404.

  So Docker's embedded DNS is correct, and the stale state is the gateway client's
  pinned keep-alive connection. This confirms that a low DNS TTL is only a
  mitigation, and that stable addresses (§4.7) are the fix.
- **Where the direct check runs:** the pinned gateway image has `python3` and no
  `curl` (verified). The check runs **inside the gateway container**, which gives a
  fresh resolution in the gateway's own network namespace without adding a port.
- **The pending marker is compare-and-clear.** A publication clears it only if no
  newer mutation arrived while it was rendering; otherwise the newer mutation's
  marker survives.
- **Static IPs create a new orphan conflict.** An orphan container that still holds
  a static address would block its service's recreation. The orphan-blocking rule
  (I10) is extended to address conflicts (§4.7).
- **Network migration of a running stack recreates every container once**,
  including the gateway. It is therefore an explicit, operator-invoked command, not
  something that happens on first use.

---

## 1. Problem

A request can wait out its admission timeout while every GPU is empty:

- idle keep-warm deployments share a hard desired set with live demand (R1);
- the planner ignores demand (C3);
- unapplied renders persist pins (C4) and publish partial projects (R3);
- the intended pressure release was never implemented (C2).

Underneath that:

- apply is not bound to its render (the round-4 race);
- **many independent code paths mutate desired state outside any publication
  protocol** (§0, finding 1);
- recreating containers can pin the gateway's traffic for one model onto another
  model's container (M1-M4, reproduced).

## 2. Goals and non-goals

**Goals**

1. Idle keep-warm residency never prevents admission.
2. Admission is lease-atomic, including renderability.
3. **Every** desired-state change is published crash-safely, whichever caller made
   it.
4. An apply acts on exactly one published generation, including its auxiliary
   files and routes. It records that generation as applied only once that is fully
   true.
5. No GPU is handed to a container, including a same-deployment replacement, while
   a previous occupant is still there.
6. Nothing destructive is decided from a failed or ambiguous observation.
7. Degraded deployments and orphans are isolated.
8. Traffic for a deployment reaches only that deployment's container.

**Non-goals**

- FIFO or reserved admission (G);
- a durable `PENDING` lease state;
- preemption;
- automatic re-pinning or migration;
- automatic bundle GC;
- multi-node placement;
- changing `observe()`'s contract.

---

## 3. Invariants

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease is admitted, and its state is published or recoverably pending. From P5 on, a failed attempt has no lease. |
| **I2** | A `LIVE` deployment holds a **hard committed allocation**, never re-placed automatically; an invalid allocation makes it DEGRADED. No path creates a LIVE deployment without one. |
| **I3** | An `IDLE` keep-warm deployment is optional, and a candidate only while **resident** (`running`, `restarting`, `paused`). infer-stack never creates its container. |
| **I4** | Admission is lease-atomic: every candidate deployment is placed and renderable, or nothing changes. |
| **I5** | A waiting attempt writes nothing on its own behalf, except discardable pre-commit staged bundles. |
| **I6** | Admission prepares and approves outside any SQLite write transaction, and commits briefly after validating `admission_state_version`, all under one `_global_lock` hold. |
| **I7** | Every published generation contains every admitted LIVE deployment; DEGRADED deployments are excluded from runtime actions. |
| **I8** | Physical residency comes from a strict, project-scoped Docker snapshot. A failure is **unknown**; duplicates are **ambiguous** and fail closed. |
| **I9** | `current` is the single published authority. apply(G) materialises G's auxiliary files and routes, acts only on G, and marks only G applied. A **missing or corrupt activated bundle fails closed**. Publication never writes runtime files. |
| **I10** | **Barrier:** before a container starts on a GPU, every other container occupying that GPU (including the same deployment's old container) has been stopped by id and its exit confirmed. An orphan occupying the GPU, **or holding the service's static address**, blocks the apply. |
| **I11** | **Selective apply:** services are classified from G's rendered config; departing containers come from the previously applied manifest and **never include DEGRADED deployments**; unknown labelled containers are orphans; no project-wide `--remove-orphans`. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`. |
| **I13** | Under unknown or ambiguous residency, only resource-neutral admission proceeds. |
| **I14** | A request routed for deployment D reaches only D's container. |
| **I15** | **Recovery:** every `_global_lock` entry and every apply first completes or compensates in-flight activations, and publishes any pending state. |
| **I16** | **Every desired-state mutator goes through the controller**, and either previews and commits an activation (admission class) or commits a compare-and-clear pending marker (removal class). No CLI, TUI or backend command writes desired state around the controller. |
| **I17** | **Read-only paths do not mutate.** `status`, `leases` and TUI polling compute TTL expiry virtually; sweeping is a controller maintenance operation that publishes. |
| **I18** | **A generation is applied only if every part of it succeeded,** including dynamic route reconciliation verified against the gateway's final route set. |

---

## 4. Design

### 4.1 Strict residency → I3, I8

Unchanged from revision 3. `Residency` keeps every matching container per
deployment, is scoped to the project label plus `infer-stack.deployment`, reads GPUs
from `HostConfig.DeviceRequests`, and treats `running`, `restarting` and `paused` as
resident. Occupants include every non-removed state. It raises `ResidencyUnknown` on
any Docker error. `observe()` is unchanged.

### 4.2 Publication protocol, bundles, recovery → I1, I9, I15, I16

**Store** (new rows in `meta` or small tables):

- `generation_counter`;
- `applied_generation`, rebased at migration (§5);
- `admission_state_version` (§4.5);
- `publication_pending(version)`: the `admission_state_version` of the newest
  unpublished mutation, or null;
- `activation(G, staged_id, lease_ids, rendered_version, created_at)`: at most one
  row.

**Admission class (acquire, renew re-admission):** preview, stage, approve, then a
short transaction that commits the overlay, bumps `admission_state_version`, and
inserts `activation(G, staged_id, lease_ids, rendered_version=new version)`.
Activation follows (§4.5).

**Removal class (release, sweep, evict, gc, rollback):** a single transaction applies
the mutation, bumps `admission_state_version` to *v*, and sets
`publication_pending = v`. Then, still under `_global_lock`, call
`publish_pending()`:

1. snapshot the ledger (version *v′* ≥ *v*);
2. `prepare()` and `stage()`. Monotonicity means this cannot fail on placement or
   renderability; any other failure leaves the marker set, and recovery retries;
3. short transaction: allocate *G* and insert
   `activation(G, staged_id, lease_ids=[], rendered_version=v′)`;
4. activate.

**Activate** (idempotent):

1. rename `staged-<uuid>` to `gen-<G>` (skipped if `gen-<G>` exists), then fsync
   the directory;
2. atomically replace `current` with *G*, unless `current` ≥ *G*;
3. short transaction: delete the activation row, and **compare-and-clear** the
   marker. Set `publication_pending = null` only if it is ≤ `rendered_version`.

**Recovery (I15)**, at every `_global_lock` entry and every apply:

1. If an activation row exists and its staged directory or `gen-<G>` exists: finish
   activation.
2. If an activation row exists and neither exists: compensate. Release its
   `lease_ids` (a removal-class mutation, so it sets the marker) and delete the row.
3. Delete `staged-*` directories not named by any activation row.
4. If `publication_pending` is set: `publish_pending()`.

**Bundle contents** (Compose): `docker-compose.yml`, `manifest.json`, and `aux/`
holding `litellm_config.yaml`, `nginx.conf`, `routes.json` and
`route_registry.json`. Paths inside the compose file point at stable runtime paths
(D9).

**apply(G)**, under the apply lock:

1. Run recovery; `G := current`.
2. **Fail closed** if `gen-<G>` or its manifest is missing, unreadable, or does not
   validate: do not advance `applied_generation`, and surface the error.
3. Materialise `aux/*` to the stable runtime paths.
4. Barrier and selective start (§4.6).
5. Reconcile dynamic routes from `aux/routes.json`, **and verify** the gateway's
   managed route set equals it. On failure, stop without advancing (I18); a retry
   heals.
6. Set `applied_generation = G`; `gen-<G>` becomes the previously applied manifest.

**Retention:** keep every bundle (D8). **Waiters** wait for
`applied_generation >= G`.

### 4.3 Planner → I2, I12

Unchanged from revision 3 (`required_ids`, `hard`, `optional_hints`; hard
allocations degrade rather than re-place; optional residents are never newly fit).

### 4.4 Ledger → I2

Unchanged from revision 3: `assigned_gpus` is set at admission commit and cleared
with any LIVE→IDLE change. IDLE→LIVE reuse adopts a unique resident container's
GPUs. The desired set splits into required and optional.

### 4.5 Admission → I1, I4-I6, I13, I15

As revision 3, with the token renamed and narrowed:

```text
with _global_lock():
    recover()
    maintenance_sweep()                      # removal class: commits with marker; publishes
    res  = residency() or UNKNOWN
    snap = ledger.snapshot()                 # carries admission_state_version
    cand = ledger.overlay_acquire(snap, ...) # pure core shared with Ledger.acquire
    ... resource-neutrality check (I13), prepare (in memory), stage, approve ...
    with store.transaction():
        if ledger.admission_state_version() != snap.version: rollback; discard staged; retry
        G = ledger.commit_overlay(...)       # bumps admission_state_version
        ledger.record_activation(G, staged.id, cand.lease_ids, rendered_version=...)
    activate(G)
_ensure_applied(G)
```

**`admission_state_version`** is bumped by changes that can affect an
admission/render snapshot:

- lease state transitions that change demand (ACTIVE→RELEASED or EXPIRED);
- claim insert or delete;
- deployment create, state, spec or `served` changes;
- `assigned_gpus` changes;
- reservation ownership;
- route registry changes (§4.12).

It is **not** bumped by:

- `applied_generation`;
- activation-row housekeeping;
- the pending marker itself;
- TTL-only renewals of a lease whose deployments are all LIVE;
- history pruning that cannot affect demand.

Secrets are provisioned outside admission; `prepare()` reads them and fails closed.
`prepare()` also detects collisions and builds the next route registry on **`current`'s
snapshot** (§4.12).

### 4.6 Selective apply and the barrier → I7, I10, I11

```text
prev = manifest(applied_generation)      # generation 0 from migration adoption (§5)
m    = manifest(G); res = residency()    # unknown -> abort; G stays unapplied
classify every service in m by config hash: unchanged | changed | new
    # models, gateway, postgres, open-webui, reverse-proxy
departing = (containers owned by prev that m drops or displaces)
            - (containers of deployments in m.degraded)          # I11
orphans   = labelled containers owned by neither prev nor m      # report only

for gpu in gpus(services in m to start or recreate):
    blockers = res.occupants(gpu) - {containers m keeps unchanged on that gpu}
    if any orphan in blockers: abort("orphan <id> occupies GPU <gpu>; run gc --orphans")
    for c in blockers: docker stop c; wait for exit (bounded, else abort)
for svc to start with a static address (§4.7):
    if an orphan holds that address: abort("orphan <id> holds <ip>; run gc --orphans")
for c in departing: docker rm -f c.container_id
start  = [s in m.services : class(s) in {new, changed}
          and s.deployment not in m.degraded and not optional(s)]
start += new-or-changed dependencies of start
docker compose -p P -f gen-G/docker-compose.yml up -d --no-deps <start, dependency order>
```

### 4.7 Gateway routing correctness → I14 (D12)

**Observed mechanism** (host reproduction, §0): the gateway pins a keep-alive
connection to a reused IP; Docker's DNS is correct throughout. So:

- **Mitigation, not the fix:** set `AIOHTTP_TTL_DNS_CACHE` low on the gateway
  service. This narrows the window in which a stale address can open the pinned
  connection, but cannot close it.
- **Fix: stable per-service addresses.**
  1. **Explicit network.** The rendered project declares a named network
     (`infer-stack`) with an IPAM subnet from settings, default `172.30.0.0/16`,
     configurable to avoid host conflicts. Every service attaches with an
     `ipv4_address`.
  2. **Append-only address table** in the ledger: `service_addresses(service_name,
     ipv4, assigned_at)`. A service name keeps its address forever. An address is
     never reassigned to a different name, even while the service is absent. The
     gateway, Postgres, UI and proxy also get fixed addresses. Allocation is
     sequential from the subnet, and exhaustion is an explicit error, never reuse.
  3. **Address conflicts:** an orphan container holding a service's address blocks
     that service's start (I10).
  4. **Migration of a running stack** is an explicit, operator-invoked
     `infer-stack network migrate`. It assigns addresses to all existing service
     names, renders a generation that uses the explicit network, and applies it,
     **recreating every container once, including the gateway**. It is never
     triggered implicitly. It is refused while any lease is ACTIVE, unless forced.
- **Upstream-direct check (detection):** in addition to the gateway probe,
  `docker exec <gateway container> python3 -c '<GET http://<service>:8000/v1/models>'`.
  This is a fresh resolution in the gateway's own network namespace, using the
  image's `python3` (verified present; `curl` is absent). The check compares the
  served names there with what the gateway returns for the alias.
  - If the upstream serves the expected model and the gateway returns
    404 "does not exist": **routing fault**, reported as such.
  - If the upstream is not listening: **not ready**.

### 4.8 Backends → D5

- **Compose:** everything above.
- **KubeAI:** immutable generation bundles containing `models.yaml` plus the
  manifest that replaces its sidecar, with `current` and apply-exactly-G. Its prune
  step reads the previously applied manifest, never a mutable sidecar. No residency,
  barrier or network work: it has no host GPUs.
- **Null:** the publication protocol reduces to the ledger. No bundles.

### 4.9 Rollback and readiness timeout

A readiness timeout (`controller.py:778`) and `_rollback_acquire` go through the
removal class (§4.2), with their "never ran" decisions taken from strict residency.
Nothing is evicted under unknown or ambiguous residency.

### 4.10 Renew → I2 (D13)

- **Fast path:** the lease is ACTIVE, and every deployment is LIVE with a valid hard
  allocation. Update TTL and heartbeat, lock-free, with no generation and no version
  bump.
- **Slow path:** any deployment is IDLE. Do not extend or reactivate inline. Enter
  §4.5 under `_global_lock`, **re-validate that the lease is still ACTIVE**, then
  atomically adopt or place, and renew.
  - If the lease has become **EXPIRED** first (a sweep won the race), renewal fails:
    `renew: lease expired, re-acquire`.
  - If adoption or placement is impossible, renewal fails:
    `renew: deployment reclaimed, re-acquire`.
- The CLI `renew` calls the controller, never `controller.ledger.renew`.

### 4.11 Observability

`status` shows:

- UNKNOWN, AMBIGUOUS, NOT RUNNING, DEGRADED, DISPLACED and ORPHAN;
- `current`, `applied_generation`, any activation, and `publication_pending`;
- **virtual TTL expiry**, computed rather than swept (I17);
- routing faults from the upstream-direct check;
- the service→address table.

Waiting and timeout messages name the contested GPU and its holder.

### 4.12 Route registry and route commands → I16

- **Generation-relative:** the route registry that `prepare()` extends is the one in
  `current`'s bundle, not the stable runtime copy. The runtime copy is materialised
  from the bundle by apply.
- **`routes prune`** and **`routes seed`** become controller operations that
  bump `admission_state_version`, then prepare, stage, commit an activation, and
  activate a new generation carrying the changed registry.
  - `routes prune`: `commands_leasing.py:2266-2278` today writes the registry
    directly.
  - `routes seed`: `2336-2345` today calls `merge_route_registry` directly.

### 4.13 Callers → I16, I17

- **CLI release-all and single release:** call `Controller.release`, never
  `controller.ledger.release` (`commands_leasing.py:1025-1033`).
- **TUI releases:** call `Controller.release` (`tui.py:2465`, `2480`).
- **CLI read commands** (`943`, `1079`, `1241`, `1362`, `1767`, `2121`, `2243`)
  and **TUI polling** (`tui.py:1100`, `1187`): replace `sweep()` with a read-only
  view that marks leases past their TTL as expired **virtually**.
- **Sweeping** happens only inside controller operations under `_global_lock`
  (§4.5's `maintenance_sweep`), and through an explicit `infer-stack gc`.

---

## 5. Migration

1. **Schema:** add `assigned_gpus`, `generation_counter`, `admission_state_version`,
   `publication_pending`, `activation`, and `service_addresses`.
2. **Generation rebase (atomic, one transaction):**
   - read the legacy `meta.desired_gen` and `meta.applied_gen`;
   - record them as `legacy_desired_gen` and `legacy_applied_gen`, for diagnostics
     only;
   - set `generation_counter = 1`, `applied_generation = 0`, and
     `admission_state_version = 1`;
   - after this, nothing reads the legacy counters.
3. **Secrets:** provision any missing secrets.
4. **Backfill hard allocations from strict residency only.** A LIVE model deployment
   with exactly one resident container adopts its GPUs. One with none or ambiguous
   containers stays **unresolved**. A LIVE `reserved-gpu` deployment stays
   **unresolved**. While anything is unresolved, new GPU allocation is refused;
   releases and sweeps proceed.
5. **Adoption:** generation 0 is a manifest built from the backfilled LIVE
   deployments and the containers that match them by label and GPU. It is written as
   `gen-0` and becomes the previously applied manifest. Everything else labelled is
   an orphan.
6. **First publication** is `gen-1`. The old top-level compose file and sidecar are
   retired after it applies.
7. **Addresses** are not assigned here. That is `infer-stack network migrate`,
   operator-invoked (§4.7).

---

## 6. Tests

Tests 1-50 are carried from revision 3, with their numbering kept; the changed ones
are noted. **R** = reviewer, **A** = author. New tests follow.

**Placement:** 1-7, unchanged.

**Admission:** 8-20, unchanged, except:

- 18: approval does not hold the SQLite lock, **and** a concurrent
  `set_applied_generation` does not invalidate the admission (**R**).
- 19: only `admission_state_version` changes cause a retry.

**Generations and recovery:** 21-28, unchanged.

**Apply, barrier, orphans:** 29-38, unchanged, except:

- 34: a DEGRADED deployment is never in `departing` (**R**).

**Residency, migration, renew, routing:** 39-50, unchanged, except:

- 46-47: include the slow path losing to a sweep that EXPIRED the lease (**R**).

**New in revision 4**

51. **R:** a removal-class crash window: the process is killed after a `release`
    commit and before publication. Recovery sees `publication_pending` and publishes
    a generation without the released deployment. Repeat for `sweep`, `evict`, `gc`
    and rollback.
52. **A:** compare-and-clear: a second mutation lands while the first is being
    published. The marker survives, and the next recovery publishes the second.
53. **R:** CLI release-all and TUI multi-release go through `Controller.release`.
    No caller writes desired state around the controller (grep test plus behaviour
    test).
54. **R:** `status`, `leases` and TUI polling perform no SQLite writes (read-only
    connection or write-count assertion). Expired-by-TTL leases are displayed as
    expired.
55. **R:** the KubeAI generation race: *G+1* rendered while *G* applies. *G* applies
    and prunes according to *G*'s manifest.
56. **R:** migration from a database with `desired_gen = 80` and `applied_gen = 67`.
    Afterwards `applied_generation = 0`, and a waiter on G=1 does not believe it is
    applied.
57. **R:** the route registry is generation-relative: *G* adds route A and has not
    applied yet; *G+1* is prepared and published; *G+1*'s registry still contains A.
58. **R:** `routes prune` and `routes seed` publish a new generation and never
    write the runtime registry directly.
59. **R:** a dynamic route reconciliation failure (gateway unreachable, or a failed
    POST) leaves `applied_generation` unchanged; the retry succeeds and advances it.
60. **R:** a missing or corrupt `gen-<current>` fails closed and does not advance.
61. **A:** stable addresses: a service removed and recreated gets the same address;
    no other service ever receives it.
62. **A:** an orphan holding a service's static address blocks that service's start
    and names `gc --orphans`.
63. **A:** `network migrate` on a running stack assigns addresses, recreates every
    container once, and is refused while leases are ACTIVE unless forced.
64. **A:** the in-network upstream check distinguishes routing fault, not ready,
    and healthy. Integration test against the gateway image.
65. **A:** the host reproduction (the gateway investigation's V3) re-run with stable
    addresses: zero misrouted samples.

---

## 7. Host verification before implementation

- **V1:** `HostConfig.DeviceRequests[].DeviceIDs` carries the rendered GPU indices.
- **V2:** a `docker ps -a` scoped to project and label lists every model container.
- **V3:** whether `up --remove-orphans` can start before an orphan frees its GPU.
- **V4:** `-p <project> -f <other path>` manages the same containers.
- **V5:** `up -d --no-deps` with an unchanged stanza does not recreate.
- **V6:** crashed-container states under `unless-stopped`.
- **V7: done.** The misroute is reproduced with IP reuse observed.
- **V8:** a `paused` container keeps its GPU memory.
- **V9:** `docker exec <gateway> python3 -c ...` can reach `http://<service>:8000`.
- **V10:** static `ipv4_address` semantics. Does a **stopped** (not removed)
  container keep its address reserved, so that starting another container with it
  conflicts? This is what makes rule I10's address clause necessary.
- **V11:** `network migrate` on a scratch copy of a running stack, measuring
  gateway downtime.
- **V12:** re-run the reproduction with `AIOHTTP_TTL_DNS_CACHE` lowered, to
  quantify the mitigation. Whether to restart the gateway for it is the operator's
  call.

---

## 8. Implementation order

A step lands only once every invariant it relies on holds.

| step | content | behaviour |
|---|---|---|
| **P1** | Strict `residency()` (§4.1); tests 39-41. **Design-ready**; complete after V1, V2, V8 | additive |
| **P2a** | Publication protocol (§4.2 store, removal class, `publish_pending`, compare-and-clear, recovery); every caller through the controller; read-only paths stop mutating (§4.13); today's acquire sets the marker too; tests 51-54 | every mutation publishes crash-safely |
| **P2b** | Immutable bundles, `current`, activation, apply-exactly-G with fail-closed and route verification (§4.2); generation-relative route registry and route commands (§4.12); KubeAI bundles (§4.8); migration rebase and adoption (§5, steps 1-3, 5-6); tests 21-28, 55-60 | fixes the round-4 race, the auxiliary race and the KubeAI race |
| **P3** | Planner keywords (§4.3); tests 1-7 | none by default |
| **P4** | Ledger `assigned_gpus` and backfill (§5, step 4); renew fast/slow paths (§4.10); tests 42-47 | allocations become hard |
| **P5** | Admission preview, stage/approve/commit (§4.5); acquire leaves the removal class; tests 8-20 | a failed attempt commits nothing |
| **P5b** | Explicit network, address table, `network migrate`, in-network check (§4.7); tests 48-49, 61-65 | I14 |
| **P6** | Selective apply, barrier, orphans, `gc --orphans` (§4.6); tests 29-38 | no `--remove-orphans` |
| **P7** | Idle keep-warm becomes optional and resident-only | **fixes the incident** |
| **P8** | DEGRADED end to end, readiness-timeout path, observability; test 50 | isolation and visibility |

---

## 9. Decisions

**Resolved**

| id | decision |
|---|---|
| D1 | Overlay admission with a pure coalescing core and a short validating commit |
| D2 | Ledger `assigned_gpus`, as a hard claim |
| D3 | Selective apply |
| D4 | Resource-neutral admission only under unknown or ambiguous residency |
| D5 | Compose and KubeAI get bundles; Null does not |
| D6 | No automatic re-pinning |
| D7 | DEGRADED is derived |
| D8 | No automatic bundle GC |
| D9 | Stable runtime paths with immutable per-generation contents |
| D10 | Warm residency is `running`, `restarting`, `paused` |
| D11 | Explicit `gc --orphans` |
| D12 | Stable per-service addresses from an append-only table, an explicit network, operator-run migration, and an in-network upstream check. A low DNS TTL is a mitigation only |
| D13 | Re-admit through the slow path; an EXPIRED lease loses |
| D14 | `admission_state_version`, with the narrowed bump rule (§4.5) |
| D15 | Release-only repair for unresolved reservations |

**Open, for the next review**

| id | question | author's lean |
|---|---|---|
| **D16** | Removal-class mutations commit first with a compare-and-clear pending marker, rather than every mutator getting a pure overlay. Is the monotonicity argument sound, meaning no removal-only mutation can make a publishable state unpublishable? | Yes. It also decouples P2 from the overlay machinery |
| **D17** | While the gateway is unreachable, a generation cannot become applied (I18), so waiters block until their timeout. Acceptable, or should an unreachable gateway in static-superset mode be treated as "no routes to reconcile"? | Acceptable in dynamic mode, where routes are part of the generation. In static-superset mode there is nothing to reconcile, so it should not gate |
| **D18** | Default subnet and conflict handling for the explicit network: a fixed default with a settings override, or auto-detection of a free range? | Fixed default with override. Auto-detection is its own source of churn |

---

## 10. Deferred

- FIFO or reserved admission (G).
- Automatic re-warming of displaced keep-warm deployments.
- Explicit migration of LIVE deployments between GPUs.
- Automatic bundle GC.
- Any change to `observe()`.
- Whether an ACTIVE lease past its TTL but unswept counts as demand. The virtual
  expiry in I17 changes only display, not demand.

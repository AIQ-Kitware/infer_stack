# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 7a** (consolidated and self-contained; §0.1 records the review of revision 7)
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** plan for review. **P1 is implemented** on `dev/0.7.1` (`675315c`);
  nothing else is written.
- **The central decision in this revision, D22 (serialised publication), is
  adopted provisionally.** It is confirmed by host measurement V15 (§7). If V15
  shows serialisation is too slow for real per-shard churn, revision 6 (`be6c89d`)
  describes the bundle/activation alternative, and only §4.3 changes.
- **History:** revisions 1-6 are `5244229`, `7be95ba`, `3aeb5c0`, `f0bf2f1`,
  `e7242f0`, `be6c89d`. Each §0 there records one review round.
- **Evidence:**
  - [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
  - [`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`](investigation-gateway-stale-upstream-after-recreate-2026-09-16.md)
    (reproduced on the host)
- **Out of scope** is documented for users and contributors in
  [`docs/planning/known-limitations.md`](../../docs/planning/known-limitations.md#leasing-scope-boundaries-and-known-faults).

---

## 0. Changes from revision 6

The reviewer agreed that **if render and apply are serialised under
`_global_lock`, the immutable-generation machinery buys no correctness.** With
serialisation, the compose-file race, the auxiliary-file race, the KubeAI race,
and every crash window are closed by a durable pending marker plus a
deterministic re-render. That machinery existed only to let renders advance while
an older render was being applied.

**Dropped:**

- immutable `gen-<G>` bundles and the `current` pointer;
- activation rows and the generation counter;
- `applied_generation` and coalesced waiters;
- reconstruction from a base generation, and the `repair_required` state machine;
- generation-relative route registries;
- KubeAI bundles;
- bundle retention rules;
- ownership by "key appears in a retained bundle".

**Kept, in simpler form:**

- the frozen profile and `config publish`;
- deterministic rendering;
- behavioural fingerprints and strict residency;
- the explicit Compose environment;
- verified route reconciliation;
- migration and adoption;
- a small approved-digest guard.

| # | finding (all code claims re-verified) | disposition |
|---|---|---|
| 1 | Serialisation removes the need for bundles; a small approved digest still guards a crash between an approved commit and its apply (e.g. across a binary upgrade). | **Adopted provisionally** (D22, §4.3) |
| 2 | Without bundles, ownership cannot be "key in a retained bundle". Define `managed` by infer-stack's own labels plus migration adoption; forged labels are already out of scope. | **Adopted** (I9, §4.7) |
| 3 | **`acquire --no-apply`** stages a lease without starting it (`commands_leasing.py:594`, `757-760`, `801-819`), and `infer-stack render` never applies. Recovery must not start a staged lease just because the controller reopened. | **Adopted:** `publication_pending.apply_requested` (§4.3) |
| 4 | Route reconciliation must gate clearing the marker. Today it is best-effort (`compose.py`, `_reconcile_routes`). Static-superset mode does not gate (D17). | **Adopted** (I5) |
| 5 | V15 must measure the **whole lock hold**, not only `up -d`. Dynamic routing waits for Postgres health (`depends_on … service_healthy`, `compose.py:792`), and route discovery retries up to 90 × 2 s (`_reconcile_routes(attempts=90, delay=2.0)`). | **Adopted** (V15 expanded) |
| 6 | `_default_docker_run` is `subprocess.check_output` with **no timeout**. A hung Docker command would hold the host-wide lock forever. | **Adopted:** bounded backend operations (I16, §4.11) |

**Refinements by this author:**

- **A timeout is not an abort.** Killing the `docker compose` client does not stop
  work the Docker daemon has already started. After a timeout, the marker stays
  set, and the next apply begins from a **strict residency snapshot**. Target
  services still in a transitional state (`created`, `restarting`, `removing`)
  make it retry rather than act (§4.11).
- **Route verification inside a host-wide lock must fail fast.** A 180 s retry
  budget while the gateway is unreachable would block every release. Inside the
  lock, verification makes a short, bounded attempt and otherwise leaves the
  marker set for the next retry. The long wait applies only to a fresh gateway
  bootstrap, which is rare and explicit (§4.11, D24).
- **Staged-lease promotion is today's behaviour, made explicit.** A later ordinary
  acquire or release applies the whole pending desired state, including a lease
  staged with `--no-apply`. The current code does the same: any apply brings up the
  whole rendered set. It is recorded, not changed (D23).

### 0.1 Amendments after the review of revision 7

The reviewer accepted revision 7's architecture and asked for no broad rewrite.
It answered D23 and D24, and found two **implementation-order** issues. Its code
claim (`_reconcile_routes` listing timeout 10 s, `compose.py:2162-2166`; route
POST timeout 30 s, `2180-2203`) was re-verified. Changes:

| # | finding | amendment |
|---|---|---|
| 1 | **P2's crash determinism depended on P4's profile.** A process could commit a mutation, die, be upgraded or reconfigured, and recover to a different render. `approved_digest` only covers approved operations, not ordinary releases. | The **minimum immutable render context moves into P2**: the initial profile is persisted once by P2's migration, P2 renders only from it, and P2 uses the explicit Compose environment. P4 keeps `config publish`, drift reporting, image pre-pull and profile-changing workflows (§8) |
| 2 | **P2 referred to selective apply, which lands in P8.** | P2 serialises **today's backend apply** under `_global_lock`, bounded, with route verification gating the marker. P8 replaces the apply algorithm. Test 21 moves to P8 (§4.3, §8) |
| 3 | **D24: a retry count does not bound the lock hold,** because each listing can take 10 s and each POST 30 s. | D24 is an **end-to-end wall-clock deadline** covering GETs, POSTs, verification and sleeps: small in steady state, larger for explicit bootstraps. Values come from V15 (§4.11) |
| 4 | **A stable `created` container would wedge recovery.** A Compose client that died between create and start leaves a container that never changes state on its own, and waiting for it retries forever. | The P8 interrupted-apply rules are replaced with a **quiescence window** and per-state rules (§4.11) |
| 5 | Test 19 ("a mutation landing during publish is not lost") assumes concurrency that D22 forbids. | Replaced with "no desired-state writer bypasses `_global_lock` or controller publication". The marker keeps its version only as defensive bookkeeping |
| D23 | Keep current staged-lease behaviour. | **Resolved**, and documented as a current limitation in `docs/planning/known-limitations.md` |

---

## 1. Problem

A request can wait out its admission timeout while every GPU is empty.

1. **Idle keep-warm deployments are in the same hard desired set as live
   demand,** and the planner orders them by creation time rather than demand.
2. **Nothing releases idle residency under pressure.**
3. **Renders that were never applied persist GPU pins** and publish partial
   projects that another process may apply.
4. **Apply is not bound to the render it was meant for:** an applier can apply an
   old file and record a newer desired state as covered.
5. **Many code paths change desired state around the controller,** and render
   inputs (catalog, settings, image pins) drift between commands.
6. **Separately:** when a container is recreated and another takes its IP, the
   gateway can keep a pooled connection to that IP and send one model's traffic to
   another indefinitely.

## 2. Goals

1. Idle keep-warm residency never prevents admission.
2. Admission is lease-atomic, including renderability.
3. Every desired-state change survives a crash and is applied exactly as rendered.
4. infer-stack never mistakes its own containers for orphans, and never touches
   containers it cannot prove it owns.
5. No GPU is handed to a container while a previous occupant is still there.
6. Nothing destructive is decided from a failed or ambiguous observation.
7. Traffic for a deployment reaches only that deployment's container.
8. No infer-stack operation holds the host-wide lock without bound.

Non-goals are in §10 and the known-limitations document.

---

## 3. Invariants

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease is admitted, and its desired state is applied or durably pending. From P6 on, a failed admission attempt leaves no lease. |
| **I2** | A `LIVE` deployment holds a **hard committed allocation**. It is never re-placed automatically; if the allocation becomes invalid, the deployment is DEGRADED. Nothing creates a LIVE deployment without one. |
| **I3** | An `IDLE` keep-warm deployment is **optional**: a placement candidate only while its container is resident (`running`, `restarting`, `paused`). infer-stack never creates its container. |
| **I4** | **Admission is lease-atomic:** every candidate deployment is placed **and renderable**, or nothing is committed. |
| **I5** | **Serialised publication.** Every desired-state mutation runs under one `_global_lock` hold: commit the mutation plus `publication_pending`; render deterministically from ledger and published profile; if `apply_requested`, apply **that exact render**; clear the marker only after the **entire** apply succeeds. In dynamic-routing mode that includes verified route reconciliation. |
| **I6** | **Recovery** at every `_global_lock` entry: if a marker is set, re-render. Apply only if `apply_requested`. If `approved_digest` is set and the re-render's digest differs, **fail closed** until an explicit, approved `infer-stack apply`. |
| **I7** | **Frozen profile.** Non-ledger render inputs (including the catalog digest and resolved image pins) change only through an explicit, preview-first `config publish`. The catalog is configuration-time state: acquiring an endpoint absent from the published profile fails. |
| **I8** | **Strict residency** comes from a project-scoped Docker snapshot. A Docker failure means **unknown**; several containers for one deployment or wanted key means **ambiguous**. Both fail closed. |
| **I9** | **Ownership by label.** A container is `managed` if it was adopted at migration, or it carries this project's label plus `infer-stack.service` and `infer-stack.fingerprint`. It `satisfies` the desired state if its (service, fingerprint) key is wanted, it is the only container with that key, and its state serves. Unmanaged project containers are orphans: reported, never removed implicitly. |
| **I10** | **Barrier:** nothing starts on a GPU while any other container occupies it. An unmanaged occupant, or an unmanaged holder of the service's static address, blocks the apply. |
| **I11** | **Selective apply:** keep what satisfies; remove managed containers that do not, **except those of DEGRADED deployments**; start only what is missing; no project-wide `--remove-orphans`. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`. |
| **I13** | Under unknown or ambiguous residency, only resource-neutral admission proceeds. |
| **I14** | A request routed for deployment D reaches only D's container. |
| **I15** | **Every desired-state mutator goes through the controller, and read-only paths do not write.** |
| **I16** | **Bounded operations:** every backend call made under `_global_lock` has a timeout. On timeout the marker stays set, the lock is released, and the next apply starts from strict residency. |

---

## 4. Design

### 4.1 Strict residency → I3, I8, I9 (P1 done; extended in P8)

**Done in P1** (`infer_stack/leasing/residency.py`, `ComposeBackend.residency()`):

- deployment-labelled containers in the project, in every state;
- every match kept, never collapsed;
- GPUs read from `HostConfig.DeviceRequests`;
- an unmappable reservation treated as every GPU;
- `ResidencyUnknown` on any failure.

**P8 extension:** list every container carrying the project label, and record
each one's `infer-stack.service` (falling back to `com.docker.compose.service`)
and `infer-stack.fingerprint`. Ownership and the adoption of infrastructure
services (Postgres, LiteLLM, Open WebUI, reverse proxy) need both.

### 4.2 Profile and `config publish` → I7

A **profile** is stored in the ledger. It holds every non-ledger render input:

- backend kind and Compose project;
- LiteLLM, UI, dynamic routing, reverse proxy settings and ports;
- **resolved image pins**;
- the **catalog digest**, plus the endpoint definitions and route basis derived
  from the catalog;
- network configuration (§4.8);
- for KubeAI: namespace, base URL and resource profile.

Its behaviour:

- **Renders use the published profile,** never settings resolved by the current
  process. Commands compare their resolved settings with the published profile
  and **warn** on drift, pointing at `config publish`.
- **`infer-stack config publish`** previews the new profile's render, takes
  approval, then commits the profile and publishes through §4.3 with
  `approved_digest` set. It also **pre-pulls every image** the profile references,
  so steady-state applies do not pull.
- **Acquire resolves endpoints only from the published profile.** An endpoint that
  is missing fails with "publish the new configuration first".
- **Explicit Compose environment:** Compose is invoked with an environment built
  from the managed `.env` plus a fixed allow-list, never the caller's shell. Shell
  variables otherwise take precedence over `--env-file` for `${HF_TOKEN:-}`, the DB
  password and the master key.

### 4.3 Serialised publication → I1, I5, I6, I15, I16

**Store:**

- `profile`: the frozen render context. **P2 persists the initial profile** (§5,
  step 3) and renders only from it; P4 adds `config publish` to change it;
- `publication_pending(version, apply_requested, approved_digest)`: at most one
  row;
- `admission_state_version` (§4.6);
- legacy `desired_gen` and `applied_gen` are left unused (§5).

**Publishing a mutation** (every controller mutator):

```text
with _global_lock():
    recover()                                      # below
    with store.transaction():                      # short
        apply the ledger mutation; bump admission_state_version if relevant
        upsert publication_pending:
            version          = admission_state_version
            apply_requested  = apply_requested OR <this call applies>
            approved_digest  = <digest approved by this call, if any>
    publish()
publish():
    render = backend.render(ledger, profile)        # deterministic; writes the mutable files
    if marker.approved_digest and digest(render) != marker.approved_digest:
        fail closed: "rendered state differs from what was approved; run `infer-stack apply`"
    if not marker.apply_requested: return           # staged; marker stays
    backend.apply(render)                           # P2: today's apply, serialised and bounded (§4.11)
                                                    # P8: replaced by selective apply (§4.7)
    if dynamic routing: verify routes (bounded)     # failure -> return; marker stays
    with store.transaction():
        delete publication_pending          # its version is defensive bookkeeping only;
                                            # D22 forbids a concurrent desired-state writer
```

**Mutator behaviour:**

| operation | `apply_requested` | notes |
|---|---|---|
| acquire, release, sweep, evict, gc, rollback, renew slow path | set true | applies the whole pending state |
| `acquire --no-apply` | left false if no apply was requested | staged; the marker stays |
| `infer-stack render` | unchanged | writes the render only; never applies |
| `infer-stack apply` | set true, then publish | applies the pending state, including staged leases |
| `config publish` | set true, with `approved_digest` | preview-first (§4.2) |

- **Promotion (D23):** once any operation sets `apply_requested`, the next publish
  applies the **whole** pending desired state, including leases staged with
  `--no-apply`. That is today's behaviour, and is documented rather than changed.
- **`recover()`** at every lock entry: if a marker exists and `apply_requested` is
  true, `publish()`. If it is false, only verify that the render on disk matches
  the ledger, and re-render if not; never apply.
- **Crash windows, all covered by the marker plus a deterministic re-render:**
  - before render;
  - after render, before apply;
  - during apply (the next apply starts from strict residency);
  - after apply, before clearing: a redundant idempotent apply.
- **Waiting:** acquire's readiness wait stays **outside** the lock, as today.
  `_ensure_applied`'s coalescing is removed; the caller's own publish has already
  applied.

### 4.4 Planner → I2, I12

```python
plan_placement(deployments, inventory, *,
               required_ids, hard, optional_hints, allowed_gpus, ...)
```

1. **Hard allocations** are validated against the full physical pool. An invalid
   one becomes `degraded` and is never re-placed.
2. **Required deployments without an allocation** are placed by explicit
   placement, then by fit, within `allowed_gpus`. `(n_eligible, created_at, id)` is
   only a tie-break.
3. **Optional residents** keep their physical GPUs where free; otherwise they are
   `displaced`. They are never newly fit.

With the new keywords omitted, behaviour is identical to today.

### 4.5 Ledger allocations and renew → I2

- **`assigned_gpus`:** a nullable column on `deployments`. It is set in the
  admission commit, and cleared in the same transaction as any LIVE→IDLE change.
- **IDLE→LIVE reuse** adopts the unique resident container's physical GPUs. It is
  placed fresh if not resident. Under unknown or ambiguous residency it is not
  resource-neutral (I13).
- **Desired set:** required = LIVE, with their allocations. Optional = IDLE
  keep-warm deployments that are uniquely resident.
- **Renew fast path:** an ACTIVE lease with every deployment LIVE and allocated
  updates TTL and heartbeat only. It is lock-free, bumps no version, and publishes
  nothing.
- **Renew slow path:** any deployment IDLE → admission (§4.6) under the lock, first
  re-validating that the lease is still ACTIVE. If it is EXPIRED, or adoption and
  placement are impossible, renew fails explicitly. The CLI `renew` goes through the
  controller.

### 4.6 Admission → I1, I4, I13

```text
with _global_lock():
    recover()
    maintenance sweep (a mutation; publishes)
    res  = residency() or UNKNOWN
    snap = ledger.snapshot()                           # includes admission_state_version
    cand = overlay_acquire(snap, owner, requests)      # pure core shared with Ledger.acquire
    if residency unknown/ambiguous for cand and not resource_neutral(cand): not admissible
    prep = backend.prepare(snap + cand, res, profile)  # in memory: placement + render + collisions
    if any cand deployment unplaced or unrenderable: not admissible
    if interactive approval needed: approve(prep)      # no SQLite write lock held
    with store.transaction():
        if admission_state_version changed: rollback; retry
        commit overlay + allocations; bump admission_state_version
        set publication_pending(apply_requested per §4.3, approved_digest = digest(prep) if approved)
    publish()
readiness wait (outside the lock)
```

- **`overlay_acquire`** is one pure coalescing core (compat key, capacity,
  sharing, IDLE→LIVE reuse, served-alias merge). `Ledger.acquire` and admission
  both apply its intended mutations. There is one copy of the rules and no write
  transaction during preview.
- **`admission_state_version` bumps on** demand-changing lease transitions,
  claims, deployment create, state, spec and served changes, `assigned_gpus`,
  reservation ownership, profile publication, and route registry changes.
- **It does not bump on** the marker itself, TTL-only renewals, or history
  pruning.
- **Not admissible:** a queued caller sleeps and retries; a non-queued caller
  raises, naming the blocking admitted demand or the residency state.

### 4.7 Ownership, selective apply, barrier → I8-I11

**Labels on every rendered service:**

- `infer-stack.service=<name>`;
- `infer-stack.fingerprint=sha256(canonical stanza excluding this label ‖ sha256 of each generated file the service consumes)`.

The gateway's fingerprint includes its `litellm_config.yaml`, and nginx's includes
`nginx.conf`. That generalises today's content-hash label on the gateway and proxy
(`compose.py:845-858`, `1030-1045`).

The fingerprint changes only when behaviour changes. Verified on a real daemon: an
unchanged label value keeps a container `Running`; a changed one recreates it. A
per-generation label would therefore recreate everything on every apply.

**Predicates:**

- `managed(c)`: `c.id` is in `adopted_containers`, **or** `c` carries this
  project's label and both `infer-stack.service` and `infer-stack.fingerprint`.
- `wanted(c)`: `(c.service, c.fingerprint)` is in the current render, and `c`'s
  deployment is not displaced.
- `satisfies(c)`: `wanted(c)`, **and** `c` is the only container with that key,
  **and** its state serves:
  - `running`: satisfies;
  - `paused`: a required service is unpaused; an optional one stays paused;
  - `restarting`: left to Docker;
  - `created`, `exited`, `dead`, `removing`: do not satisfy.

```text
res = residency()                                   # unknown -> abort; marker stays
if any wanted key has >1 container: abort (ambiguous)
if any target service is in a transitional state from an interrupted apply: abort; retry (§4.11)
keep      = [c : satisfies(c)]
departing = [c : managed(c) and not satisfies(c) and c.deployment not DEGRADED]
orphans   = [c : not managed(c)]                    # report
to_start  = [svc in render : not DEGRADED, not optional, no kept container has its key]
for gpu in gpus(to_start):
    blockers = res.occupants(gpu) - keep
    if any unmanaged blocker: abort("unmanaged container <id> occupies GPU <gpu>; see gc --orphans")
    for b in blockers: docker stop b; wait for exit (bounded)
for svc in to_start with a static address: if an unmanaged container holds it: abort
for c in departing: docker rm -f c.id
unpause required paused kept containers
docker compose -p P -f <compose file> up -d --no-deps <to_start + new/changed deps, dependency order>
```

`infer-stack gc --orphans` is the only way to remove orphans: explicit, strict
snapshot, with approval or `--yes`.

### 4.8 Routing correctness → I14

The mechanism was reproduced on the host. The gateway pins a pooled connection to
a reused IP, while Docker DNS stays correct.

- **Fix: stable per-service addresses.**
  - **Explicit network:** the rendered project declares a named network with an
    IPAM subnet.
  - **Persisted subnet:** it is persisted in `network_config` at the first
    allocation, and in the profile. A later settings mismatch is rejected: changing
    it requires `network migrate --subnet`.
  - **Append-only address table:** `service_addresses(service, ipv4)`. An address
    is never reassigned to another name. Allocation is sequential, skipping the
    network, gateway, broadcast and Docker-reserved addresses. Before the first
    allocation, a preflight rejects subnets that overlap existing Docker networks
    or host routes.
  - **Blocking holders:** an unmanaged container holding a service's address
    blocks that service (I10).
  - **`infer-stack network migrate`** is operator-invoked. It assigns addresses to
    every existing service and recreates every container once, including the
    gateway. It is refused while leases are ACTIVE, unless forced.
- **Detection:** an upstream check via
  `docker exec <gateway> python3 -c '<GET http://<service>:8000/v1/models>'`. The
  image has `python3` and no `curl`. It resolves fresh inside the gateway's
  network, and a mismatch with the gateway's answer is reported as a **routing
  fault**, distinct from "not ready".
- **Mitigation only:** a low `AIOHTTP_TTL_DNS_CACHE` on the gateway.

### 4.9 Route registry → I15

The route registry has **one controller-owned authoritative copy**, changed only
under `_global_lock`.

`routes prune` and `routes seed` become controller mutations that publish through
§4.3. Today `commands_leasing.py:2266-2278` writes the registry directly and
`2336-2345` merges into it directly.

### 4.10 Callers and read paths → I15

- **Releases:** the CLI release-all (`commands_leasing.py:1025-1033`) and TUI
  releases (`tui.py:2465`, `2480`) call `Controller.release`.
- **Read-only commands** (`commands_leasing.py:943`, `1079`, `1241`, `1362`,
  `1767`, `2121`, `2243`) and **TUI polling** (`tui.py:1100`, `1187`) stop calling
  `sweep()`. They show TTL expiry **virtually**.
- **Sweeping** happens only inside controller operations, and through `gc`.

### 4.11 Bounded operations and interrupted applies → I16

**Timeouts (P2).** Docker calls under the lock get per-operation timeouts: short
for `ps` and `inspect`, moderate for `stop` and `rm`, longer but bounded for
`up -d` (D25). On timeout:

1. kill the process group;
2. leave the marker set;
3. release the lock;
4. fail the operation, saying the apply is pending.

**Route reconciliation deadline (P2, D24).** Dynamic-mode reconciliation runs
against **one end-to-end wall-clock deadline**, covering listing GETs, POSTs,
verification and retry sleeps. A retry count alone cannot bound it: today a
listing can take 10 s and each POST 30 s.

- Steady state: a small deadline.
- Explicit bootstraps (`config publish`, `network migrate`, the first apply against
  an empty project): a larger one.
- On expiry, the marker stays set and the operation returns.
- Static-superset mode does not reconcile routes (D17).
- Values come from V15.

**Interrupted applies (P8, with selective apply).** A killed Compose client does
not stop work the Docker daemon already started. A request may still complete
just after recovery takes its first look. So when recovering an
`apply_requested` marker:

1. **Quiescence window.** Take strict residency snapshots until the project's
   container set and states are identical for two consecutive samples, or a
   bounded settle deadline expires. Act only on a settled view. If the deadline
   expires unsettled, fail and leave the marker set.
2. **Per-state rules** on the settled view:
   - **`removing`:** wait for the container to disappear, within the settle
     deadline. If it does not, fail and leave the marker set.
   - **`created`:** managed but not satisfying. Nothing will advance it on its own.
     **Resume** it by starting the desired Compose service, or remove and recreate
     it if its key is no longer wanted. **Never wait for it to change
     spontaneously.**
   - **`restarting`:** left to Docker's restart policy, as §4.7 says. It still
     occupies its GPU for the barrier. Readiness, outside the lock, decides whether
     the workload succeeds.
   - **Duplicate wanted realisations:** fail closed, as always.

### 4.12 Backends

- **Compose:** everything above.
- **KubeAI:** serialised render and apply of its mutable models file and prune
  under the lock, with bounded `kubectl` calls. No residency, barrier or
  addresses.
- **Null:** the ledger and marker only.

### 4.13 Observability

`status` shows:

- the pending marker (staged or requested, plus approved-digest state);
- UNKNOWN, AMBIGUOUS, NOT RUNNING, DEGRADED, DISPLACED and ORPHAN;
- profile drift;
- routing faults;
- the address table;
- virtual TTL expiry.

Waiting and timeout messages name the contested GPU and its holder.

---

## 5. Migration

One-time, explicit, on upgrade:

1. **Schema:** `assigned_gpus`, `admission_state_version`, `publication_pending`,
   `profile`, `adopted_containers`, `service_addresses`, `network_config`.
2. **Legacy generation counters** (`meta.desired_gen`, `meta.applied_gen`) are
   left in place for diagnostics and never read again.
3. **Initial profile** is resolved **once** from the current settings and catalog.
   Secrets are provisioned if missing.
4. **Hard allocations** are backfilled from strict residency only:
   - a LIVE model deployment with exactly one resident container adopts its GPUs;
   - one with no container, or ambiguous containers, stays **unresolved**;
   - a **LIVE reservation** (which runs no container) stays **unresolved**.

   While anything is unresolved, new GPU allocation is refused; releases proceed.
   Operator repair is release-only (D15).
5. **Adoption:** every project container is recorded in `adopted_containers` if it
   matches a backfilled LIVE deployment (by label and GPUs) or an infrastructure
   service in the initial render. Adopted containers are managed until their
   service changes; the first recreation stamps the new labels. **Migration
   recreates nothing.**
6. **Addresses** are not assigned here: `network migrate` is a separate, explicit
   step.

---

## 6. Tests

**Placement and allocations**

1. The incident: an old idle resident against new live demand on a full host;
   live demand is placed and the idle deployment displaced.
2. A pinned idle resident loses its GPU to live demand.
3. A non-resident idle keep-warm deployment is absent from the plan.
4. IDLE→LIVE reuse adopts the resident's GPUs.
5. A hard allocation outside `allowed_gpus` stays valid.
6. An invalid hard allocation is DEGRADED and not re-placed.
7. With the new keywords omitted, plans are identical to today's.

**Serialised publication and recovery**

8. Crash **before render**: recovery re-renders and applies.
9. Crash **after render, before apply**: recovery re-renders and applies.
10. Crash **after apply, before clearing**: a redundant apply, then clear.
11. Crash **mid-apply**: the next apply starts from residency and completes.
12. `acquire --no-apply`, then reopen the controller: **nothing starts**, and the
    marker stays.
13. `infer-stack render` never applies.
14. A later ordinary release applies the pending state, **including** a staged
    lease (D23).
15. `infer-stack apply` applies the pending state and clears the marker.
16. Dynamic mode: a route verification failure leaves the marker set; a retry
    clears it.
17. Static-superset mode: an unreachable gateway does not gate clearing.
18. Approved digest mismatch (simulated renderer change between commit and apply):
    fails closed until an explicit approved `apply`.
19. No desired-state writer bypasses `_global_lock` or controller publication
    (behaviour test plus a check that no module outside the controller writes
    desired state).

**Bounded operations**

20. A hung `docker compose up` times out: the process group is killed, the marker
    stays, and the lock is released.
21. **(P8)** An interrupted apply is recovered on a **settled** view: a request
    that completes just after the first snapshot is seen; a stable `created`
    container is resumed or replaced, never waited on; `removing` is awaited within
    the deadline; `restarting` is left to Docker.
22. Route reconciliation under the lock honours its **total** deadline while the
    gateway is unreachable **and** while POSTs are slow, including several route
    changes; the lock is never held for the bootstrap deadline outside a bootstrap.

**Profile**

23. With local settings or the catalog changed, a release renders from the
    published profile and no unrelated service changes.
24. A package upgrade changing `PINNED_IMAGES` changes no image until
    `config publish`.
25. Acquiring an endpoint absent from the published profile fails, naming
    `config publish`.
26. `config publish` is preview-first; an unrenderable profile commits nothing.
27. With a conflicting `HF_TOKEN` exported in the caller's shell, the applied
    container matches the managed environment.

**Admission**

28. A non-queued request blocked only by warm cache is admitted.
29. A queued request blocked by admitted demand: failed retries commit nothing.
30. An unrelated request that fits is admitted while another waits.
31. Two processes contend for the last GPU: exactly one is admitted.
32. A placed but unrenderable candidate never creates an ACTIVE lease.
33. Approval waiting for a human does not hold the SQLite write lock, and a
    concurrent fast-path renew succeeds.
34. An `admission_state_version` change between prepare and commit causes a retry.
35. Unknown or ambiguous residency admits only resource-neutral requests.
36. Renew fast path is lock-free; the slow path re-admits; the slow path loses to
    an EXPIRED lease.

**Residency, ownership, apply**

37. P1 residency: state, GPU, scoping and ambiguity rules (**done**,
    `tests/test_leasing_residency.py`).
38. The residency extension lists infrastructure containers with service and
    fingerprint.
39. G starts B and route verification fails; a later render dropping B removes it
    as managed, never as an orphan.
40. A crash mid-apply leaves labelled containers that are managed, kept if wanted,
    and removed otherwise.
41. A gateway config content change with an identical stanza changes the
    fingerprint and recreates the gateway; identical content does not.
42. A managed `exited` or `dead` container with the wanted key is replaced.
43. Two containers with the same wanted key fail closed.
44. A required `paused` container is unpaused; an optional one stays paused.
45. Barrier ordering: a fake runner rejects starting onto an occupied GPU; this
    includes **same-deployment recreation**.
46. An unmanaged container on a target GPU, or holding a target address, blocks
    the apply.
47. A DEGRADED deployment is never started or removed, and a release elsewhere
    still applies.
48. A fresh dynamic bootstrap starts Postgres before the gateway.
49. `gc --orphans` removes exactly the reported containers, with approval.

**Callers, migration, routing**

50. CLI and TUI releases go through `Controller.release`; nothing writes desired
    state around the controller.
51. `status`, `leases` and TUI polling perform no SQLite writes.
52. `routes prune` and `routes seed` publish through the controller.
53. Migration adopts infrastructure and model containers; nothing is recreated.
54. A LIVE reservation at migration stays unresolved and blocks new allocation;
    releasing it resolves.
55. A stale sidecar is never used as an allocation.
56. The subnet is persisted; a settings mismatch is rejected; reserved addresses
    are skipped; an overlapping subnet fails preflight.
57. A recreated service keeps its address; no other service ever receives it.
58. `network migrate` recreates each container once and is refused while leases
    are ACTIVE.
59. The upstream check distinguishes routing fault, not ready and healthy.
60. The host reproduction re-run with stable addresses shows zero misrouted
    samples.
61. KubeAI: serialised render and apply under the lock; no concurrent render
    overwrites a file mid-apply.

---

## 7. Host verification

| id | check | gates |
|---|---|---|
| V1 | `HostConfig.DeviceRequests[].DeviceIDs` holds rendered GPU indices (confirmed on the guest daemon) | P1 close |
| V2 | project- and label-scoped `docker ps -a` lists every model container | P1 close |
| V8 | a `paused` container keeps its GPU memory | P1 close |
| **V15** | **Whole `_global_lock` hold** for: a no-op apply; adding one model to a running static-gateway stack; removing one; the same add and remove with dynamic routing; a fresh dynamic gateway and Postgres bootstrap, measured separately; KubeAI apply if used; and several concurrent per-shard callers | **D22 final** |
| V3 | whether `up --remove-orphans` can start a service before an orphan frees its GPU | confirms the barrier is necessary |
| V5 | `up -d --no-deps <svc>` with an unchanged stanza does not recreate | P8 |
| V6 | crashed-container states under `unless-stopped` | P8 |
| V9 | `docker exec <gateway> python3` can reach `http://<service>:8000` | P7 |
| V10 | whether a stopped container keeps a static `ipv4_address` reserved | P7 |
| V11 | `network migrate` on a scratch copy of a running stack, measuring gateway downtime | P7 |
| V12 | the misroute reproduction with a low `AIOHTTP_TTL_DNS_CACHE` (operator's call) | informational |
| V13 | a changed label value recreates a service on the host daemon (confirmed on the guest) | P8 |
| V7 | the misroute reproduction | **done** |

---

## 8. Implementation order

| step | content | behaviour |
|---|---|---|
| **P1** | Strict residency (§4.1) | **done**, `675315c` |
| **P2** | Serialised publication (§4.3): the marker with `apply_requested` and `approved_digest`; **today's backend apply**, serialised under `_global_lock` and bounded (§4.11); route reconciliation with a total deadline gates the marker; every mutator through the controller, **including `renew`** (serialised under the lock for now; P5 adds the lock-free TTL-only fast path); read paths non-mutating (§4.10); `_ensure_applied` coalescing removed; registry commands through the controller (§4.9); **minimum immutable render context: initial profile persisted once, rendering only from it, explicit Compose environment** (§4.2, §5 step 3). Tests 8-20, 22, 27, 50-52 | closes the apply/render race and the crash windows, with deterministic recovery. **Valid under both outcomes of V15** |
| **P3** | Planner keywords (§4.4). Tests 1-7 | none by default |
| **P4** | `config publish` (preview-first), drift reporting, image pre-pull, catalog digest and the unpublished-endpoint check, profile-changing workflows (§4.2). The profile store itself landed in P2. Tests 23-26 | profile becomes changeable only explicitly |
| **P5** | `assigned_gpus`, backfill, and renew paths (§4.5); allocation migration. Tests 36, 54-55 | allocations become hard |
| **P6** | Admission preview (§4.6). Tests 28-35 | failed attempts commit nothing |
| **P7** | Stable addresses, persisted subnet, `network migrate`, upstream check (§4.8). Tests 56-60 | I14 |
| **P8** | Residency extension, labels and fingerprints, ownership, selective apply **replacing P2's apply**, barrier, interrupted-apply quiescence (§4.11), `gc --orphans`, adoption migration (§4.1, §4.7). Tests 21, 38-49, 53 | no `--remove-orphans` |
| **P9** | Idle keep-warm becomes optional and resident-only | **fixes the incident** |
| **P10** | DEGRADED end to end and observability (§4.13) | |

**If V15 fails:** P2 remains correct as a transitional mechanism. Reinstate
revision 6's bundle and activation design in place of P2's "final" status. Nothing
in P3-P10 changes.

---

## 9. Decisions

**Resolved:**

- **D1-D21:** as recorded in revisions 2-6.
- **D22: serialised publication**, provisionally adopted, final after V15.
- **D23:** a later ordinary apply starts leases staged with `--no-apply`. That is
  current behaviour, kept and documented.
- **D24:** route reconciliation under the lock has an end-to-end wall-clock
  deadline: small in steady state, larger for explicit bootstraps. Values come from
  V15.

**Open:**

| id | question | author's lean |
|---|---|---|
| **D25** | Timeout values for Docker operations under the lock | Set from V15 measurements: roughly 5× the observed steady-state p95 per operation, with a floor of 30 s for `up -d` |

---

## 10. Out of scope

Documented in
[`docs/planning/known-limitations.md`](../../docs/planning/known-limitations.md#leasing-scope-boundaries-and-known-faults).
Do not approximate these through existing code paths.

- Hot catalog or configuration changes during a leasing epoch.
- FIFO or reserved admission; a large request can be starved.
- Preemption of admitted leases.
- Automatic migration of LIVE deployments between GPUs.
- Automatic re-warming of displaced keep-warm deployments.
- Multi-node placement.
- Forged ownership labels.
- Making `observe()` strict.
- **Concurrent or coalesced publication** (added if D22 is confirmed): desired-state
  changes are applied one at a time under a host-wide lock.
- Whether an unswept past-TTL ACTIVE lease counts as demand; left as today.
- **Staged leases that only a manual `apply` may start** (D23). A lease staged with
  `--no-apply` is part of the desired state, and any later ordinary apply starts it.

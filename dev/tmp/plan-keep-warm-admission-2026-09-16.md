# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 3**
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** **plan for review. No code written.**
- **History:** revision 1 `5244229`; revision 2 `7be95ba`; revision 2 addendum on
  gateway misrouting `90d4ca4`. This revision answers the independent review of
  revision 2 and folds the addendum in. §0 lists every finding and its
  disposition.
- **Code baseline:** `dev/0.7.1` at `e751676`. Leasing code is unchanged since;
  line references were re-checked for this revision.
- **Evidence:**
  - [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
    (C1-C7, R1-R5, A-G, and the round-4 generation race);
  - [`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`](investigation-gateway-stale-upstream-after-recreate-2026-09-16.md)
    (M1-M4).

---

## 0. Changes from revision 2

The reviewer judged the core design correct, and found that the **transaction and
generation boundaries** were weaker than the invariants claimed. Revision 3
closes them. Every factual claim below was re-verified against the code.

| # | finding | re-verified | disposition |
|---|---|---|---|
| 1a | Staging and interactive approval ran inside `BEGIN IMMEDIATE`, contradicting "short transaction". The `renew` CLI writes SQLite directly without `_global_lock` (`commands_leasing.py:1530`, `controller.ledger.renew`), so a waiting approval could make renewals fail at the 5 s busy timeout (`store.py:93`). | yes | Admission restructured: prepare and approve **outside** any SQLite write transaction, then a short validating commit (§4.5) |
| 1b | `Ledger.renew` re-LIVEs an IDLE deployment and bumps the generation without rendering (see its docstring). After P4 that yields a LIVE deployment with no hard allocation, violating I2. | yes | **Accepted, refined.** Renew stays lock-free and TTL-only while its deployments are LIVE; an IDLE one is **re-admitted through admission**, not refused, so a running job is not killed (D13, §4.10) |
| 2 | Commit→publish is exception-safe but not **crash**-safe: a SIGKILL between COMMIT and `current` leaves an ACTIVE lease pointing at nothing. Two persistent "published" authorities also need a precedence rule. | — | **Accepted.** Durable staging before commit, an `activation` record committed with the lease, idempotent repair on every lock entry, and `current` as the **single** published authority; `published_generation` dropped (§4.2, §4.5) |
| 3a | D9: stable **paths** are fine but shared **contents** are not. `infer-stack.config-hash` (`CONFIG_HASH_LABEL`, `compose.py:770`, set at `853` and `1039`) would label a container with *G*'s hash while it reads *G+1*'s file. Routes are written at render (`compose.py:2010`) and read at apply (`2081`). | yes | **Accepted.** Auxiliary configs and route sets live in the immutable bundle and are **materialised to their stable runtime paths by apply(G)** under the apply lock. Publication never writes runtime files (§4.2) |
| 3b | `master_key()` also writes `.env` (via `write_env_file`), not only `db_password()`. | yes | **Accepted.** Secrets are provisioned independently of admission; `prepare()` reads them without writing and fails closed if missing (§4.5) |
| 4a | Selective apply classified only models and the gateway, not Postgres, Open WebUI or the reverse proxy (`compose.py:791`, `881`, `884`). `up --no-deps litellm` does not bootstrap Postgres. | yes | **Accepted.** Every service is classified from its rendered config, and dependencies are started explicitly (§4.6) |
| 4b | The barrier missed **same-deployment recreation**: a changed stanza recreates a container that still holds its GPU. | — | **Accepted.** Any recreation of a GPU service stops the old container first (I10) |
| 5a | "Departing = every labelled resident absent from G" auto-removes historical orphans, contradicting D11. | — | **Accepted.** Departure is derived from the **previously applied manifest**; unknown labelled containers are orphans, reported and never auto-removed. **Refined:** an orphan on a GPU being handed off **blocks** the apply (§4.6) |
| 5b | `dict[deployment_id, Resident]` silently loses duplicate containers. | — | **Accepted.** Residency keeps all matches; duplicates are ambiguous and fail closed (§4.1) |
| 6 | `reserved-gpu` deployments run no container (`test_reservation_renders_no_service_and_is_not_unrenderable`), so Docker-only backfill cannot establish their allocation. | yes | **Accepted.** A LIVE reservation stays unresolved and blocks new allocation until released, expired, or repaired (§5) |
| 7 | D8 "last N + current + applied" can delete an in-flight apply's bundle. | — | **Accepted.** No automatic bundle GC in this campaign |
| D10 | Which container states count as warm residency. | reviewer cited Docker docs | **Accepted:** `running`, `restarting`, `paused`; not `created`, `exited`, `removing`, `dead`. Non-resident containers remain visible to the barrier and GC |
| D11 | Orphan cleanup. | — | **Accepted:** `infer-stack gc --orphans`, explicit only, strict snapshot, existing approval / `--yes` |

### Additions by this author in this round

- **Refusing renew would kill running jobs.** The `renew` docstring describes a
  legitimate case: an ACTIVE lease whose TTL lapsed unswept, so its deployment went
  IDLE while the job was still using it, followed by a heartbeat. Refusing that
  renewal fails a running job over a bookkeeping race. D13 re-admits instead: it is
  resource-neutral when the container is still resident on GPUs nobody else holds,
  and an explicit failure otherwise.
- **Staged bundles are named by id; generations are assigned at commit.** A bundle
  must be durable before commit, but allocating its generation number before
  commit leaves gaps and contention from aborted attempts. The commit record maps
  the staged id to *G*.
- **The activation record carries its lease ids,** so recovery can compensate if a
  staged bundle has gone missing after a crash.
- **An orphan on a target GPU blocks the apply.** Never removing orphans
  automatically is right, but starting a service on a GPU an orphan occupies would
  violate I10. The apply refuses and names the orphan and `gc --orphans`.
- **Migration adopts matching containers.** With no previously applied manifest,
  every existing container would look like an orphan. A one-time adoption builds a
  generation-0 manifest from strict residency, but only for containers whose label
  **and** GPUs match a backfilled LIVE allocation.
- **Holding `_global_lock` through interactive approval is not new.** Today's
  converge already approves under the render lock. What changes is that the SQLite
  write lock is no longer held while approval waits.

---

## 1. Problem

A request can wait out its admission timeout while every GPU is empty:

- idle keep-warm deployments share a hard desired set with live demand (R1);
- the planner ignores demand (C3);
- unapplied renders persist pins (C4) and publish partial projects (R3);
- the intended pressure was never implemented (C2);
- apply is not bound to the render it was meant for (the round-4 race);
- recreating containers can make the gateway misroute a model's traffic to another
  container (M1-M4).

## 2. Goals and non-goals

**Goals**

1. Idle keep-warm residency never prevents admission.
2. Admission is lease-atomic, including renderability, and **crash-safe**.
3. An apply acts on exactly one published generation, **including its auxiliary
   configs and routes**, and records exactly that generation as applied.
4. No GPU is handed to a container while a previous occupant is still there,
   including a replacement for the same deployment.
5. Nothing destructive is decided from a failed or ambiguous observation.
6. Degraded deployments and orphans are isolated: nothing starts or removes them
   implicitly, and they block only what they actually conflict with.
7. Traffic for a deployment reaches only that deployment's container.

**Non-goals**

- FIFO or reserved admission (G);
- a durable `PENDING` state;
- preemption;
- automatic re-pinning or migration;
- automatic bundle GC;
- multi-node placement;
- changing `observe()`'s contract.

---

## 3. Invariants

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease is admitted, and its generation is published **or recoverably awaiting activation**. A waiting request has no lease. |
| **I2** | A `LIVE` deployment holds a **hard committed allocation**. It is never re-placed automatically; an invalid allocation makes it DEGRADED. No path, renew included, creates a LIVE deployment without one. |
| **I3** | An `IDLE` keep-warm deployment is optional, and a candidate only while **resident** (`running`, `restarting`, `paused`). infer-stack never creates its container. |
| **I4** | Admission is lease-atomic: every candidate deployment is placed and renderable, or nothing changes. |
| **I5** | A waiting attempt writes nothing on its own behalf, beyond the sweep carve-out and **pre-commit staged bundles, which are unreferenced and discardable**. |
| **I6** | Preparation and approval happen **outside** any SQLite write transaction. The commit is short, validates that its snapshot still holds, and records an activation, all within one `_global_lock` hold. |
| **I7** | Every published generation contains every admitted LIVE deployment. DEGRADED deployments are excluded from runtime actions. |
| **I8** | Physical residency comes from a strict, project-scoped Docker snapshot. A failure is **unknown**; several containers for one deployment is **ambiguous** and fails closed for that deployment. |
| **I9** | `current` is the **single authority** for the published generation. apply(G) materialises G's auxiliary files and routes, acts only on G, and marks only G applied. Publication never writes runtime files. |
| **I10** | **Barrier:** before any container starts on a GPU, whether a different deployment or a recreation of the same one, every other container occupying that GPU has been stopped by id and its exit confirmed. An **orphan** occupying it blocks the apply. |
| **I11** | **Selective apply:** every service is classified from G's rendered config. Departing containers come from the **previously applied manifest**, not "everything absent". Unknown labelled containers are orphans. No project-wide `--remove-orphans`. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`. |
| **I13** | Under unknown or ambiguous residency, only resource-neutral admission proceeds. Ledger releases and sweeps continue; destructive runtime reconciliation waits. |
| **I14** | A request routed for deployment D reaches only D's container, including just after containers are removed or created. |
| **I15** | **Recovery:** every entry into `_global_lock`, and every apply, first completes or compensates any committed-but-unactivated generation. |

---

## 4. Design

### 4.1 Strict residency → I3, I8

```python
@dataclass(frozen=True)
class Container:
    container_id: str
    deployment_id: str
    gpus: tuple[int, ...]
    state: str       # running | restarting | paused | created | exited | removing | dead

@dataclass(frozen=True)
class Residency:
    by_deployment: dict[str, tuple[Container, ...]]   # all matches, never collapsed
    def resident(self, gid) -> Container | None: ...   # exactly one warm container, else None
    def ambiguous(self, gid) -> bool: ...              # more than one container
    def occupants(self, gpu: int) -> tuple[Container, ...]: ...  # any non-removed state

def residency(self) -> Residency:     # raises ResidencyUnknown on any Docker error
```

- **Scope:** labels `com.docker.compose.project=<project>` **and**
  `infer-stack.deployment`, listed with `docker ps -a` and read with
  `docker inspect`.
- **GPUs:** from `HostConfig.DeviceRequests[].DeviceIDs` (V1).
- **Resident** (D10): `running`, `restarting`, `paused`.
- **Occupants** include every non-removed state, because a `created` or `exited`
  container still carries a device request that `up` may start.
- `observe()` is unchanged.

### 4.2 Generations, bundles, recovery → I1, I9, I15

**Bundle layout.** A staged bundle lives at `renders/staged-<uuid>/`; an activated
one at `renders/gen-<G>/`. Each contains:

- `docker-compose.yml`;
- `manifest.json`: services with their config hashes, hard allocations, optional
  placements, degraded set, and GPU map;
- `aux/litellm_config.yaml` and `aux/nginx.conf`, when present;
- `aux/routes.json`: the dynamic route set;
- `aux/route_registry.json`: the static-superset registry state.

Paths inside `docker-compose.yml` point at the **stable runtime paths** under
`state_dir`, because the gateway mount must not move (D9). The bundle carries the
*contents*.

**Store additions** (no `published_generation`):

- `generation_counter`;
- `activation(G, staged_id, lease_ids, created_at)`, with at most one row;
- `applied_generation`, as today.

**Commit** (inside the admission transaction): allocate *G*, insert
`activation(G, staged_id, lease_ids)`, and write the lease, claims and allocations.

**Activate** (after commit, idempotent):

1. rename `staged-<uuid>` to `gen-<G>` (skip if `gen-<G>` already exists), then
   fsync the directory;
2. atomically replace `current` with *G* (write, fsync, rename), unless `current`
   is already ≥ *G*;
3. a short transaction deleting the `activation` row.

**Recovery (I15)**, run at the start of every `_global_lock` hold and every apply:

- If an `activation` row exists and `staged-<uuid>` or `gen-<G>` exists: finish
  activation.
- If an `activation` row exists and neither directory exists: **compensate**.
  Release its `lease_ids`, delete the row, and publish an admitted-only
  generation.
- `staged-*` directories not named by any activation row are pre-commit leftovers:
  delete them.

**Precedence:** `current` says what is published; the activation row says only that
an activation is in flight. Nothing else claims publication.

**apply(G)**, under the apply lock:

1. Run recovery, then set `G := current`.
2. Materialise `gen-<G>/aux/*` to the stable runtime paths, with atomic writes.
3. Run the barrier and selective start (§4.6) via
   `docker compose -p <project> -f renders/gen-<G>/docker-compose.yml`.
4. Reconcile dynamic routes from `gen-<G>/aux/routes.json`.
5. Set `applied_generation = G`, and record `gen-<G>` as the previously applied
   manifest.

**Retention:** keep every bundle (D8). **Waiters** wait for
`applied_generation >= G`.

### 4.3 Planner: required, optional, hard → I2, I12

Unchanged from revision 2:

```python
plan_placement(deployments, inventory, *, required_ids, hard, optional_hints, allowed_gpus, ...)
```

1. Hard allocations are validated against the full pool. An invalid one becomes
   `degraded` and is never re-placed.
2. Required deployments without an allocation are placed by explicit placement,
   then fit, restricted to `allowed_gpus`.
3. Optional residents keep their hint GPUs where free; otherwise they are
   `displaced`. They are never newly fit.

With the new keywords omitted, placement is identical to today.

### 4.4 Ledger → I2

- **`assigned_gpus`:** a nullable column on `deployments`. It is set in the
  admission commit, and cleared in the same transaction as any LIVE→IDLE change.
- **IDLE→LIVE reuse:** adopt the physical GPUs of a unique resident container. If
  none is resident, place it fresh. If residency is unknown or ambiguous, the
  request is not resource-neutral (I13).
- **Desired set:** required = LIVE, with their hard allocations. Optional = IDLE
  keep-warm deployments that are uniquely resident.

### 4.5 Admission → I1, I4-I6, I13, I15

```text
with _global_lock():
    recover()                                                    # I15
    swept = ledger.sweep()                                       # I5 carve-out
    try: res = backend.residency()
    except ResidencyUnknown: res = UNKNOWN
    snap  = ledger.snapshot()               # read-only; includes a validation token
    cand  = ledger.overlay_acquire(snap, owner, requests, ttl)   # in memory (D1)
    if (res is UNKNOWN or res.ambiguous(any cand dep)) and not resource_neutral(cand, snap):
        raise NotAdmissible('residency unknown or ambiguous')
    prep  = backend.prepare(snap + cand, res)   # in memory; no writes; secrets read-only
    if prep.unplaced_or_unrenderable & cand.deployment_ids:
        raise NotAdmissible(prep)
    staged = backend.stage(prep)                # durable staged-<uuid>, unreferenced
    backend.approve(staged)                     # may wait for a human; no SQLite lock held
    with store.transaction():                   # SHORT
        ledger.validate(snap.token)             # changed -> rollback, discard staged, retry
        G = ledger.commit_overlay(cand, prep.allocations)
        ledger.record_activation(G, staged.id, cand.lease_ids)
    backend.activate(G, staged)                 # idempotent; a crash here is repaired by recover()
_ensure_applied(G)
```

- **`overlay_acquire`** runs the same coalescing rules as `Ledger.acquire` (compat
  key, capacity, sharing, IDLE→LIVE reuse; `ledger.py:324-342`) against a snapshot,
  **without writing**. Refactor both into one pure core that returns the intended
  mutations. `Ledger.acquire` applies them inside its transaction; admission
  applies them in `commit_overlay`. That gives one source of rules, and no write
  transaction during preview.
- **Validation token (D14):** a `state_version` counter bumped by every ledger
  mutation except TTL-only renewal. Lock-free renewals (§4.10) do not invalidate a
  pending admission.
- **Secrets:** `master_key()` and `db_password()` both write `.env`. They are
  provisioned by backend initialisation and by `doctor`. `prepare()` reads them and
  raises if they are missing; it never creates them.
- **`prepare()`** renders in memory, including service-name collisions
  (`compose.py:1106-1122`) and the next route-registry contents, and writes nothing.
- **Not admissible:** a queued caller sleeps and retries; a non-queued caller raises,
  naming the blocking admitted demand or the residency state. If `swept` changed
  state, publish an admitted-only generation through the same
  stage→commit→activate path.

### 4.6 Selective apply and the barrier → I7, I10, I11

```text
prev = manifest(applied_generation)     # generation 0 comes from migration adoption (§5)
m    = manifest(G); res = residency()   # unknown -> abort; G stays unapplied
for svc in m.services: classify(svc) in {unchanged, changed, new}
                       # by config hash: models, gateway, postgres, open-webui, reverse-proxy
departing = containers owned by prev (by deployment id) that m drops or displaces
orphans   = labelled containers owned by neither prev nor m       # report; never auto-remove

for gpu in gpus(services in m to start or recreate):
    blockers = res.occupants(gpu) - {containers m keeps unchanged on that gpu}
    if any orphan in blockers: abort(f"orphan {id} occupies GPU {gpu}; run gc --orphans")
    for c in blockers:      # departing, or the old container of a changed service
        docker stop c; wait for exit (bounded, else abort)
for c in departing: docker rm -f c.container_id
start = [s for s in m.services if class(s) in {new, changed}
         and s.deployment not in m.degraded and not optional(s)]
start += required dependencies of start that are new or changed    # e.g. postgres for litellm
docker compose -p P -f gen-G/docker-compose.yml up -d --no-deps <start, in dependency order>
```

- **A changed GPU service** has its old container stopped by the barrier before
  `up` recreates it. This is the same-deployment case.
- **Optional residents** are never started; one that vanished stays gone.
- **DEGRADED deployments** are never started, stopped or removed.
- **Orphans** are reported in `status`, and cleaned only by `gc --orphans`.

### 4.7 Gateway routing correctness → I14

The mechanism is chosen at D12. The candidates are in the gateway investigation:

- setting `AIOHTTP_TTL_DNS_CACHE` low on the gateway service (verified
  environment-overridable in the pinned image);
- stable per-service IPs;
- per-instance aliases;
- connection draining.

In every case, add an **upstream-direct readiness check** (`GET /v1/models` on the
service, verifying the served name), so a gateway/upstream disagreement is reported
as a routing fault. Because the barrier (§4.6) creates remove-then-create
sequences, I14 lands **before** P6.

### 4.8 Backends → D5

The admission capability is `residency`, `prepare`, `stage`, `approve`,
`activate`, `recover`.

- **Compose:** all of the above.
- **Null / KubeAI:** overlay admission and a short commit only. No residency,
  bundles or barrier; `activate` is a no-op, so recovery has nothing to finish.

### 4.9 Rollback and readiness timeout

- Readiness timeout today calls `self.release(...)` directly
  (`controller.py:778`). Route it through a strict release path.
- `_rollback_acquire`'s "never ran" decision uses strict residency, and evicts
  nothing when residency is unknown or ambiguous.

### 4.10 Renew → I2 (D13)

- **All of the lease's deployments LIVE:** update TTL and heartbeat only, lock-free
  as today, with no generation change.
- **Any deployment IDLE:** renew does **not** revive it inline. It returns
  `NEEDS_READMISSION`, and the controller re-admits **the same lease id** through
  §4.5:
  - if the container is still uniquely resident, on GPUs no other allocation holds,
    this is a resource-neutral adoption and the caller sees a successful renew;
  - otherwise it is ordinary admission (place, queue or fail), and a failure is
    reported explicitly (`renew: deployment reclaimed, re-acquire`). A deployment
    is never silently re-LIVEd without an allocation.
- The CLI `renew` switches from `controller.ledger.renew` to a controller method
  that does this.

### 4.11 Observability

`status` shows:

- UNKNOWN, AMBIGUOUS, NOT RUNNING, DEGRADED, DISPLACED and ORPHAN;
- `current`, `applied_generation`, and any in-flight activation;
- routing faults from the upstream-direct check.

Waiting and timeout messages name the contested GPU and what holds it.

---

## 5. Migration

1. **Schema:** add `assigned_gpus`, `generation_counter`, and the `activation`
   table.
2. **Secrets:** the migration provisions any missing secrets. Admission never does.
3. **Backfill hard allocations from strict residency only:**
   - a LIVE model deployment with exactly one resident container adopts that
     container's GPUs;
   - one with no resident container, or an ambiguous set, stays **unresolved**;
   - a **LIVE `reserved-gpu` deployment** runs no container, so its allocation
     cannot be established from Docker. It stays **unresolved**, because its
     descriptor promised an exact `CUDA_VISIBLE_DEVICES`, and re-placing it would
     break that contract. It resolves on release or expiry, or through an explicit
     operator release (D15);
   - while anything is unresolved, **new GPU allocation is refused**, but releases
     and sweeps proceed. A sidecar value may be logged for comparison, never used.
4. **Adoption:** generation 0 is a manifest built from the backfilled LIVE
   deployments and the containers matching them by label **and** GPU. It becomes the
   previously applied manifest. Every other labelled container is an orphan.
5. The first real publication writes `gen-1`. The old top-level
   `docker-compose.yml` and the sidecar are retired after that.

---

## 6. Tests

**R** marks tests the reviewer added; **A** marks tests this author added. Each
test names the invariant or risk it guards.

**Placement (I2, I3, I12)**

1. Incident shape: an unpinned idle resident against new LIVE demand, on a full
   host.
2. An idle resident's physical hint loses to LIVE demand.
3. A non-resident idle keep-warm deployment is absent from the plan.
4. IDLE→LIVE reuse adopts the resident container's GPUs.
5. A hard allocation outside `allowed_gpus` stays valid.
6. An invalid hard allocation becomes DEGRADED and is not re-placed.
7. With the new keywords omitted, plans are identical to today's.

**Admission (I1, I4-I6, I13)**

8. A non-queued request blocked only by warm cache is admitted.
9. A queued request blocked by admitted demand: failed retries change nothing, and
   staged directories are cleaned by the next recovery.
10. An unrelated request that fits is admitted while another waits.
11. A release applies while another request waits.
12. Two processes contend for the last GPU: exactly one is admitted.
13. A waiter's sweep publishes an admitted-only generation.
14. A killed waiter leaves nothing but a discardable staged directory.
15. A coalescing admission still produces a new generation (drift healing).
16. **R:** a placed but unrenderable candidate never creates an ACTIVE lease.
17. Declined approval: no commit, and the staged bundle is discarded.
18. **R, A:** approval waiting for a human does **not** hold the SQLite write lock;
    a concurrent lock-free `renew` of a LIVE lease succeeds.
19. **A:** a validation-token change between prepare and commit causes rollback and
    a retry.
20. Under unknown or ambiguous residency, only resource-neutral coalescing is
    admitted.

**Generations and crash recovery (I1, I9, I15)**

21. **R:** the generation race (commit before publication) cannot mark an unapplied
    generation applied.
22. **R:** `current` advancing during an apply does not change that apply's target.
23. **R:** the process is **killed between SQLite COMMIT and the `current`
    replacement** (a real subprocess kill, then reopen). Recovery activates *G*,
    and I1 holds.
24. **A:** the process is killed after COMMIT and the staged directory is lost.
    Recovery releases the activation's leases and publishes an admitted-only
    generation.
25. **A:** the process is killed before COMMIT. The staged directory is discarded,
    and no lease exists.
26. **R:** **G/G+1 auxiliary race:** *G+1* is published while *G* is being applied.
    *G*'s gateway container reads *G*'s config and carries *G*'s hash, and *G*'s
    routes are the ones reconciled.
27. Waiters wait on their own *G*.
28. A gateway stanza unchanged across generations does not recreate the gateway.

**Apply, barrier, orphans (I7, I10, I11)**

29. Barrier ordering for displacement: a fake runner rejects any start onto an
    occupied GPU.
30. **R:** **same-deployment recreation:** a changed stanza on a GPU service stops
    the old container before `up`.
31. The barrier leaves unaffected residents alone.
32. **R:** a vanished optional resident is not started.
33. **A:** a crashed optional resident (`restarting`) is left to Docker.
34. A DEGRADED deployment is untouched, and a release elsewhere still applies.
35. **R:** **fresh dynamic-routing bootstrap:** Postgres starts with the gateway, in
    dependency order.
36. **R:** an **unknown orphan** is reported and not removed.
37. **A:** an orphan occupying a GPU that is about to be started **blocks** the
    apply, and the message names `gc --orphans`.
38. `gc --orphans` removes exactly the reported containers, with approval.

**Residency, migration, renew, routing (I8, I13, I14, D13)**

39. `residency()` is project-scoped; labelled containers in other projects are
    ignored.
40. `residency()` raises on a Docker error, while `observe()` still returns `set()`.
41. **R:** **duplicate deployment labels** are ambiguous and fail closed for that
    deployment.
42. A stale sidecar is never accepted as an allocation.
43. **R:** a **LIVE reservation at migration** stays unresolved and blocks new
    allocation; releasing it resolves.
44. **A:** migration adoption: matching containers become generation 0, and
    non-matching ones are orphans.
45. **A:** renew on an all-LIVE lease is lock-free and changes no generation.
46. **A:** renew finding an IDLE deployment still uniquely resident gets transparent
    re-admission with the adopted GPUs.
47. **A:** renew finding an IDLE deployment whose GPU is now held elsewhere fails
    explicitly, and no LIVE deployment is left without an allocation.
48. Routing: remove X, create Y, and recreate X under continuous traffic. Every
    request for X reaches X.
49. The upstream-direct check reports a routing fault distinctly from "not ready".
50. A readiness timeout goes through the strict release path, and evicts nothing
    under unknown residency.

---

## 7. Host verification before implementation

- **V1:** `HostConfig.DeviceRequests[].DeviceIDs` carries the rendered GPU indices.
- **V2:** a `docker ps -a` scoped to the project and the deployment label lists
  every model container.
- **V3:** whether `up --remove-orphans` can start a new service before an orphan
  frees its GPU. This confirms the barrier is necessary.
- **V4:** `-p <project> -f <other path>` manages the same containers as today's
  file.
- **V5:** `up -d --no-deps <svc>` with an unchanged stanza does not recreate the
  container.
- **V6:** the states Docker reports for a crashed container under
  `unless-stopped`.
- **V7:** the gateway misroute reproduction, with container IPs recorded (V3-V4 in
  the gateway investigation).
- **V8:** a `paused` container keeps its GPU memory, which justifies D10's
  inclusion of `paused`.

---

## 8. Implementation order

A step may land only once every invariant it relies on is established. **P2 does
not start until the recovery rule and materialisation (§4.2) are implemented, with
tests 21-28.**

| step | content | behaviour |
|---|---|---|
| **P1** | Strict `residency()`, including duplicates (§4.1); tests 39-41 | additive |
| **P2** | Bundles, `current` as the authority, activation and recovery, materialisation on apply (§4.2); secrets provisioned outside admission; tests 21-28 | **changes apply**; fixes the round-4 race and the auxiliary race |
| **P3** | Planner keywords (§4.3); tests 1-7 | none by default |
| **P4** | Ledger `assigned_gpus`, migration backfill and adoption, reservation handling (§4.4, §5); **renew routed through the controller (§4.10)**; tests 42-47 | allocations become hard; renew no longer re-LIVEs inline |
| **P5** | Overlay admission with a short validating commit and stage/approve/activate (§4.5), with idle keep-warm **still required**; tests 8-20 | waiting requests write nothing |
| **P5b** | Gateway routing correctness per D12, and the upstream-direct check (§4.7); tests 48-49 | routing faults prevented and reported |
| **P6** | Selective apply, the barrier including same-deployment recreation, orphans, `gc --orphans` (§4.6); tests 29-38 | **changes apply**; no `--remove-orphans` |
| **P7** | Idle keep-warm becomes optional and resident-only (§4.4) | **idle yields to live**: fixes the incident |
| **P8** | DEGRADED end to end, the strict readiness-timeout path (§4.9), observability (§4.11); test 50 | isolation and visibility |

---

## 9. Decisions

**Resolved**

| id | decision |
|---|---|
| D1 | Overlay admission: one pure coalescing core; preview without a write transaction; a short validating commit |
| D2 | Ledger `assigned_gpus`, as a hard claim |
| D3 | Selective apply, not stanza carry-forward |
| D4 | Unknown or ambiguous residency: resource-neutral admission only |
| D5 | Backend capability; null and KubeAI get overlay admission only |
| D6 | No automatic re-pinning |
| D7 | DEGRADED is derived, not persisted |
| D8 | No automatic bundle GC in this campaign |
| D9 | Stable runtime paths, with immutable per-generation contents materialised by apply |
| D10 | Warm residency is `running`, `restarting`, `paused` |
| D11 | `infer-stack gc --orphans`, explicit only |

**Open, for the next review**

| id | question | author's lean |
|---|---|---|
| **D12** | How to establish I14 | Stable per-service IPs plus the upstream-direct check. A low `AIOHTTP_TTL_DNS_CACHE` is an acceptable interim measure once V7 shows Docker's DNS updates immediately |
| **D13** | Renew finding an IDLE deployment: re-admit the same lease (§4.10), or refuse and require re-acquire? | Re-admit. Refusing fails running jobs over a bookkeeping race |
| **D14** | Validation token: an explicit `state_version` counter, or one derived from generations and row versions? | An explicit counter bumped by every non-renew mutation; simplest to reason about |
| **D15** | Should the operator repair for an unresolved reservation accept GPUs as input, or only release it? | Release only, in this campaign. Entering GPUs by hand is a footgun |

---

## 10. Deferred

- FIFO or reserved admission (G).
- Automatic re-warming of displaced keep-warm deployments.
- Explicit migration of LIVE deployments between GPUs.
- Automatic bundle GC, which needs apply-lock or in-use protection.
- Any change to `observe()`.
- Whether an ACTIVE lease past its TTL but not yet swept should still count as
  demand. That is the root of the renew case in §4.10, and is out of scope here.

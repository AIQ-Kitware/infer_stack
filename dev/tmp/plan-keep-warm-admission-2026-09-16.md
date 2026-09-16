# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 2**
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** **plan for review. No code written.** Revision 1 (`5244229`) was
  reviewed by an independent model. This revision incorporates that review; §0
  records each finding and what was done with it.
- **Code baseline:** `dev/0.7.1` at `e751676`. There are no leasing changes since
  then; every line reference below was re-checked against that commit.
- **Evidence:**
  [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md),
  cited by its claim IDs (C1-C7, R1-R5, A-G).

---

## 0. Changes from revision 1

The required/optional/admission model survived review. What did not survive is
the layer underneath it: **the mutable, full-project render and apply protocol
is not strong enough to carry the invariants this plan wants.** Revision 2 adds
a render-generation protocol and selective apply, and reorders the
implementation so that nothing can displace a running container before those
exist.

| # | review finding | disposition |
|---|---|---|
| 1 | Apply is not tied to a render generation: an applier can apply an older file and mark a newer generation applied; the barrier and `up` can see different files | **Accepted, and it is pre-existing** (see below). New I9, §4.2; now step P2 |
| 2 | Carrying a degraded stanza forward (old D3) does not isolate it: a global `up` can still start or remove it, and can cold-start a vanished optional resident | **Accepted.** Replaced by selective apply (I11, §4.6) |
| 3 | Preview checks placement but not renderability (service-name collisions); publication can fail after commit | **Accepted.** The preview prepares the full render in memory; approval happens before commit; there is a compensation rule (§4.5) |
| 4 | A committed allocation is not a planner pin: pins are soft and get re-placed | **Accepted.** `assigned_gpus` is a hard claim; invalid means DEGRADED, never re-placed; no automatic re-pinning (I2, §4.3-4.4) |
| 5 | Migration's sidecar fallback turns stale renders into permanent ownership | **Accepted.** Backfill from Docker only; unresolved allocations block new allocation (§5) |
| 6 | "Allow admissions that displace nothing" under unknown residency is unprovable; "last applied optional set" has no source | **Accepted.** Only resource-neutral coalescing proceeds (I13); the unsourced sentence is removed |
| 7 | P1-P8 were not independently safe: optional displacement before admission atomicity can displace for a request that then fails | **Accepted.** Reordered (§8); "no behaviour change" claims corrected |
| — | Readiness timeout calls `self.release()` directly (`controller.py:778`), not `_rollback_acquire` | **Accepted.** §4.7 now names that path |
| — | `residency()` should be scoped to this Compose project and use `docker ps -a`/inspect | **Accepted.** §4.1 |
| — | Admission preview cannot require `backend.plan()`: only the compose backend has it | **Accepted.** Capability-specific (§4.8) |

**Pre-existing race, worth fixing on its own.** Finding 1 is a bug in *today's*
code, independent of keep-warm. `acquire` commits the ledger, bumping the desired
generation, and then renders, under `_global_lock` (`controller.py:699`).
`_ensure_applied` takes only the apply lock (`controller.py:516`), reads
`desired_generation()` (519), applies whatever file is on disk, and marks that
generation applied (523). A concurrent applier landing between the commit and the
file write applies the old file and marks the new generation applied. The
acquirer then skips its own apply, and its deployment is never started: a silent
readiness timeout. P2 fixes this even if nothing else here lands.

**Additions by this author on re-verification**, beyond the review:

- **Gateway config paths must stay stable.** The LiteLLM service mounts
  `{aux_dir}/litellm_config.yaml` by a path embedded in its stanza
  (`compose.py:863`), and static-superset mode exists so the gateway is *never
  recreated* (`compose.py:25`, `124`). Per-generation bundles must therefore not
  move `aux_dir`. Only the compose file and its manifest are per-generation (D9).
- **Docker's restart policy is a form of residency.** Model services render with
  `restart: unless-stopped` (`compose.py:351`, `412`). An optional resident that
  crashes is restarted by *Docker*, not by an infer-stack render. That preserves
  residency and is allowed by I3. A resident whose container is **removed** is
  gone. "Resident" is defined on container existence, not on being up this
  instant (§4.1, D10).
- **Departing services cannot be named to `docker compose`.** A service absent
  from the new compose file cannot be stopped with `docker compose stop <svc>`
  against that file, so selective apply stops and removes departing containers
  **by container id** taken from residency (§4.6).
- **Render has side effects today.** Before `render_compose`, converge persists
  the LiteLLM DB secret (`compose.py:1933-1936`, `db_password()`) and the route
  registry (`_update_route_registry`, persisted when it changes). An in-memory
  preview must not perform these on the candidate's behalf (§4.5).

---

## 1. Problem

A request can wait out its admission timeout while every GPU is empty. Idle
keep-warm deployments share one hard desired set with deployments under active
demand (R1). The planner orders by pins and creation time without regard to
demand (C3). Renders that were never applied persist pins (C4) and publish
partial projects that another process may apply (R3). The intended "yield under
pressure" (C2) was never implemented. Underneath all of that, apply is not bound
to the render it was meant for (§0, finding 1).

## 2. Goals and non-goals

**Goals**

1. Idle keep-warm residency never prevents a request from being admitted.
2. Admission is lease-atomic, including renderability.
3. An apply acts on exactly one published render and records exactly that
   render as applied.
4. No GPU is handed to a container while its previous occupant is still there.
5. Nothing destructive happens on the strength of an observation that failed.
6. A degraded deployment is isolated: nothing starts it, removes it, or blocks
   others on its account.

**Non-goals:** FIFO or reserved admission (G); a durable `PENDING` state;
preempting admitted deployments; automatic re-pinning or migration; multi-node
placement; changing `observe()`'s best-effort contract.

---

## 3. Invariants

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease is an **admitted and published** request. A request still waiting has no lease. |
| **I2** | A `LIVE` deployment is required and holds a **hard committed allocation** (`assigned_gpus`). It is never re-placed automatically. If that allocation becomes invalid, the deployment is **DEGRADED**, not moved. |
| **I3** | An `IDLE` keep-warm deployment is **optional**, and is a candidate only while it is **resident**: its container exists in this project. It may keep its GPU if capacity allows. infer-stack never *creates* its container. |
| **I4** | Admission is **lease-atomic**: every candidate deployment is placed **and renderable**, or the attempt changes nothing. |
| **I5** | A waiting attempt writes nothing on its own behalf. **Carve-out:** it may sweep TTL-expired admitted state and publish the resulting admitted-only render. |
| **I6** | Admission is **one speculative transaction under one global-lock hold**. Operator approval precedes commit, and a publication failure after commit is compensated before the lock is released. |
| **I7** | Every published render contains every admitted LIVE deployment. A DEGRADED deployment is excluded from runtime actions by selective apply (I11). |
| **I8** | Physical residency comes from a **strict snapshot** of this project's labelled containers (`ps -a` plus inspect). A failed snapshot is **unknown**, never empty. |
| **I9** | **An apply operates on one immutable published render generation and marks only that generation applied.** A waiter waits for the *published* generation that contains its change. |
| **I10** | **Handoff barrier:** every container on a GPU being handed to another deployment is stopped by container id, and its exit confirmed, before anything starts on that GPU. The barrier and the start use the same render generation. |
| **I11** | **Selective apply:** start or reconcile only services that must start or have changed; stop and remove departing containers by id; never start an optional resident; never act on a DEGRADED service; no project-wide `--remove-orphans` in the reconcile path. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`; only *new* allocations are restricted to it (`placement.py:250-258`). |
| **I13** | Under **unknown residency**, only **resource-neutral** admission proceeds: coalescing onto an already-LIVE deployment with a committed allocation and no topology change. Releases and sweeps still update the ledger; destructive runtime reconciliation is deferred until residency is known. |

---

## 4. Design

### 4.1 Strict residency snapshot → I3, I8

```python
@dataclass(frozen=True)
class Resident:
    deployment_id: str
    container_id: str
    gpus: tuple[int, ...]
    state: str            # created | running | restarting | exited | paused | dead

class ResidencyUnknown(Exception): ...

def residency(self) -> dict[str, Resident]:   # raises ResidencyUnknown
```

- **Scope:** containers carrying **both** `com.docker.compose.project=<project>`
  and `infer-stack.deployment` (`DEPLOYMENT_LABEL`, `compose.py:101`). Listed with
  `docker ps -a --filter label=...` and read with `docker inspect`.
- **GPUs:** from the container's actual device request. Services render
  `deploy.resources.reservations.devices[].device_ids` (`_gpu_reservation`);
  expected at `HostConfig.DeviceRequests[].DeviceIDs`. Verify first (§7, V1).
- **Resident** means the container **exists** in a state that holds or will
  reclaim its GPU: running, restarting, or exited while its restart policy will
  restart it. A removed container is not resident (D10 fixes the exact rule).
- **Failure:** any Docker error or unparseable output raises
  `ResidencyUnknown`. `observe()` is unchanged: its empty-on-error result is a
  tested contract (`test_leasing_compose.py:494`).
- Fix the docstring at `compose.py:161`, which says `observe()` uses the label.

### 4.2 Render generations → I9 (fixes the pre-existing race)

- **Published generation.** Add `published_generation` to the store beside
  `desired_generation` and `applied_generation`.
- **Immutable bundle per generation.** Publishing generation *G* writes
  `renders/gen-<G>/docker-compose.yml` plus `renders/gen-<G>/manifest.json`
  (hard allocations, optional placements, service→deployment map, degraded set,
  generation) to a temporary directory, then renames it into place. It then
  atomically replaces a `current` pointer file naming *G*.
- **Stable shared paths.** The gateway and proxy configs stay at their existing
  stable paths under `state_dir`, because the LiteLLM stanza mounts them by path
  and must stay byte-stable (§0; D9). The bundle's compose file references those
  stable paths. Only model-service stanzas and the manifest vary by generation.
- **Apply reads one bundle.** `_ensure_applied` reads `current` → *G*, then runs
  the barrier and the start (§4.6) against `gen-<G>` using
  `docker compose -p <project> -f renders/gen-<G>/docker-compose.yml`, and sets
  `applied_generation = G`. Never `desired_generation()`.
- **Waiters** capture the generation their own publication produced and wait for
  `applied_generation >= that`.
- **Retention:** keep the last *N* bundles; never delete `current` or
  `applied` (D8).
- The sidecar (`leasing-compose-state.json`) is retired as a source of truth. Its
  service map moves into the manifest.

### 4.3 Planner: required, optional, and hard allocations → I2, I12

```python
plan_placement(deployments, inventory, *,
               required_ids: set[str],
               hard: dict[str, list[int]],      # committed allocations
               optional_hints: dict[str, list[int]],  # residents' physical GPUs
               allowed_gpus=..., reserved=..., skip_display=...)
```

1. **Hard allocations first**, validated against the full physical pool (I12).
   A valid one is taken as-is. An invalid one (a GPU missing, or a global
   `reserved` change) is reported in `plan.degraded` and **not re-placed**
   (unlike today's pins, `placement.py:298-311`).
2. **Required without an allocation** (the candidate's new deployments): explicit
   placements, then fit, restricted to `allowed_gpus`.
3. **Optional residents**: honour `optional_hints` where free; otherwise put them
   in `plan.displaced`. Optional deployments are never *newly* fit, because I3
   forbids creating residency.

- `(n_eligible, created_at, id)` survives only as a tie-break within step 2.
- **Compatibility:** with the new keywords omitted, behave exactly as today
  (tested).

### 4.4 Ledger: hard allocations → I2

- **Schema:** nullable `assigned_gpus` (JSON list) on `deployments`.
- **Set** in the admission transaction, for every candidate deployment that
  becomes LIVE.
- **IDLE→LIVE reuse** (`_find_or_create_deployment`, `ledger.py:324-342`): if the
  deployment is resident, adopt its **physical** GPUs from residency as the hard
  allocation, with no move. If it is not resident, place it fresh. If residency is
  unknown, the reuse is not resource-neutral, so it waits (I13).
- **Cleared atomically** in the same transaction that takes a deployment from
  LIVE to IDLE (release, sweep, rollback). A still-running container keeps its GPU
  only as an optional hint, via residency.
- **Desired set:** required = LIVE, each with its hard allocation. Optional = IDLE
  keep-warm ∩ resident. Under unknown residency, the optional set is unknown and
  nothing is displaced (I13).

### 4.5 Admission as a speculative transaction → I1, I4, I5, I6

```text
with _global_lock():
    swept = ledger.sweep()                                   # I5 carve-out
    try: residency = backend.residency()
    except ResidencyUnknown: residency = UNKNOWN
    with store.transaction() as txn:                         # BEGIN IMMEDIATE
        cand = ledger._acquire_core(txn, owner, requests, ttl)
        if residency is UNKNOWN and not resource_neutral(cand):
            raise NotAdmissible('residency unknown')          # rollback
        prepared = backend.prepare(desired(txn, residency),   # IN MEMORY
                                   required_ids=..., hard=..., hints=...)
        if prepared.unplaced_or_unrenderable & cand.deployment_ids:
            raise NotAdmissible(prepared)                     # rollback
        ledger._commit_allocations(txn, prepared)
        bundle = backend.stage(prepared, generation=next_gen) # writes gen-<G>.tmp only
        backend.approve(bundle)                               # operator diff; may raise -> rollback
        # COMMIT happens here, at the end of the with-block
    try:
        backend.publish(bundle)                               # rename + `current` pointer
    except Exception:
        ledger.release(cand.lease_id)                         # compensation, still under the lock
        backend.publish_admitted_only()                       # best effort
        raise
G = bundle.generation
_ensure_applied(G)                                            # outside the lock
```

- **`prepare()`** is the full backend render in memory: placement plus
  `render_compose`, including service-name collision detection
  (`compose.py:1106-1122`). It performs **no** writes: no DB secret, no route
  registry, no files. Those side effects move into `publish()`.
- **Not admissible:** a queued caller sleeps and retries; a non-queued caller
  raises `PlacementError` naming the blocking **admitted** demand, or the unknown
  residency. If `swept` changed the desired set, publish the admitted-only render
  (I5).
- **Displaced optionals** stay IDLE and appear in the manifest's displaced list.
  `evict_idle` is not called.
- **Drift healing is preserved:** a successful coalescing admission still produces
  a new published generation, so an apply runs and restarts crashed containers
  (`ledger.py:158-163`).
- **D1** is this speculative transaction: one transaction that commits on success,
  not a preview followed by a second commit.
- **Lock cost:** the residency snapshot is taken **before** `BEGIN IMMEDIATE`, and
  `prepare()` must not call Docker, so the SQLite write lock is held only for
  in-memory work.

### 4.6 Selective apply and the barrier → I10, I11, I7

`apply(G)` under the apply lock, against `renders/gen-<G>`:

```text
res = residency()                        # unknown -> abort; G stays unapplied; retry later
m   = manifest(G)
departing  = residents not in m.required ∪ m.optional_kept     # incl. displaced
handoff    = departing whose gpus ∩ gpus(m.required ∪ m.optional_kept) ≠ ∅
for r in handoff:   docker stop r.container_id; wait for exit (bounded) -> else abort
for r in departing: docker rm -f r.container_id                # by id, not by service
to_start = [svc for d in m.required
            if d not in m.degraded and (not resident(d) or stanza_changed(d, G))]
if gateway stanza changed: to_start += [gateway]
docker compose -p P -f gen-G/docker-compose.yml up -d --no-deps <to_start...>
reconcile LiteLLM routes via the admin API (as today)
applied_generation = G
```

- **Optional residents** are never in `to_start`. If one's container vanished
  between render and apply, it stays gone (I3).
- **DEGRADED** services are never in `to_start` and never in `departing`.
  Nothing starts or removes them, and an unrelated release still applies (I7).
- **No `--remove-orphans`** in this path. Stray containers from before this
  change are handled by an explicit, operator-invoked cleanup (D11).
- `stanza_changed` compares the service's rendered stanza, or its config hash,
  between the applied generation and *G* (§7, V5).

### 4.7 Rollback and readiness timeout → goal 5

- Admission failure needs no rollback: nothing was committed (§4.5).
- **Readiness timeout after admission** today calls `self.release(...)` directly
  (`controller.py:778`). Route it through one strict release path, `_release_strict`.
- `_rollback_acquire`'s "never ran" decision (`controller.py:610-646`) uses strict
  residency. On `ResidencyUnknown` it evicts nothing.

### 4.8 Backend capability → D5

Admission preview is a backend capability (`prepare`, `stage`, `publish`,
`residency`).

- **Compose:** everything above.
- **Null / KubeAI:** lease-atomic ledger admission only. No host placement, no
  residency, no barrier. `prepare()` is renderability-free and always admits what
  the ledger accepts.

### 4.9 Observability

- `status` distinguishes **UNKNOWN**, **NOT RUNNING**, **DEGRADED**, and
  **DISPLACED** (optional, not resident).
- Waiting and timeout messages name the admitted lease and deployment holding the
  contested GPU, or say that residency is unknown.
- Expose `published_generation` and `applied_generation` in `status` so a stuck
  apply is visible.

---

## 5. Migration

- **Schema:** add `assigned_gpus`, and add the `published_generation` counter.
- **Backfill hard allocations for existing LIVE deployments from strict residency
  only.** If residency is unknown, or a LIVE deployment has no resident container,
  leave its allocation **unresolved**. While any allocation is unresolved, refuse
  *new* GPU allocations (I13 applies) but allow releases and sweeps. A sidecar
  value may be logged as a hint to compare against Docker; it is **never** the
  authority.
- **First publication** after upgrade writes `gen-<G>` from the current ledger.
  The old top-level `docker-compose.yml` is left in place until then, then removed.
- Existing idle keep-warm deployments that are not running need no migration: I3
  stops desiring them.
- Manual `infer-stack apply` applies `current` through §4.6 and prints why when it
  refuses.

---

## 6. Tests

Each test names what it guards. "Real planner" means `plan_placement` on
`simulate_inventory`. **R** marks tests added by the reviewer; **A** marks those
added on re-verification.

**Placement (I2, I3, I12)**

1. Old unpinned idle resident vs new LIVE, full host: LIVE placed; idle in
   `displaced`. This is the incident's shape.
2. Old idle resident holding a physical hint vs new LIVE: LIVE wins that GPU.
3. Idle keep-warm that is not resident is absent from the plan despite free GPUs.
4. IDLE→LIVE reuse of a resident adopts its physical GPUs as the hard allocation.
5. A hard allocation outside the caller's `allowed_gpus` stays valid (I12).
6. An invalid hard allocation is reported DEGRADED and is **not** re-placed.
7. With the new keywords omitted, plans are identical to today's.

**Admission (I1, I4-I6, I13)**

8. Non-queued `acquire` blocked only by warm cache succeeds immediately.
9. Queued `acquire` blocked by admitted demand: each failed retry leaves the
   ledger, all three generations, the bundles, and `current` unchanged.
10. An unrelated request that fits is admitted while another waits.
11. A release applies while another request waits.
12. Two processes contend for the last GPU: exactly one admitted (real `flock`).
13. A waiter's sweep frees capacity and publishes admitted-only; nothing of the
    candidate's is written.
14. A killed waiter leaves nothing to clean up.
15. Coalescing admission still produces a new published generation (drift healing).
16. **R:** a placed but **unrenderable** candidate (service-name collision) never
    creates an ACTIVE lease.
17. **A:** approval declined → the transaction rolls back and no bundle is published.
18. **A:** publication fails after commit → the lease is released under the same
    lock hold, and no ACTIVE lease survives.
19. Unknown residency: new deployment and IDLE→LIVE reuse are refused;
    coalescing onto a LIVE deployment with an allocation is admitted.

**Render generations (I9)**

20. **R:** deterministic race: desired generation committed before its render is
    published; a concurrent applier cannot mark it applied without applying it.
21. **R:** the published render changes between barrier inspection and start; the
    applied target does not change.
22. **A:** a waiter waits for its own published generation, not a desired
    generation.
23. **A:** a gateway stanza unchanged across generations does not recreate the
    gateway (stable `aux_dir` paths).

**Apply, barrier, degraded (I7, I10, I11)**

24. Barrier ordering, with a fake runner that rejects the start of a service
    whose GPU is held by a container it still reports: `stop(B)` → B exited →
    `up`.
25. Barrier does not stop residents whose GPUs are not handed off.
26. **R:** an optional resident that **disappears** (container removed) between
    render and apply is **not** started.
27. **A:** an optional resident that **crashes** (container exists; restart policy
    active) is left to Docker and not removed by apply.
28. A DEGRADED LIVE deployment is neither started nor removed, and a release
    elsewhere still applies.
29. No `--remove-orphans` in the reconcile path; departing containers are removed
    by id.

**Residency and migration (I8, goal 5)**

30. `residency()` maps by label within this project; a same-labelled container in
    another project is ignored.
31. `residency()` raises on a Docker error; `observe()` still returns `set()`.
32. Readiness timeout and `_rollback_acquire` evict nothing under unknown
    residency.
33. **R:** a stale sidecar is never accepted as a committed allocation when Docker
    cannot corroborate it; the allocation stays unresolved and new allocation is
    refused.

---

## 7. Verification before implementation (host)

- **V1:** `docker inspect --format '{{json .HostConfig.DeviceRequests}}' <model container>`
  shows the indices written by `_gpu_reservation`.
- **V2:** `docker ps -a --filter label=com.docker.compose.project=<project> --filter label=infer-stack.deployment`
  lists every model container, including exited ones.
- **V3:** confirm whether `docker compose up --remove-orphans` can start a new
  service before an orphan releases its GPU. The barrier does not depend on the
  answer; it tells us whether the barrier is necessary or only defensive.
- **V4:** `docker compose -p <project> -f <other path>/docker-compose.yml up -d --no-deps <svc>`
  manages the same containers as the current top-level file, i.e. the project
  name, not the file path, determines identity.
- **V5:** `up -d --no-deps <svc>` with an unchanged stanza does not recreate the
  container. This is what makes "start or changed" cheap and safe.
- **V6:** confirm how Docker reports a crashed container under
  `restart: unless-stopped` between restarts, so D10's resident states are exact.

---

## 8. Implementation order

Each step preserves every invariant established by the steps before it. Steps
that change behaviour are marked. **P6 is the first step at which a running
container can be displaced**, and it requires P2, P4 and P5.

| step | content | behaviour |
|---|---|---|
| **P1** | Strict `residency()` (§4.1); tests 30-31; docstring fix | additive |
| **P2** | Render generations, immutable bundles, apply-one-generation (§4.2); tests 20-23 | **changes apply**; fixes the pre-existing race; worth landing on its own |
| **P3** | Planner `required_ids` / `hard` / `optional_hints` with compatibility default (§4.3); tests 1-7 at planner level | none for existing callers |
| **P4** | Ledger `assigned_gpus`, strict backfill, clear on LIVE→IDLE (§4.4, §5); test 33 | allocations become hard |
| **P5** | Admission as a speculative transaction with full `prepare`, approval before commit, compensation (§4.5), with idle keep-warm **still treated as required**; tests 8-19 | **waiting requests write nothing**; no displacement yet |
| **P6** | Selective apply plus barrier (§4.6); tests 24-29 | **changes apply**; no `--remove-orphans` |
| **P7** | Desired split: idle keep-warm becomes optional and resident-only (§4.4) | **idle yields to live**: this fixes the incident |
| **P8** | Degraded state end to end, strict readiness-timeout path (§4.7), observability (§4.9); test 32 | degraded isolation |

---

## 9. Decisions

**Resolved in review**

- **D1:** speculative transaction around shared acquire logic, coupled to a
  side-effect-free full `prepare()`.
- **D2:** ledger, as a **hard** allocation, not a pin.
- **D3:** not carry-forward; selective apply (I11).
- **D4:** under unknown residency, only resource-neutral coalescing (I13).
- **D5:** admission preview is a backend capability; null and KubeAI get ledger
  atomicity only.
- **D6:** no automatic re-pinning. Any future migration must go through the
  barrier.
- **D7:** DEGRADED is derived from the hard allocation plus inventory and runtime,
  not persisted as a lifecycle state.

**Open, for the next review**

- **D8:** bundle retention: how many generations to keep, and when to garbage
  collect. The bundles currently applied or `current` must never be removed.
- **D9:** gateway config is at a stable, mutable path while compose bundles are
  immutable. Is it safe for generation *G*'s apply to see a gateway config written
  for *G+1*? In static-superset mode it only accumulates routes, so *probably
  yes*; in dynamic-routing mode routes go through the admin API. Confirm for both.
- **D10:** exact resident states: running, restarting, and exited with an active
  restart policy count as resident. Is exited-with-non-zero-and-restart-exhausted
  resident? Does `paused` count?
- **D11:** an explicit `infer-stack gc --orphans` for stray containers left from
  before P6, since the reconcile path no longer uses `--remove-orphans`.

---

## 10. Explicitly deferred

- FIFO or reserved admission for large requests (G). Add `PENDING` only then,
  probably as lease metadata without claims.
- Automatic re-warming of displaced keep-warm deployments (rejected by I3).
- Explicit migration of LIVE deployments between GPUs (D6).
- Any change to `observe()`'s best-effort semantics.

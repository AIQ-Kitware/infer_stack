# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** **plan for review. No code written.** It is meant to be
  corroborated by an independent reviewer before implementation.
- **Code baseline:** `dev/0.7.1` at `e751676`.
- **Evidence:**
  [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md),
  including its three review rounds. This plan does not repeat the evidence; it
  cites claims by their IDs there (C1-C7, R1-R5, A-G).

---

## 1. Problem, in one paragraph

A request for GPUs can wait out its whole admission timeout while every GPU is
empty. Idle **keep-warm** deployments are merged into the same hard desired set
as deployments with active lease demand (R1). The planner orders them by pins and
creation time with no notion of demand (C3), so idle deployments win GPUs that a
live request needs. Renders that were never applied persist those choices as
pins (C4) and publish partial compose projects that another process may apply
(R3). The design intent was always that idle keep-warm should yield "under
pressure" (C2); that pressure was never implemented.

## 2. Goals and non-goals

**Goals**

1. Idle keep-warm residency can never prevent a request from being admitted.
2. Admission is **lease-atomic**: a request is either admitted whole, or it
   changes nothing.
3. Nothing published to disk ever omits an admitted deployment, and nothing a
   waiting request does is ever published.
4. A GPU is never handed to a new deployment while the previous occupant's
   container is still on it.
5. Decisions that destroy state (eviction, teardown, displacement) are never made
   from an observation that failed.

**Non-goals** (recorded so they are not scope-crept in)

- FIFO fairness or reservations for waiting requests. Small requests can still
  starve a large one (G); that is a documented follow-up.
- A durable `PENDING` lease state (not needed; see I5).
- Preempting admitted deployments. Established LIVE deployments are
  non-preemptible.
- Multi-node placement.
- Changing `observe()`'s best-effort contract.

---

## 3. Invariants

These are the acceptance criteria. Every implementation step below exists to
establish one of them, and every test in §6 names the invariant it guards.

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease means an **admitted** request. A request still waiting for admission has no lease. |
| **I2** | A `LIVE` deployment is **required** capacity. It is placed before any optional deployment, in every tier, pins included. |
| **I3** | An `IDLE` keep-warm deployment is **optional**, and is a placement candidate **only while it is physically resident**. It may keep its GPU if capacity allows. It is never started by a render it was not requested in. |
| **I4** | Admission is **lease-atomic**: either every deployment of the candidate request is placed, or the attempt changes nothing. |
| **I5** | A waiting attempt writes nothing on its own behalf: no lease, claim, deployment state change, desired-generation bump, compose file, or sidecar. **Carve-out:** it may sweep TTL-expired *admitted* state, and if that changes the desired set, publish the resulting admitted-only render (D). |
| **I6** | The admission preview and the admission commit happen under **one hold** of the cross-process global lock (C). |
| **I7** | A **published** render places every admitted LIVE deployment. If one cannot be placed, the deployment is **admitted-but-degraded**: surfaced, not silently revoked, not torn down by orphan removal, and it does not block other leases' releases (E). |
| **I8** | "What is physically on which GPU" comes from a **strict residency snapshot** of Docker state: deployment label plus device reservation. It never comes from the last rendered sidecar. A failed snapshot is **unknown**, not empty (B, F). |
| **I9** | **Handoff barrier:** before a render that assigns GPU *k* to deployment X is applied, any other container resident on *k* has been stopped, and its exit confirmed (Q10). |
| **I10** | Placement stability for established allocations ignores the calling job's `allowed_gpus`; only *new* placements are restricted to it. This preserves today's Slurm behaviour (`placement.py:250-258`). |

---

## 4. Design, by component

### 4.1 Physical residency snapshot (compose backend) → I8

Add a strict operation beside `observe()`, leaving `observe()` unchanged:

```python
@dataclass(frozen=True)
class Resident:
    deployment_id: str
    container_id: str
    gpus: tuple[int, ...]
    running: bool          # State.Running

class ResidencyUnknown(Exception): ...

def residency(self) -> dict[str, Resident]:   # raises ResidencyUnknown
```

- **Source:** containers carrying the `infer-stack.deployment` label
  (`DEPLOYMENT_LABEL`, `compose.py:101`), found via
  `docker ps --filter label=infer-stack.deployment` plus `docker inspect`. Query
  by label rather than by compose project, so a container left behind by an
  earlier render, or one no longer listed in the sidecar, is still seen.
- **GPUs:** from the container's actual device reservation. Services render
  `deploy.resources.reservations.devices[].device_ids`
  (`_gpu_reservation`, `compose.py`), which should surface as
  `HostConfig.DeviceRequests[].DeviceIDs`. **Verify on the host before relying
  on it** (§7, V1).
- **Failure:** any Docker error, unparseable output, or a label with no matching
  container that is not explicitly "absent" raises `ResidencyUnknown`. It never
  returns `{}` on error.
- **Other backends:** `null` and `kubeai` have no physical GPU view here. They
  return a result that makes optional residency empty and the barrier a no-op.
  They keep today's behaviour and gain nothing (decision D5).

Fix the misleading docstring at `compose.py:161`, which claims `observe()` uses
the label.

### 4.2 Planner: required before optional → I2, I10

Extend `plan_placement` (`placement.py`) with explicit requiredness rather than
inferring it from `Deployment.state`. A candidate that reuses an IDLE deployment
must be required in the preview while the ledger still says IDLE:

```python
plan_placement(deployments, inventory, *, required_ids: set[str], pinned=..., ...)
```

Tier order:

1. required pins
2. required explicit placements
3. required fit
4. optional pins
5. optional explicit placements
6. optional fit

- Pins are validated against the full physical pool (`pin_pool_set`) exactly as
  today (I10).
- Within a tier, `(n_eligible, created_at, id)` remains a deterministic
  tie-breaker only; it is no longer an admission policy (Q11).
- An unplaced optional deployment is not an error. It appears in a separate
  `plan.displaced` list, not in `plan.errors`.
- **Compatibility:** when `required_ids` is omitted, treat everything as required.
  This gives byte-identical plans for existing callers and tests (step P2).

### 4.3 Where pins come from → I8, and decision D2

Today pins are the sidecar's last *rendered* assignments (C4). Two options:

- **D2-a (recommended): committed allocation lives in the ledger.** At admission
  commit, write the assignments for the candidate's LIVE deployments into the
  deployment rows (a nullable `assigned_gpus` column), in the same transaction
  (I6). Required pins come from the ledger; optional pins come from the residency
  snapshot's actual GPUs. The sidecar keeps only the service-name map, used by
  `observe()` and the renderer.
- **D2-b: split the sidecar** into `rendered` and `applied` assignment sections.
  This avoids a schema change, but keeps pins in a file that a render rewrites,
  and still needs its own atomicity story.

D2-a matches "committed LIVE allocation is logical ownership" and makes I6
straightforward. It needs a ledger migration (§5).

### 4.4 Desired set → I2, I3

Replace `desired_deployments()` (`controller.py:395-406`) with a split:

```python
def desired(self, residency) -> tuple[list[Deployment], list[Deployment]]:
    required = LIVE deployments
    optional = IDLE keep-warm deployments that are resident (residency[gid].running)
```

- If `residency` is unknown: optional residents are **not dropped**. Carry
  forward the last applied optional set (fail safe, goal 5), and do not apply
  anything that would displace them. See D4.
- IDLE deployments that are not resident are simply not desired. Their ledger
  state is unchanged. This resolves Q1: the incident's non-running idle
  deployments would never have entered the plan.

### 4.5 Admission as preview-then-commit → I1, I4, I5, I6

`Controller.acquire` (`controller.py:648-800`) today writes the lease first and
rolls back on failure (R4). New flow, entirely under one `_global_lock` hold:

```text
with _global_lock():
    swept = ledger.sweep()                         # I5 carve-out
    residency = backend.residency()                # may raise -> not admissible now
    with store.transaction() as txn:               # BEGIN IMMEDIATE
        candidate = ledger._acquire_core(txn, owner, requests, ttl)   # D1
        required, optional = desired(residency)    # sees candidate's writes
        plan = backend.plan(required + optional,
                            required_ids={ids of required} | {candidate deployment ids})
        if any candidate deployment unplaced:
            raise NotAdmissible(plan)              # ROLLBACK: nothing written
        ledger._commit_assignments(txn, plan)      # D2-a
    # committed
    publish render                                 # 4.6
g_target = desired_generation(); _ensure_applied(g_target)  # outside the lock, as today
```

- **Not admissible:** if `swept` changed desired state, publish the admitted-only
  render (I5 carve-out). A queued caller sleeps and retries; a non-queued caller
  raises `PlacementError` naming the blocking **admitted** demand.
- **Displaced optionals:** `plan.displaced` optional deployments are not rendered,
  and **stay IDLE**; `evict_idle` is not called (round 2). The barrier (4.7)
  handles their containers.
- **Keep today's drift healing:** a successful admission still bumps the desired
  generation even when it only coalesces onto an existing deployment
  (`ledger.py:158-163`), because that forced apply restarts crashed containers.
- **Why a rolled-back transaction rather than a pure preview function (D1):**
  `_find_or_create_deployment` (`ledger.py:324-342`) holds the coalescing rules
  (compat key, capacity, sharing, IDLE→LIVE reuse). Running the real code inside
  a transaction that rolls back guarantees the preview and the commit agree. A
  separate pure function would duplicate those rules and could drift.
  `store.transaction()` is `BEGIN IMMEDIATE` with rollback on exception
  (`store.py`), but it does not nest. `Ledger.acquire` therefore needs splitting
  into a transaction-free `_acquire_core(conn, ...)` used both by the existing
  public method and by the preview.
- **Lock cost:** the SQLite write lock is held during planning. That is acceptable
  because `_global_lock` already serialises writers, and WAL readers such as
  `status` and the TUI are unaffected. Planning must not call Docker for anything
  slow inside the transaction: take the residency snapshot **before** opening it.

### 4.6 Publishing a render → I7, and decision D3

A render is published (compose file and sidecar written) only if every admitted
LIVE deployment is placed, **or** under the degraded rule:

- An admitted LIVE deployment that cannot be placed (GPU gone, global `reserved`
  change) is **degraded**. Its service stanza is carried over from the last
  published render **unchanged**, so `up --remove-orphans` neither removes nor
  moves its container. It is reported in `status`, in `leases`, and in the
  acquire output.
- If its stanza cannot be carried over (no previous render), publish without it
  but **do not apply**. Fail loudly with the reason.
- **Releases must still apply.** A release render that removes deployments is
  publishable even while another deployment is degraded.

D3 asks the reviewer to confirm that "carry the stanza forward" is acceptable. It
means the compose file can contain a service that `plan` could not place.

### 4.7 Apply with a handoff barrier → I9

In `ComposeBackend.apply()` (`compose.py:~2030-2070`), under the existing apply
lock, **before** `docker compose up -d --remove-orphans`:

```text
residency = self.residency()                 # unknown -> abort apply, retry later
for resident r not in the new render:
    if r.gpus ∩ (GPUs assigned to any deployment in the new render) ≠ ∅:
        docker stop r.container_id           # respects the stop grace period
        wait until docker inspect shows not running (bounded; timeout -> abort apply)
then: docker compose up -d --remove-orphans
```

- Only residents whose GPUs are being **reassigned** are stopped first. Other
  departing residents are left to `--remove-orphans` as today.
- `restart: unless-stopped` does not restart a container stopped with
  `docker stop`.
- The fake-runner test must reject the start of a service whose GPU is held by a
  container the runner still considers running. It must assert the **order** of
  calls, not the final container set.

### 4.8 Rollback after admission → goal 5

With transient admission, a request that was never admitted needs no rollback.
Rollback remains only for a request that was admitted but then failed readiness
(`test_wait_ready_timeout_keepwarm_idles_deployment`). In `_rollback_acquire`
(`controller.py:610-646`), replace the lenient `observe()` used to decide
"never ran" with `residency()`. On `ResidencyUnknown`, **do not evict**: leave
the deployments IDLE. At worst that is a phantom candidate; I3 already ignores
non-resident IDLE deployments.

### 4.9 Observability

- `status` distinguishes **UNKNOWN** (residency check failed) from **NOT RUNNING**
  and from **DEGRADED** (I7).
- Placement and queue messages name what is blocking: "waiting on GPU *k*, held by
  admitted lease L (deployment D)". Idle keep-warm can no longer block, so it
  should never appear there.
- Queue timeout errors say whether admitted demand or an unknown residency check
  prevented admission.

---

## 5. Migration and compatibility

- **Ledger schema (D2-a):** add a nullable `assigned_gpus` to `deployments`. On
  first start after upgrade, existing LIVE deployments have no committed
  allocation. Backfill from the residency snapshot if it is known, otherwise from
  the sidecar's assignments, which is the one-time use of the old source. Then
  stop reading pins from the sidecar.
- **Existing IDLE keep-warm deployments that are not running**, as in this
  incident, need no migration. I3 simply stops desiring them.
- **`infer-stack apply` by hand** now goes through the barrier, and refuses to
  apply when residency is unknown. It must print why.
- **Tests that construct plans without `required_ids`** keep passing via the
  compatibility default (4.2).
- **`null` / `kubeai` backends:** no optional residency and no barrier.
  Admission atomicity (4.5) still applies to them, because it is backend-agnostic.

---

## 6. Tests

Each test names the invariant it guards. "Real planner" means `plan_placement` on
`simulate_inventory`, not `BudgetBackend`.

**Placement (I2, I3, I10)**

1. Old **unpinned** idle resident vs new LIVE on a full host: LIVE placed; idle
   in `displaced`. This is the incident's shape; the investigation's R2 table
   gives the exact fixture.
2. Old **pinned** idle resident vs new LIVE: the LIVE request wins the pinned GPU.
3. Idle keep-warm that is **not resident** is absent from the plan even with
   free GPUs.
4. Re-acquiring a warm deployment (IDLE→LIVE reuse) keeps its GPU and container.
5. A pinned LIVE deployment outside the caller's `allowed_gpus` keeps its pin
   (I10 regression).
6. With `required_ids` omitted, plans are byte-identical to today's for the
   existing fixtures.

**Admission (I1, I4, I5, I6)**

7. Non-queued `acquire` blocked only by warm cache succeeds immediately.
8. Queued `acquire` blocked by **admitted** demand: every failed retry leaves the
   ledger, desired generation, compose file and sidecar byte-identical.
9. An unrelated request that fits is admitted while another request is waiting
   (the admitted/pending boundary).
10. A release applies while another request is waiting.
11. Two processes preview the last free GPU: exactly one is admitted (real
    `flock`, subprocesses).
12. A waiting caller's sweep of an expired admitted lease frees capacity,
    publishes the admitted-only render, and writes nothing of the candidate's.
13. A request killed while waiting leaves nothing to clean up.
14. Successful coalescing admission still bumps the desired generation (drift
    healing preserved).

**Publishing and concurrency (I7, R3)**

15. Another process's `_ensure_applied` cannot apply a render containing a
    not-yet-admitted request. With I5 no such render can exist; assert that.
16. An admitted deployment made unplaceable (simulated GPU loss) is DEGRADED:
    its stanza is carried forward, orphan removal leaves its container alone,
    and a release elsewhere still applies.

**Residency and barrier (I8, I9, goal 5)**

17. `residency()` maps by label, not the sidecar: a running container whose name
    is no longer in the sidecar is still reported.
18. `residency()` raises `ResidencyUnknown` on a Docker error; `observe()` still
    returns `set()` (existing contract, `test_leasing_compose.py:494`).
19. Unknown residency: optional residents are not dropped, nothing is displaced,
    and apply is refused with a reason.
20. Barrier: the fake runner records `stop(B)` → `inspect` shows B not running →
    `up`, in that order, when E takes B's GPU. It fails if `up` precedes B's exit.
21. Barrier does not stop residents whose GPUs are not being reassigned.
22. `_rollback_acquire` under `ResidencyUnknown` evicts nothing.

---

## 7. Verification before implementation

- **V1 (host):** on a running vLLM container,
  `docker inspect --format '{{json .HostConfig.DeviceRequests}}' <container>`
  shows the GPU indices written by `_gpu_reservation`. If they appear elsewhere,
  for example only in `Config.Env` as `NVIDIA_VISIBLE_DEVICES`, 4.1 must read
  that instead.
- **V2 (host):** `docker ps --filter label=infer-stack.deployment` lists every
  model container, including ones from renders no longer in the sidecar.
- **V3:** the Compose ordering claim (orphan removal is independent of service
  creation) was stated by the reviewer and not verified here. The barrier does not
  depend on it, but a quick host experiment would confirm the barrier is
  necessary and not just defensive.

---

## 8. Implementation order

Each step is independently mergeable and leaves the suite green. The barrier
lands **before** anything can displace a running resident.

| step | content | behaviour change |
|---|---|---|
| **P1** | `residency()` + `ResidencyUnknown` (4.1); tests 17-18; fix `compose.py:161` docstring | none |
| **P2** | `plan_placement(required_ids=...)` with compatibility default (4.2); tests 1-2, 5-6 at planner level | none for existing callers |
| **P3** | handoff barrier in `apply()` (4.7); tests 20-21 | none yet (nothing displaces) |
| **P4** | `_rollback_acquire` uses strict residency (4.8); test 22 | safer rollback |
| **P5** | ledger `assigned_gpus` + migration (4.3, §5) | pins move off the sidecar |
| **P6** | desired split and optional residency (4.4); tests 3-4, 19 | **idle yields to live** |
| **P7** | preview-then-commit admission (4.5); tests 7-14 | **waiting requests write nothing** |
| **P8** | publish rule and degraded state (4.6), observability (4.9); tests 15-16 | degraded handling |

P6 and P7 together fix the incident. P1-P5 make them safe.

---

## 9. Decisions for the reviewer

| id | question | recommendation |
|---|---|---|
| **D1** | Preview by rolled-back transaction around shared `_acquire_core`, or a separate pure preview function? | Rolled-back transaction, for a single source of coalescing rules (4.5) |
| **D2** | Committed allocation in the ledger (`assigned_gpus`) or a split sidecar? | Ledger (4.3) |
| **D3** | Degraded LIVE: carry its stanza forward unchanged, publish without it and refuse to apply, or something else? | Carry forward; refuse to apply only without a prior stanza (4.6) |
| **D4** | On `ResidencyUnknown`: refuse admission and apply outright, or allow admissions that displace nothing? | Allow admissions that need no displacement; refuse anything that would stop or move a container |
| **D5** | `null` / `kubeai`: admission atomicity only, with no residency or barrier? | Yes |
| **D6** | Should the barrier also cover a deployment moving between GPUs (a re-pin), not just displacement? | Yes if P5 can change a LIVE deployment's GPUs; otherwise out of scope |
| **D7** | Degraded state: a new `DeploymentState`, or derived at render time? | Derived; avoids a state machine change |

---

## 10. Explicitly deferred

- FIFO or reserved admission for large requests (G). Introduce `PENDING` only
  then, probably as lease metadata without claims.
- Automatic re-warming of displaced keep-warm deployments (rejected under I3).
- Any change to `observe()`'s best-effort semantics.

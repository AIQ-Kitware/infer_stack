# Investigation: idle keep-warm deployments starve queued leases

- **Date:** 2026-09-16
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`, from a guest VM
  with read-only access to the serving host's catalog, ledger and rendered
  compose project. The GPU host itself was operated by a human.
- **Code examined:** this repo at `dev/0.7.1` (`30fa46b`); MAGNET's
  `magnet/leasing.py` for how leases are requested.
- **Status:** investigation only. **No code has been changed.** Everything below
  is for another reviewer to corroborate before any patch is planned.

Each claim is tagged:

- **[code]** read directly in the source, with a location;
- **[observed]** seen in logs, the ledger, or `nvidia-smi` during the incident;
- **[inferred]** a conclusion drawn from the above that has **not** been
  verified. These are the ones most in need of review.

---

## Summary

On a 4-GPU host, two batch jobs each requested a lease on two single-GPU
deployments. They waited in the admission queue, making no progress, while
`nvidia-smi` showed all four GPUs empty. Every retry assigned three of the four
GPUs to two **idle keep-warm deployments left over from a run eleven days
earlier**, so one of the two requested deployments could never be placed.
Because a lease is applied only once everything in it is placed, nothing was
started, not even the requested deployment that did fit.

Evicting the two idle deployments by hand (`infer-stack evict <alias> <alias>`)
unblocked the queue on its next retry, and the models came up.

The behaviour appears to follow directly from four properties of the current
code, none of which is individually a bug:

1. idle keep-warm deployments are part of the desired set;
2. placement ignores state and demand, ordering by pins and then creation time;
3. the admission queue waits for capacity to free but never creates it;
4. no test exercises a keep-warm deployment competing with a queued request.

The docstring does promise a resolution, "survive idle **until pressure**", but
no code path implements that pressure.

---

## The incident

### Host and request

**[observed]** Four identical GPUs, 97,887 MiB each. Serving via the compose
backend with a LiteLLM gateway.

**[observed]** The jobs were two shards of one pipeline stage, launched under
Slurm through MAGNET's per-node leasing. Each shard asked for the same pair of
endpoints: an answerer and an auxiliary model.

**[code]** MAGNET wraps each node as
`infer-stack run --endpoint <a>,<b> --timeout 1800 --queue -- <cmd>`
(`magnet/leasing.py`: `lease_timeout=1800`, `lease_queue=True`, and the argv
built at lines 136-139). So each job waits up to 30 minutes for placement.

### Deployments involved

Labels are used instead of model names. VRAM and parallelism are what matter.

| label | deployment id | ledger state | reclaim | tensor parallel | created |
|---|---|---|---|---|---|
| **I1** | `grp-04e5391bdb73` | IDLE | keep-warm | 1 | 09-05 19:18 |
| **I2** | `grp-723e460d31c4` | IDLE | keep-warm | 2 | 09-05 19:18 |
| **L1** | `grp-edb7a0e879ff` | LIVE (demand 2) | keep-warm | 1 | 09-05 19:29 |
| **L2** | `grp-f227afd92ac4` | LIVE (demand 2) | stop | 1 | 09-16 15:01 |

**[observed]** Read from `ledger.db` (table `deployments`, `spec` JSON). **L1**
was an existing deployment reused by the new leases, so its `created_at` is
eleven days old even though its demand is new.

### What every retry did, every ~5 s from 15:01 to 15:22

**[observed]** From the jobs' own output:

```
Converging 4 deployment(s): I1, I2, L1, L2
  placed I1 on GPU(s) [0]
  placed I2 on GPU(s) [1, 2]
  placed L1 on GPU(s) [3]
  placement: L2: need 1 eligible GPU(s) (>= 51.75 GiB) but only 0 free
rendered 4 service(s) to .../docker-compose.yml (not applied; `infer-stack apply` to bring it up)
```

**[observed]** Meanwhile `nvidia-smi` showed 2 MiB used and no processes on all
four GPUs. `infer-stack status` listed **L1** and **L2** as `STALE` ("recorded
live but no container is running"). **I1** and **I2** did not appear there at
all.

### The manual unblock

**[observed]** Between 15:22:16 and 15:22:23 the operator ran
`infer-stack evict` on I1 and I2. The next retry:

```
15:22:23 Converging 2 deployment(s): L1, L2
15:22:23   placed L1 on GPU(s) [3]
15:22:23   placed L2 on GPU(s) [0]
```

**[observed]** By about 15:32 the containers for L1 and L2 (and a later shard's
deployment on GPU 1) were `Running`, and the pipeline moved on to later shards.
The ledger then held only live deployments, and the compose sidecar held only
their assignments.

This works as a controlled experiment: removing the two idle keep-warm
deployments, and changing nothing else, made the unplaceable request placeable
within one retry.

---

## Mechanism: claims to corroborate

### C1. Idle keep-warm deployments are in the desired set  [code]

`controller.py:395-406`, `Controller.desired_deployments`: LIVE deployments,
plus IDLE deployments whose `spec['reclaim']` (default `keep-warm`) is
`keep-warm`. The module docstring (`controller.py:9`) states the same rule.

### C2. "Until pressure" has no implementation  [code]

`controller.py:13-15` describes keep-warm as "survive idle until pressure,
avoids cold-start thrash". `grep -rn pressure infer_stack/ docs/` finds that
docstring and nothing else. The only callers of `Ledger.evict_idle`
(`ledger.py:261`) are:

- `controller.py:637`: on release, for deployments that never ran;
- `controller.py:810`: the explicit `evict` command;
- `controller.py:831`: `gc(evict_idle=True)`.

None of these is triggered by a request that cannot be placed.

> Reviewer: please confirm there is no other path, such as in the backend or the
> TUI, that evicts idle deployments under contention.

### C3. Placement ignores state and demand  [code]

`placement.py`, `plan_placement`:

- deployments are first sorted by `(created_at, id)` (`_sorted`, line 222);
- **step 1** honours pins, in that order (line 298);
- **step 2** honours explicit placements (line 313);
- **step 3** fits the rest by `fit_order = (n_eligible, created_at, id)`
  (lines 339-350).

Nothing in any step looks at `DeploymentState`, lease demand, or whether a
deployment is running. On a host of identical GPUs `n_eligible` ties, so step 3
comes down to creation time. **L1**'s old `created_at` came from reuse, not age
of demand.

### C4. Pins can come from a render that was never applied  [code] + [inferred]

**[code]**

- `ComposeBackend.plan` (`compose.py:1649-1667`) loads pins from the sidecar
  (`leasing-compose-state.json`, `assignments`). Its docstring says the result
  reflects where deployments "are (for running ones) or *would* be (for
  not-yet-started ones)".
- `converge` writes the sidecar at `compose.py:2012`, **before** the
  `if not apply: return` at line 2016. So every non-applied render in the queue
  loop rewrites the pins.

**[inferred]** Once a render assigns an idle keep-warm deployment a GPU, later
renders honour that as a pin in step 1, ahead of any fit ordering. The idle
deployment keeps "its" GPU even though no container ever ran there. If so, a
fix that only reordered step 3 by demand would **not** resolve this incident.

> Reviewer: this is the claim I am least sure of in effect. The logs cannot
> distinguish whether I1 and I2 were placed by step 1 (pin) or by step 3 (fit
> order); both produce the same assignments here. Suggested check: a unit test
> in which a render with `apply=False` places an idle keep-warm deployment, then
> a new LIVE request that needs that GPU is rendered. Does the pin win?

### C5. The admission queue waits for capacity but never creates it  [code]

`controller.py:717-745`, in `acquire` with `wait_for_placement` (loop at line 739):

1. `_infeasible_alone` re-plans the request **alone**, via `plan_on_idle_host`,
   which drops unrelated deployments' pins (`compose.py:1669-1716`). The request
   fits, so it is not rejected as impossible. That part is correct.
2. It then loops: sleep, `_render()` (which calls `converge(..., apply=False)`,
   `controller.py:461`), and recompute `unplaced`, until the deadline.
3. Each render sweeps TTL-expired leases. It does not evict idle deployments.

So the queue can only succeed if something **else** frees capacity: another
lease releasing, a TTL expiring, or a human. Idle keep-warm deployments with no
lease never expire.

Note the asymmetry: `_infeasible_alone` already reasons "unrelated deployments
are contention, look past them", but the loop that follows never acts on that
conclusion.

### C6. A lease applies nothing until all of it is placed  [code] + [observed]

**[code]** In the queue loop the render is never applied. Apply happens only
after `unplaced` is empty (`controller.py:759`).

**[observed]** L1 was placed on every retry for 21 minutes and never started.

This is probably the right behaviour for lease atomicity. It is listed because
it hides partial progress: from outside, the host looks entirely idle.

### C7. The test suite does not cover this interaction  [code] + [inferred]

**[code]** `tests/test_leasing_controller_queue.py:95`: the helper `vreq` sets
`reclaim='stop'` by default. `test_acquire_queues_until_a_gpu_frees` and
`test_acquire_queue_times_out_when_never_freed` therefore never involve a
keep-warm deployment. They also use `BudgetBackend`, a slot counter, not the
real `plan_placement`.

**[inferred]** No test sets an idle keep-warm deployment against a queued
request that needs its GPU. This is based on test names and a grep for
`keep-warm`/`keep_warm`, not an exhaustive read. Reviewer: please search
`test_leasing_compose.py` and `test_leasing_coalesced_apply.py` in particular.

---

## Open questions about intended behaviour

**Q1. Why were I1 and I2 idle in the ledger but not running?**
The ledger says IDLE and desired, but there were no containers. Candidates: a
host reboot or `docker compose down` outside infer-stack since 09-05, or every
render since then was also stuck not applied. Whatever the cause, reconcile does
not seem to notice "desired but not running" for idle deployments, and `status`
does not show them (it listed only L1 and L2 as STALE). **Is a desired
deployment with no container meant to be restarted, dropped, or reported?**

**Q2. What should "until pressure" mean precisely?**

- (a) Evict idle keep-warm deployments only when a **waiting** request cannot be
  placed otherwise?
- (b) Place demanded deployments first in every render, so idle ones simply lose
  their GPUs and are not rendered?
- (c) Something coarser, such as idle keep-warm deployments never holding a pin?

(a) keeps the cold-start benefit longest. (b) is simpler but may tear down warm
models that a following job would have reused.

**Q3. When several idle deployments could be evicted, which go first?**
Oldest idle, largest footprint, fewest GPUs needed to satisfy the waiter, or
least recently used? Should eviction be minimal, freeing just enough?

**Q4. Should pins from a never-applied render be honoured at all?**
The `plan()` docstring says pins describe where a not-yet-started deployment
"would be", so this looks intentional. Is it intended for **idle** deployments,
which by definition have no demand to protect?

**Q5. Is it safe that a stuck queue keeps rewriting `docker-compose.yml`?**
Each non-applied render writes the compose file and sidecar to a plan that is
never applied. If a different process's coalesced apply, or a human running
`infer-stack apply`, ran at that moment, would it bring up the idle keep-warm
deployments from that plan?

**Q6. Should `created_at` be the tie-breaker?**
A reused deployment keeps its original `created_at` (L1: 09-05) regardless of
when its demand arrived, so "older" says nothing about priority.

**Q7. Timeouts.**
With `--queue --timeout 1800`, a job facing this condition burns 30 minutes of
its Slurm allocation and then fails. Should the queue say *why* it is waiting,
for example "waiting on GPUs held by idle keep-warm deployment X, which will not
free on its own"? That turns a silent 30-minute wait into an actionable message.

**Q8. The end of one job's log is unexplained.**
One shard's output ends at 15:31:42 with another `rendered 3 service(s) ... (not
applied)` line, after both its deployments were placed and while other shards
had containers running. Its subsequent state was not verified. Reviewer: check
whether this is a normal post-readiness re-render or a separate issue.

---

## Things believed true that deserve a skeptical second look

- **B1** [inferred]: the incident reproduces with **any** idle keep-warm
  deployment whose GPU a queued request needs. It does not depend on tensor
  parallelism, Slurm, or MAGNET, only on C1-C5.
- **B2** [inferred]: without an operator, this state is permanent. Keep-warm
  idle deployments have no TTL, so every future queued request that needs their
  GPUs waits out its timeout and fails.
- **B3** [inferred]: `infer-stack gc --evict` (or `evict --all`) before a
  batch run is a sufficient operational workaround. It is blunt: it also
  discards warm models the batch would have reused.
- **B4** [inferred]: C4, not C3, may be the proximate cause. Reordering by demand
  without addressing pins may not fix it (see C4).

---

## Candidate directions: not a plan

These are recorded only so a reviewer can argue with them. None is chosen.

1. **Evict on pressure inside the queue loop.** When `unplaced` remains and
   evicting idle keep-warm deployments (minimal set, some documented order)
   would make the request placeable, evict them and re-render. This implements
   C2's docstring literally.
2. **Demand-aware placement.** Place LIVE deployments before IDLE ones in both
   step 1 (pins) and step 3 (fit), so an idle deployment's pin is honoured only
   if nothing live needs the GPU.
3. **Do not persist pins for idle deployments** from non-applied renders.
4. **Observability only.** Name the blocking idle deployments in the placement
   warning and in the queue's timeout error, and have `status` show desired
   deployments that are not running.

Direction 4 is useful regardless of which of 1-3 is chosen.

---

## Tests a fix should add

Written as behaviour, not implementation:

- An idle keep-warm deployment on the only suitable GPU, plus a queued request
  for that GPU: the request is placed before its timeout, and the idle
  deployment is evicted or not rendered.
- The same, where the idle deployment's GPU is **pinned** from an earlier
  non-applied render (C4).
- The same on the real `plan_placement` and compose backend, not only
  `BudgetBackend` (C7).
- An idle keep-warm deployment that a queued request **reuses** is not evicted:
  the cold-start benefit is preserved.
- Several idle deployments where evicting one suffices: only that one is evicted
  (if Q3 settles on minimal eviction).
- A waiting request that no eviction could satisfy still times out, with a
  message naming what blocked it.

---

## Operational workaround in use

Before a batch run on a shared host, evict idle deployments the batch will not
reuse:

```bash
infer-stack evict <alias> [<alias> ...]    # or: infer-stack evict --all
```

Live deployments (active lease) are never evicted by this command
(`EvictCLI` help text).

---

## Review round 1 (2026-09-16): second reviewer, then re-verification

A second reviewer, a different model, read this report, the June design
journal, and the code at `1030c36`. Its findings are below, each re-checked by
the original author. The headline: **the diagnosis stands, C4 was framed wrongly,
C6 was too generous, and Q5's answer is yes.** It also proposes a fix shape, and
this section agrees with most of it and disputes one invariant.

### R1. The missing abstraction: required placement vs opportunistic residency. AGREED

`desired_deployments()` (`controller.py:395-406`) merges two different things
into one hard desired set:

- **LIVE:** there is lease demand; the deployment *must* be placed;
- **IDLE + keep-warm:** there is no demand; residency is a cache optimisation.

Once they are merged, `plan_placement` cannot know that one should yield to the
other. **[code, re-verified]** This is an unfinished piece of the original
design, not a regression: `dev/journals/claude.md:577-579` (June) says "the
reclaim model has no *pressure* concept yet: keep-warm idle groups stay up
forever ... the Compose backend will need a pressure-driven reaper when GPUs are
contended." The admission queue is what eventually exposed it.

### R2. C4 is real, but it amplifies the bug rather than causing it. AGREED; C4 revised

The reviewer confirmed the sidecar mechanism and, more importantly, showed that
pins are **not needed** for the first bad placement. Re-verified by running the
real planner against the incident's shape, with no repo changes:

```python
from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.models import Deployment, DeploymentState as S
from infer_stack.leasing.placement import plan_placement

def dep(gid, state, t, tp=1):
    return Deployment(gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
                      {'engine': 'vllm', 'runtime': {'tensor_parallel_size': tp}},
                      {}, state, t, t)

inv = simulate_inventory('4x96')
I1, I2 = dep('I1', S.IDLE, 1.0), dep('I2', S.IDLE, 1.0, tp=2)
L1, L2 = dep('L1', S.LIVE, 2.0), dep('L2', S.LIVE, 3.0)
ds = [L2, L1, I2, I1]
```

| call | assignments | unplaced |
|---|---|---|
| `plan_placement(ds, inv)`, **no pins** | I1→[0], I2→[1,2], L1→[3] | **L2** |
| with the stale pins `{I1:[0], I2:[1,2], L1:[3]}` | same | **L2** |
| with **only idle pins removed** `{L1:[3]}` | same | **L2** |
| idle deployments **not in the set** | L1→[3], **L2→[0]** | none |

The last row exactly matches the host's placement after the manual evict. So:

- **C4 revised:** stale pins are confirmed, but they only make the bad decision
  deterministic retry after retry; they do not cause it. My note that "C4, not
  C3, may be the proximate cause" (B4) was **wrong**. C3, state-blind ordering,
  produces the incident on its own.
- These would all be **incomplete** fixes: sorting only step 3 by demand (idle
  pins still win step 1); deleting only idle pins (row 3); evicting only inside
  the queue loop (a non-queued `acquire` also fails, because it rolls back on any
  unplaced request, `controller.py:746+`, despite reclaimable idle capacity).

### R3. Lease atomicity is local, not global. AGREED; C6 revised, Q5 answered yes

C6 said nothing applies until a lease is fully placed. That is true only within
the queueing caller's own control flow. **[code, re-verified]**

- `converge(..., apply=False)` writes `docker-compose.yml` and the sidecar before
  returning (`compose.py:2012` vs `2016`);
- `ComposeBackend.apply()` runs `docker compose up -d --remove-orphans` on
  **whatever file is on disk** (`compose.py:2054`);
- `_ensure_applied` (`controller.py:497-528`) snapshots `desired_generation()`,
  applies the on-disk file, and marks that generation applied.

So another process's coalesced apply, or a human's `infer-stack apply`, can bring
up a partial render belonging to a lease that has not placed all of its
deployments, and mark that generation covered. The reviewer's interleaving:

```text
queued Q:   desired_generation -> G; render I1, I2, L1 (L2 unplaced);
            write compose + sidecar; release render lock; sleep
other P:    take apply lock; sees desired_generation == G;
            docker compose up on Q's partial file; set applied_generation = G
```

This did not happen in the incident, since nothing else applied during the
21-minute wait, but nothing in the code prevents it. It belongs to the bug, not
merely to observability.

### R4. The proposed invariant, and the one part disputed

The reviewer proposes three invariants:

1. **Every LIVE deployment is required.** Agreed.
2. **An IDLE keep-warm deployment is reclaimable and never prevents placement of
   a LIVE one.** Agreed. Place all LIVE deployments ahead of idle keep-warm ones
   in **every** tier, pins included, then let idle ones take what remains. A
   reused warm deployment is protected automatically, because
   `Ledger._find_or_create_deployment` flips IDLE→LIVE before placement
   (`ledger.py:324-342`, re-verified).
3. **"No render with an unplaced LIVE deployment should become an applyable
   shared artifact."** **Disputed as stated.**

Why (3) is too strong: **[code, re-verified]** `Controller.acquire` writes the
lease and its deployments into the shared ledger **as LIVE before placement is
attempted** (`controller.py:698-716`: `ledger.acquire(...)` then `_render()`).
A lease waiting in the admission queue is therefore LIVE in *everyone's* desired
set for its whole wait. Today other callers are unaffected, because each checks
only its own requests (`unplaced = requested & set(rec.unplaced)`). Under
invariant (3) as written, any render while a queued lease remains unplaceable
would be unpublishable. That includes an unrelated lease acquiring a small model
that fits, and a release whose teardown is the very thing that would free the
GPU. That is **head-of-line blocking** of the whole host behind one waiting
request.

**Proposed refinement (for review, not decided):** the distinction that matters
is not LIVE vs unplaced but **admitted vs pending** demand.

- A lease in the admission queue should not yet be LIVE in the shared desired
  set: either a distinct PENDING state, or planned speculatively (like
  `plan_on_idle_host`) without writing anything.
- The publish invariant then becomes: **a render publishes placements only for
  admitted deployments, and never writes a compose file or pins on behalf of a
  pending lease.** That keeps R3's hole closed without blocking unrelated leases.

The reviewer's broader point stands: this should be fixed by invariants in the
desired set and the render, **not** by "the queue calls `evict_idle()` when
stuck".

### R5. New open questions raised by the fix shape

- **Q9. When should displaced idle deployments be evicted?** Marking them STOPPED
  (via `evict_idle`) is destructive: the warm cache is lost. If a queued lease
  re-plans every 5 s and ultimately fails for some *other* reason, such as
  LIVE-vs-LIVE contention, repeated evictions destroy cache for nothing. Should
  eviction happen only when it makes a pending lease **fully** placeable
  (minimal sufficient eviction, see Q3)?
- **Q10. Same-GPU handoff.** If a running idle deployment on GPU *k* is displaced
  and a LIVE one is rendered onto *k*, a single
  `docker compose up -d --remove-orphans` both removes the orphan and starts the
  new service. Is removal guaranteed to finish, and VRAM to be freed, before the
  new container allocates? Not verified. A wrong order means a transient
  out-of-memory on start.
- **Q11. LIVE-vs-LIVE priority.** Once idle deployments always yield, contention
  is only between LIVE deployments, where `created_at` still decides. Is that
  acceptable, or should admitted-before-pending apply there too (Q6)?

### Regression tests, merged list

From the reviewer, plus R4 and R5:

- old **pinned** idle vs new LIVE; old **unpinned** idle vs new LIVE;
- the same with `--queue`, and with an ordinary non-queued `acquire`;
- re-acquiring the warm deployment itself keeps it warm (IDLE→LIVE reuse);
- another process's `_ensure_applied()` cannot apply, or mark covered, a
  generation containing a partially placed pending lease;
- **an unrelated lease that fits is admitted while another lease is still
  queued** (guards against the head-of-line blocking in R4);
- a release still applies while another lease is queued;
- eviction under pressure is minimal and does not recur on every retry of a
  lease that stays unplaceable for another reason (Q9).

### Status of the original claims after this round

| claim | status |
|---|---|
| C1, C2, C3, C5, C7 | confirmed by the second reviewer |
| C4 | confirmed as a mechanism; **revised** to an amplifier, not the cause |
| C6 | **revised**: atomicity holds only locally; see R3 |
| B4 | **withdrawn** |
| Q5 | **answered: yes** |
| Q1, Q2-Q4, Q6-Q8 | open; Q2 and Q3 now shaped by R4 and R5 |

---

## Review round 2 (2026-09-16): architecture converging

The second reviewer accepted R4's objection and revised its design. In short:
**admission becomes a side-effect-free preview; no lease exists until the whole
request fits.** Most of this is agreed. The additions below come from
re-checking the parts of the design that depend on facts about current code.

### Agreed

- **No durable PENDING state in this fix.** A waiting request is planned
  transiently and is not in the ledger. `ACTIVE` lease means admitted; `LIVE`
  means required capacity. There is no half-admitted lease to clean up after a
  killed waiter, and nothing new in `demand()`.
- **Invariant, restated correctly:** every published render places every admitted
  LIVE deployment. A request waiting for admission is not LIVE, is not in the
  desired set, and cannot affect the published render. R4's head-of-line
  objection no longer applies, because an unrelated lease that fits is admitted
  regardless of what is waiting.
- **Pressure does not call `evict_idle()`.** IDLE keep-warm becomes *optional*
  placement. A displaced deployment stays IDLE and simply gets no assignment.
  `evict` keeps its stronger meaning: an explicit decision that the deployment
  is no longer wanted as cache.
- **Requiredness is explicit in the planner** (for example
  `plan_placement(..., required_ids=...)`), not derived from
  `Deployment.state`. A candidate that reuses an existing IDLE deployment must be
  required in the preview even though the ledger still says IDLE. Tier order:
  required pins → required explicit → required fit → optional pins → optional
  explicit → optional fit.
- **Q9 answered:** displacement happens only inside a preview that proves the
  whole candidate is admissible. A retry that still cannot admit changes nothing,
  so there is zero cache churn. A non-queued `acquire` blocked only by warm cache
  succeeds immediately; `--queue` is needed only when *admitted* demand blocks it.
- **Q10 answered: a teardown barrier is required.** A displaced resident is
  stopped, and its exit awaited, before anything starts on its GPU, and a fake
  runner test must prove the ordering, not just the final container set. The
  reviewer states that Compose schedules orphan removal independently of service
  creation; that was **not verified by this author**. The barrier is warranted
  either way, because nothing documents the order.
- **Q11 answered:** established LIVE deployments are non-preemptible, and
  admission is first-successful under the global lock. `created_at` survives only
  as a deterministic tie-breaker. The documented lack of FIFO fairness is left
  alone here.

### Concerns and refinements raised in re-verification

**A. Optional re-placement oscillates under batch workloads.** In the reviewer's
example, B is displaced by E and "when E later releases, the next reconcile can
place B again". That means reconcile **cold-starts a model nobody requested**
whenever capacity frees. The workload that hit this incident acquires and
releases a lease per shard every few minutes, so B would be started, then
displaced (through the Q10 barrier) by the next shard, then started again,
indefinitely.

*Proposed refinement:* optional placement **preserves** residency but never
**creates** it. An IDLE keep-warm deployment is optional **only while it is
actually running**. If it is not running (displaced, crashed, or gone after a
host restart), it is not desired at all, while staying IDLE in the ledger. A later
request that reuses it flips it to LIVE (`ledger.py:324-342`) and starts it then.
This also answers **Q1**: the incident's I1 and I2 were idle *and not running*,
so under this rule they would never have entered the plan, before any ordering
question arose. The cost: warm models are not automatically restarted after a
host reboot. **Open: does anything rely on that?**

**B. "Running" is not reliably observable today.** **[code, re-verified]**
`ComposeBackend.observe()` (`compose.py`) lists running compose services, then
maps service names to deployment ids **through the sidecar's `services` map**,
which every render rewrites, including renders that are never applied. It also
returns `set()` on any Docker error ("observe is best-effort"). Consequences:

- a render that drops a service name makes its still-running container invisible
  to `observe()`;
- one transient Docker error makes every warm model look not-running.

So refinement A and the Q10 barrier both need a source of truth for **which
deployment is physically on which GPU** that does not go through the sidecar:
container labels and device requests via Docker, or a separate record of the last
**applied** (not last rendered) assignments. And a failed observation must fail
safe: treat state as unknown and keep the last good render, rather than dropping
optional residents or skipping the barrier.

**C. The preview-and-commit must be one critical section.** The feasibility
preview, the lease insert, and the publish must happen under a single hold of
`_global_lock`, which is re-verified to be a cross-process `flock` plus a thread
lock. Otherwise two waiting processes can both preview "fits" and both commit,
overcommitting the host. The reviewer implies this; the plan should state it as
an invariant with a two-process test.

**D. Sweeping is a ledger write the queue depends on.** **[code, re-verified]**
`_render()` begins with `self.ledger.sweep()` (`controller.py:452`), which
reclaims TTL-expired *admitted* leases. The current queue relies on this to free
capacity held by crashed jobs. "Pending attempts never alter the ledger or
desired generation" therefore needs an explicit carve-out: a waiting caller may
sweep expired admitted state and, if that changes desired state, publish the
admitted-only render. Only the candidate's own effects are forbidden.

**E. "Never replace the last good render" can wedge the host.** An admitted LIVE
deployment can become unplaceable for reasons unrelated to admission: a GPU
disappearing, `reserved` changing, or a changed `allowed_gpus` under Slurm, which
is per call. If every render refuses to publish from then on, releases and
teardowns can never apply either. The plan must define this case, for example:
fail loudly, still publish renders that only *remove* deployments, and surface
the violation in `status`.

**F. Published is not applied.** The sidecar currently conflates "last rendered"
with "last applied". The barrier's question, *who is on GPU k right now*, needs
the latter. Even with the new invariant, a published render can fail to apply.

**G. Starvation will bite this workload.** Deferring FIFO is reasonable for this
fix, but small leases repeatedly beating a large one is the expected pattern when
per-shard leases mix one-GPU requests with two-GPU requests (answerer plus
auxiliary model). This should be recorded as a known limitation with a follow-up,
not an unstated one.

### Additional tests from this round

- an IDLE keep-warm deployment that is **not running** is never started by a
  reconcile it was not requested in (A);
- a Docker error during observation does not drop running warm residents and does
  not skip the teardown barrier (B);
- two processes previewing the last free GPU: exactly one is admitted (C);
- a waiting caller's sweep of an expired admitted lease frees capacity and
  publishes, with nothing of the candidate's written (D);
- with an admitted deployment made unplaceable, a release still publishes and
  applies its teardown (E);
- the barrier consults physical placement, not the last rendered sidecar (B, F).

### Where this leaves the design

Converged: transient admission, required vs optional placement, no eviction
under pressure, the teardown barrier, and no PENDING state. **Still to settle
before a plan:** A (should optional residency require that the deployment is
running?), B/F (what the physical-placement source of truth is), and E (behaviour
when an admitted deployment becomes unplaceable).

---

## Review round 3 (2026-09-16): open items settled; plan written

The second reviewer re-checked against `e751676`, a merge of other work with no
leasing changes since `a02c730`, and settled the three open items. Each point
was re-verified by this author. The implementation plan is
[`plan-keep-warm-admission-2026-09-16.md`](plan-keep-warm-admission-2026-09-16.md).

**Settled.**

- **A = yes:** optional keep-warm residency requires a container that is actually
  resident. This generalises existing intent rather than inventing it.
  `_rollback_acquire` already says "keep-warm only means something for a
  deployment that came up" and evicts idled deployments that never ran
  (`controller.py:612-637`, re-verified).
- **B/F = a strict physical-residency snapshot from Docker:** deployment identity
  from the `infer-stack.deployment` label, which every vLLM and Ollama service
  already carries (`compose.py:352`, `413`), and GPUs from the container's actual
  device reservation. Not the sidecar, and not by changing `observe()`, whose
  empty-on-error result is a tested contract
  (`test_observe_tolerates_unreadable_compose_file`, `test_leasing_compose.py:494`).
  Note the existing contradiction: the service-naming docstring
  (`compose.py:161`) says `observe()` correlates containers by that label, but
  `observe()` actually maps names through the sidecar.
- **E = admitted-but-degraded:** an admitted LIVE deployment whose allocation
  becomes impossible stays admitted and surfaced. Its failure must not silently
  revoke its lease, and must not block unrelated releases.

**Corrections to earlier rounds.**

- **E's Slurm example was wrong.** Pins are validated against the **full**
  physical pool, deliberately ignoring the calling job's `allowed_gpus`
  (`placement.py:250-258` and `pin_pool_set`). A different job's allow-list cannot
  invalidate an established allocation, and the plan must preserve that. Genuine
  invalidation is: a GPU physically disappearing, a global `reserved` change, or
  runtime state inconsistent with the committed allocation.
- **"Won't come back after a reboot" was overstated.** Services render with
  `restart: unless-stopped` (`compose.py:351`, `412`), so a container that
  survives a reboot is restarted by Docker. Only a displaced, removed container
  stays gone, and that is the intended outcome.

**A new instance of B found while planning.** `_rollback_acquire` decides
"never ran" with the lenient `observe()`. On a transient Docker error that
returns `set()`, so a rollback would **evict every deployment the release
idled**, including warm residents that were running fine. The plan replaces this
with the strict snapshot.

**Current behaviour under E (reviewer, re-verified):** an unplaceable LIVE
deployment is omitted from the plan, the render skips deployments without an
assignment (`compose.py:1099`, `1118`), and the next
`up -d --remove-orphans` (`compose.py:2054`) would remove its container. The
degraded rule has to stop that.

---

## Review round 4 (2026-09-16): plan revision 1 reviewed

The second reviewer read plan revision 1 (`5244229`) and accepted its
required/optional/admission model, but found the render/apply layer beneath it
too weak for its invariants. Seven findings; all were accepted after
re-verification, and the plan is now at **revision 2**. Its §0 lists each finding
and its disposition. The most important, recorded here because it is a bug in
current code independent of keep-warm:

**Pre-existing generation race [code, re-verified].** `acquire` bumps the desired
generation when it commits the ledger, then renders, under `_global_lock`
(`controller.py:699`). `_ensure_applied` holds only the apply lock
(`controller.py:516`); it snapshots `desired_generation()` (519), applies the file
on disk, and marks that generation applied (523). An applier running between the
commit and the file write applies the old project and marks the new generation
covered. The acquirer then skips its apply, and its deployment never starts. The
symptom is a readiness timeout with nothing wrong in any log. Plan step P2
(render generations, apply exactly one published bundle) addresses it.

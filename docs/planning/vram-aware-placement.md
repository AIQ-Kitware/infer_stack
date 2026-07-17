# VRAM-aware placement: endpoints declare what they need, GPUs satisfy what they can

**Status:** proposed 2026-07-17 · not started
**Origin:** eval_audit Qwen3.5 small-model planning (yardrat, heterogeneous
2-GPU host). Written down so the objective survives even if this particular
plan gets reconsidered.

---

## Objective (read this first — it outlives the plan)

> **An `infer-stack acquire <endpoint>` must land the deployment on a GPU that
> can actually run it — and only such a GPU — without the operator encoding a
> model→GPU mapping anywhere.**
>
> - Deployments **gladly take any eligible GPU** they can get their hands on.
> - Ineligible GPUs are **never tried** (no OOM-at-container-start as the
>   discovery mechanism).
> - When no eligible GPU is *free*, the request **queues** (existing
>   `acquire --queue` semantics).
> - When no eligible GPU *exists* on the host, the failure is a **clear
>   planning-time error** naming the endpoint, its requirement, and the host's
>   actual inventory.
> - The **same catalog works on every host** — a 48+16 workstation, a 4×96
>   server — because eligibility is computed against local inventory, not
>   configured per host.

Non-objectives (for now, but leave the door open):
- **Co-hosting** several small deployments on one big GPU. Deferred — but the
  internal bookkeeping must be *capacity subtraction*, not a boolean
  "GPU taken" flag, so co-hosting later is a policy flip, not a rewrite.
- **Multi-node placement, preemption, migration.** Still out of scope
  (KubeAI/k8s/Slurm territory), unchanged from the original design.

### Scope amendment being made here

`infer_stack/leasing/placement.py`'s module docstring declares
"bin-packing … explicitly out of scope". This plan **narrows that exclusion**:
*single-host eligibility filtering + greedy best-fit* is now in scope;
multi-node bin-packing and preemption remain out. At the scale this tool
targets (≤ 8 GPUs, a handful of deployments) greedy best-fit-decreasing is
adequate forever; we will not need an ILP solver and should not pretend
otherwise.

---

## Motivating scenario

yardrat has two GPUs:

| idx | card | VRAM | note |
|-----|------|------|------|
| 0 | Quadro RTX 8000 | 48 GiB | fits everything we serve |
| 1 | Quadro RTX 5000 | 16 GiB | fits ≤ ~13.6 GiB at `gpu_memory_utilization: 0.85` |

Models in play (Qwen3.5 family, fp16 weights): 0.8B ≈ 1.7 GB, 2B ≈ 4.5 GB,
4B ≈ 9.3 GB (all fit GPU 1), 9B ≈ 19.3 GB (**only** fits GPU 0).

Today's placer is count-based first-fit **by GPU index** with zero VRAM
awareness: it would happily assign the 9B to the 16-GiB card (OOM at
container start), or park the 0.8B on the 48-GiB card and block the 9B behind
it. The only remedies today are operator pinning
(`INFER_STACK_ALLOWED_GPUS`, one allow-list per leaser process) or the
undocumented `runtime: {gpu_indices: [...]}` — both are the operator encoding
the schedule by hand, which is exactly what the objective forbids.

## Current state (verified 2026-07-17, with citations)

- **Inventory already knows VRAM.** `hardware.detect_inventory()` records
  `memory_mib` / `memory_gib` per GPU (`infer_stack/hardware.py:67-68`).
  Placement ignores it.
- **Placement is count-based first-fit.** `plan_placement()`
  (`infer_stack/leasing/placement.py:108`) applies three tiers — pinned,
  explicit, first-fit — where first-fit is `hardware.first_fit()`
  (`infer_stack/hardware.py:98`): take the first N free indices, in index
  order. `required_gpu_count()` (`placement.py:72`) is tp×pp×dp — *how many*,
  never *which kind*.
- **The fit vocabulary already exists — in the wrong layer.**
  `leasing/suggest.py` has `min_vram_gib_per_replica` (field, line 66),
  `fits_on()` (line 156), and `_host_gpus()` (line 163) which already picks
  the *smallest* GPUs that fit, with a written rationale ("it must reserve
  enough on the tightest GPU the placer might choose"). None of this is
  consulted at placement time — it only powers the `suggest` command.
- **Ledger does no GPU assignment** (by design): `acquire()` tracks demand;
  the Compose backend places the whole live set at converge
  (`placement.py:1-22`, `compose.py` converge path). Queueing
  (`acquire --queue` → wait_for_placement) and re-planning per generation
  already exist — the scheduling *loop* needs no changes, only the *plan*
  step.
- **`plan_placement` is a pure function with its own test file**
  (`tests/test_leasing_placement.py`, 20 tests) — the ideal seam.
- **`simulate_inventory('4x96')`** (`infer_stack/hardware.py`) builds fake
  homogeneous inventories for tests; it cannot express a heterogeneous host
  yet.

## Options considered (kept for future reconsideration)

1. **Operator pinning per runbook** (`INFER_STACK_ALLOWED_GPUS=1` for the
   small-model schedule, `=0` for the big one). Works today, zero code.
   Rejected as the *default*: it is a hand-encoded schedule, per-host,
   per-model-split; it rots and it doesn't transfer between machines. It
   remains valuable as an *operator restriction* on shared hosts (its
   original purpose) and as the SLURM composition path.
2. **Per-endpoint `runtime: {gpu_indices: [N]}`** — functional but
   undocumented and unvalidated by `Catalog.errors()`; still manual mapping.
   Rejected.
3. **SLURM with typed GRES/constraints.** Doesn't dissolve the problem —
   someone still writes "this model needs that GPU type" in every job spec,
   plus a slurmctld/slurmd configuration project for a 2-GPU workstation.
   Notably `placement.py`'s `allowed_gpus` asymmetry was *designed* to compose
   with SLURM (`$SLURM_JOB_GPUS` per job); if a real cluster appears later,
   this plan slots underneath SLURM unchanged. Rejected for now.
4. **VRAM-eligibility placement in the planner** (this plan). The catalog
   declares a per-GPU VRAM requirement; the planner filters GPUs to eligible
   ones and picks best-fit. Chosen: smallest diff that achieves the
   objective, reuses machinery already present, deterministic, testable.

## Design

### 1. Catalog: a validated `placement` block on endpoints

```yaml
endpoints:
  qwen3-5-9b-base-single:
    engine: vllm
    model: qwen3-5-9b-base
    placement:
      min_vram_gib: 24        # per GPU (per tp shard); measured, not guessed
    runtime:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.85
```

- `min_vram_gib` is **per GPU**: with `tensor_parallel_size: N`, each of the
  N GPUs must individually satisfy it (same semantics as suggest.py's
  `min_vram_gib_per_replica`).
- **Declared, not sniffed.** It is a recorded substrate fact: works offline,
  survives HF metadata drift, and is auditable. (`infer-stack suggest`
  already computes the number; a helper can pre-fill it, the catalog still
  records it.)
- Validated by `Catalog.errors()` (positive float, warn if an endpoint's
  `gpu_memory_utilization × min_vram_gib` arithmetic is obviously
  inconsistent — exact rule TBD in implementation).
- A `placement:` block (rather than a bare scalar) leaves room for future
  keys: `min_compute_capability` (e.g. bf16 needs ≥ sm_80),
  `co_host: allow` (Phase-future), etc. Ollama runtime_hosts already use a
  `placement:` block for `gpu_indices` — this extends the same idiom to
  vLLM endpoints.
- **Undeclared ⇒ all GPUs eligible ⇒ exactly today's behavior.** Fully
  backward compatible; nothing in existing catalogs changes meaning.

### 2. Planner: eligibility filter + deterministic anti-starvation ordering

Within the existing three-tier structure of `plan_placement()`:

- **Eligibility:** a GPU is eligible for a deployment iff
  `memory_gib >= min_vram_gib` (and passes today's pool filters:
  allowed_gpus, display, reserved). The first-fit tier only ever considers
  eligible GPUs.
- **Most-constrained-first:** order unplaced deployments by
  (number of eligible GPUs ascending, then deployment name for determinism).
  The 9B (1 eligible GPU) places before the 0.8B (2 eligible GPUs).
- **Best-fit:** each deployment takes the *smallest* eligible free GPU
  (tie-break by index). Small models gravitate to the small card; the big
  card stays free for the model that needs it. This mirrors
  `suggest._host_gpus()`'s smallest-that-fits logic.
- **Pinned tier wins over new declarations:** an already-realized deployment
  keeps its persisted GPUs even if a newly added `min_vram_gib` says
  otherwise (stability across reconciles, same principle as today) — but log
  a warning so the operator knows the declaration and reality disagree.
- **Capacity accounting internally:** the planner tracks per-GPU
  `remaining_gib = memory_gib − Σ(placed min_vram_gib)`, with today's policy
  that any placement also marks the GPU **exclusive** (one deployment per
  GPU). Co-hosting later = relax the exclusive flag for endpoints that opt
  in; the arithmetic is already there.
- **Errors:** "no eligible GPU exists" fails the plan with endpoint name,
  required GiB, and the host inventory table — copy-pasteable. "No eligible
  GPU currently free" is not an error; it queues, as today.

### 3. Accepted limitation (document, don't solve)

Without preemption/migration (out of scope), a temporal ordering can still
block: if the 16-GiB card is busy and a small model is therefore placed on
the 48-GiB card, a 9B arriving later queues until the small model releases.
Best-fit minimizes how often this happens; `reclaim: stop` bounds how long it
lasts. Fine at this scale.

### 4. Test/verification plan

- Extend `simulate_inventory` to accept heterogeneous specs
  (e.g. `'48,16'` — comma-separated per-GPU GiB) so tests and demos can model
  yardrat exactly.
- New cases in `tests/test_leasing_placement.py`:
  - 9B(24 GiB) + 4B(12) + 0.8B(3) on `'48,16'` → 9B→GPU0, one small→GPU1,
    rest queue; **never** 9B→GPU1.
  - Regression: small model must not take the only GPU the big model fits on
    while a smaller eligible GPU is free (the anti-starvation ordering).
  - Undeclared requirements ⇒ byte-identical plans to today (backward
    compat over the existing 20 tests).
  - Pinned deployment keeps its GPU despite a new conflicting declaration
    (+ warning emitted).
  - tp=2 with per-shard requirement on mixed inventory.
  - `allowed_gpus` ∩ eligibility (both filters compose).
  - No-eligible-GPU-exists → planning error naming endpoint + requirement +
    inventory.
- End-to-end (yardrat, manual): two concurrent schedules, no
  `INFER_STACK_ALLOWED_GPUS` set, 9B + smalls in one catalog; observe smalls
  land on GPU 1, 9B on GPU 0.

## Phases

- **Phase 0 — pin semantics in tests.** Heterogeneous `simulate_inventory`;
  write the placement tests above against the agreed semantics (they fail).
- **Phase 1 — catalog.** Parse + validate `placement.min_vram_gib` on vLLM
  endpoints; thread through the resolved deployment spec to the planner
  input.
- **Phase 2 — planner.** Eligibility, most-constrained-first, best-fit,
  capacity internals with exclusive flag, error messages. Update the
  `placement.py` docstring (the scope amendment above).
- **Phase 3 — adoption (eval_audit side, tracked there).** Declare
  `min_vram_gib` in the Qwen3.5 catalogs (measured numbers); retire the
  plan to pin small-model runbooks via `INFER_STACK_ALLOWED_GPUS`; then the
  9B re-run and the small-model batch share yardrat under concurrent
  schedules with no GPU indices anywhere in config.
- **Phase 4 — future.** Co-hosting opt-in (multiple vLLM servers on one big
  GPU for very small models — the greedy capacity accounting from Phase 2 is
  the whole "knapsack" we need); additional `placement` keys such as
  `min_compute_capability`.

## Open questions

1. Exact `min_vram_gib` numbers for the Qwen3.5 family — measure on yardrat
   (weights + activation overhead at our `max_model_len`/batch settings),
   don't guess from weight bytes alone.
2. Should `suggest` learn to emit the `placement:` block into generated
   catalog entries? (Probably yes, cheap.)
3. Does the KubeAI backend need to *reject* `placement.min_vram_gib`
   (k8s owns placement there) or silently ignore it? Leaning: warn-and-ignore
   with a note that k8s resource requests are the equivalent mechanism.
4. Reservations (`acquire --reserve-gpus N`) stay count-based for now; should
   they later accept `--min-vram-gib`? Follow-up, not blocking.

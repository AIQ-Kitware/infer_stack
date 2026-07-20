# VRAM-aware placement: endpoints declare what they need, GPUs satisfy what they can

**Status:** proposed 2026-07-17 · open questions resolved same day (see
"Resolutions") · **Phases 0–3 implemented 2026-07-17** · Phase 4 in
progress (eval_audit) · Phase 5 not started

Phases 0–2: heterogeneous `simulate_inventory`, catalog
`placement.min_vram_gib`, eligibility + most-constrained-first + best-fit
planner with `GpuPlan.warnings`; legacy plans byte-identical.
Phase 3: `leasing/vram.py` (profile parser / floor / overlay /
OOM classifier), plan-time enrichment in the compose backend
(declared > measured-overlay > floor), the guided OOM hint on
not-ready acquires, `infer-stack measure <endpoint> [--record]`, and
kubeai warn-and-ignore. One semantics refinement discovered during
implementation, now in design §2/§3: the automatically-enriched floor gates
ELIGIBILITY only — a floored-but-undeclared deployment keeps legacy
index-order selection, so merely downloading weights never moves an
existing catalog's deployments on a host where everything fits.
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
  survives HF metadata drift, and is auditable. (Where the number comes
  from — a best guess, clamped by the weight-bytes floor, tightened by
  on-demand measurement — is design §3.)
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
- **Best-fit (declared deployments only):** each *declared* deployment takes
  the *smallest* eligible free GPU (tie-break by index). Small models
  gravitate to the small card; the big card stays free for the model that
  needs it. This mirrors `suggest._host_gpus()`'s smallest-that-fits logic.
  **Undeclared deployments keep legacy index-order first-fit** — not
  best-fit — so pre-declaration catalogs place byte-identically (an
  undeclared 9B on yardrat still lands on GPU 0 exactly as today, rather
  than being "best-fit" onto the 16-GiB card it can't run on).
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

### 3. Where the numbers come from: best guess first, measurement on demand

Precomputing `min_vram_gib` from first principles has been tried and does not
produce exactly-right numbers (activation/compile overhead, allocator
fragmentation, engine-version drift). But we don't need exactly-right up
front — we need a **best guess that usually works, a guided recovery when it
doesn't, and an automatic floor that makes the worst mistakes impossible**:

- **Declared best guess (the normal path).** The operator writes
  `placement.min_vram_gib` into the catalog from whatever they know
  (weight bytes + gut margin is fine). This is the number placement uses.
  Being somewhat wrong is acceptable in both directions: too high wastes an
  eligible small GPU; too low is caught by the failure path below. Neither
  is an OOM mystery.
- **Guided failure path (when the guess was too low).** If a serve dies at
  startup with CUDA OOM on a GPU the declaration said was eligible, the
  error message must be the *nice* kind: name the endpoint, the declared
  number, the GPU it failed on, and the **exact command that computes the
  right number** — `infer-stack placement measure <endpoint>` (name TBD).
  An OOM against a declared eligibility is a *diagnosed misdeclaration*,
  not a generic container crash.
- **Measurement (optional, on demand — not a pipeline stage).**
  `placement measure` serves the endpoint once on the biggest available
  GPU, derives the real requirement, prints it, and optionally records it.
  ⚠️ **Not** by reading `nvidia-smi memory.used`: vLLM preallocates KV
  cache to fill `gpu_memory_utilization`, so observed usage reflects the
  *knob* on *that card* (a 0.8B "uses" ~41 GiB of a 48 GiB card at 0.85),
  not the requirement. Instead parse vLLM's own memory-profiling breakdown
  from the serve log (model weights + non-torch + activation peak; our
  images are version-pinned so the format is stable) and add the chosen KV
  budget for our `max_model_len`/`max_num_seqs` settings plus a small
  margin. Keyed by (model, engine image, dtype, max_model_len) — the things
  that change it.
- **Weight-bytes floor (automatic, offline, always sound).** fp16 weight
  bytes (from the HF safetensors index, or the local cache once downloaded)
  is a guaranteed *underestimate* of need: a GPU that cannot even hold the
  weights can never be eligible — even if a declared guess says otherwise,
  the floor clamps it up. This alone prevents the catastrophic misplacement
  (9B, 19.3 GB weights, offered a 16 GiB card) with zero declarations and
  zero measurements. It is never treated as the final number.

Resolution order at plan time: **max(declared-or-measured, floor)** — the
overlay's measured value backs an absent declaration; the floor clamps
everything.

**Storage: measured values do not silently rewrite `catalog.yaml`.** The
catalog is a hand-edited, git-tracked recorded fact; a machine mutating it in
place is against the grain. `placement measure` records into a
machine-managed overlay under `data_dir` (e.g. `leasing/measurements.json`)
that the resolver consults, and/or prints the number for the operator to
paste into the catalog; an explicit promote command can copy overlay →
catalog as a git-diffable operator action.

### 4. Accepted limitation (document, don't solve)

Without preemption/migration (out of scope), a temporal ordering can still
block: if the 16-GiB card is busy and a small model is therefore placed on
the 48-GiB card, a 9B arriving later queues until the small model releases.
Best-fit minimizes how often this happens; `reclaim: stop` bounds how long it
lasts. Fine at this scale.

### 5. Test/verification plan

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
  - Floor clamp: a declared guess *below* the weight-bytes floor is raised
    to the floor (a 9B declared at 8 GiB still never lands on the 16-GiB
    card).
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
- **Phase 3 — floor + guided-failure path + on-demand measurement.**
  Weight-bytes floor wired into eligibility (design §3); the
  OOM-against-declared-eligibility error message pointing at
  `placement measure`; the `placement measure` command itself (profiling-
  breakdown parse → print + optional overlay record). Measurement is a
  tool, not a pipeline stage — the planner from Phase 2 needs no changes.
- **Phase 4 — adoption (eval_audit side, tracked there).** Declare best
  guesses for the Qwen3.5 family (weight bytes + margin); tighten via
  `placement measure` only if a guess fails; retire the plan to pin
  small-model runbooks via `INFER_STACK_ALLOWED_GPUS`; then the 9B re-run
  and the small-model batch share yardrat under concurrent schedules with
  no GPU indices anywhere in config.
- **Phase 5 — future.** Co-hosting opt-in (multiple vLLM servers on one big
  GPU for very small models — the greedy capacity accounting from Phase 2 is
  the whole "knapsack" we need); additional `placement` keys such as
  `min_compute_capability`.

## Resolutions (2026-07-17, with Jon)

1. **Where do the exact numbers come from?** → **Best guess first,
   measurement on demand** (design §3): the operator declares a best guess;
   the weight-bytes floor clamps unsound guesses automatically; an OOM
   against a declared eligibility produces a guided error naming the exact
   `placement measure` command that computes the right number (vLLM
   profiling-breakdown parse). Measurement is optional tooling, never a
   required pipeline stage; recorded values go to a `data_dir` overlay and
   reach `catalog.yaml` only by explicit promote — never a silent rewrite.
   Now Phase 3.
2. **Should `suggest` precompute `placement:` numbers?** → **No.** Tried;
   precomputed numbers are never exactly right (activation/compile overhead,
   fragmentation, version drift). Precomputation survives only as the
   weight-bytes *floor*, which doesn't need to be exact — just sound.
   Measurement owns precision.
3. **KubeAI/k3s backend policy for the new field** → **warn-and-ignore**
   (k8s owns placement there; resource requests/limits are the equivalent
   mechanism on that side).
4. **VRAM-aware reservations** (`acquire --reserve-gpus N --min-vram-gib`) →
   explicitly **deferred, no decision now**. Reservations stay count-based;
   revisit after Phases 0–3 land.

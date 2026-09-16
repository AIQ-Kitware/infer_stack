# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 6**
- **Author:** Claude Opus 5 (1M context), `claude-opus-5[1m]`
- **Status:** **plan for review.** Step **P1 is implemented** on `dev/0.7.1`
  (`675315c`); nothing else is written.
- **History:**

  | revision | commit |
  |---|---|
  | 1 | `5244229` |
  | 2 | `7be95ba` (addendum `90d4ca4`) |
  | 3 | `3aeb5c0` |
  | 4 | `f0bf2f1` |
  | 5 | `e7242f0` |

  Each revision's §0 records the review round it answers. This one answers the
  review of revision 5. It is deliberately narrow: the reviewer judged the
  architecture settled, and the user asked that it not grow.
- **Scope boundaries** (non-goals and known faults) are documented for users and
  contributors in
  [`docs/planning/known-limitations.md`](../../docs/planning/known-limitations.md#leasing-scope-boundaries-and-known-faults).
  §10 of this plan lists the same items.
- **Code baseline:** `dev/0.7.1` at `675315c`. That is P1 on top of `e751676`; the
  rest of the leasing code is unchanged.
- **Evidence:**
  - [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
  - [`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`](investigation-gateway-stale-upstream-after-recreate-2026-09-16.md)
    (reproduced on the host)

---

## 0. Changes from revision 5

The reviewer closed the three revision-4 gaps in architecture (compensation, the
frozen profile, and ownership), accepted D16, and **narrowed scope** under an
operating assumption from the user: the catalog is published before a leasing
workload, not edited during it. What remains is three mechanical corrections to
ownership, and binding P2a's interim publication to the render it applied. Every
code claim was re-verified.

| # | finding | re-verified | disposition |
|---|---|---|---|
| D19 | The catalog is configuration-time state, published before the leasing epoch. No hot catalog merge. | — | **Resolved and simplified** (I19, §4.2). The catalog digest goes in the profile, and acquiring an endpoint absent from the published profile fails with "publish the new configuration first". The live-catalog read at acquire is removed. Documented as a scope boundary |
| 1 | **D21: hash behaviour, not only the stanza.** The gateway and nginx configs are bind-mounted at stable paths, and today's code already stamps a hash of their **contents** on the label ("same trick as LiteLLM", `compose.py:845-858`, `1030-1045`). A stanza-only hash would call two different gateway configs the same realisation. | yes | **Accepted.** The ownership key is a **behavioural fingerprint**: the canonical stanza plus the hashes of the generation-owned files that service consumes (I11, §4.6) |
| 2 | **Owned is not the same as satisfying G.** An `exited` or `dead` container with the wanted fingerprint must not be "kept" and block a restart. `paused` holds its GPU but serves nothing. Two containers with the same wanted fingerprint must fail closed. | — | **Accepted.** `managed`, `wanted` and `satisfies` are distinct. A required service is satisfied only by a unique `running` container; `paused` is unpaused, `restarting` is left to Docker, and anything else is replaced. Duplicate wanted realisations are ambiguous (§4.6) |
| 3 | **D20: P2a must bind clearing the marker to applying that exact render.** Otherwise the round-4 mutable-file race persists during the P2a period. | — | **Accepted.** P2a renders **and applies synchronously under `_global_lock`**, and clears the marker only after that apply succeeds (§8). This is transitional; see D22 |
| 4 | A bundle must be self-sufficient for values infer-stack owns. Compose interpolates `${HF_TOKEN:-}`, the DB password and the master key at apply time (`compose.py:323`, `802`, `837`), and shell variables take precedence over `--env-file`. | yes | **Accepted.** Apply runs Compose with an **explicit environment**: the managed `.env` plus a fixed allow-list, never the caller's shell (§4.2) |
| — | User direction: anything marked out of scope must be documented in infer-stack's docs so it is not shoehorned in later. | — | **Done.** `docs/planning/known-limitations.md` gains a leasing section, separating today's behaviour (current) from decisions that constrain the redesign (design boundary), plus the two known faults with workarounds |

### A simplification question raised by this round (D22)

The reviewer accepts, as a transitional step, that P2a render and apply
synchronously under the global lock. If that serialisation is acceptable
**permanently**, a large part of P2b exists only to preserve coalesced,
concurrent applies, and could be dropped:

- immutable bundles and the `current` pointer;
- activation rows and digest reconstruction;
- `repair_required`;
- generation-relative route registries and materialisation;
- KubeAI bundles.

What would remain is a pending marker, a deterministic render from ledger plus
profile, and a synchronous apply under the lock. Crash recovery re-renders from
the ledger and re-applies.

The cost is that every acquire and release waits behind every other one's
`docker compose up -d`. That call returns once containers **start**, not once they
are **ready**, and readiness waiting already happens outside the lock. So the
question is how long `up -d` takes in steady state. Pulling images dominates, and
under the catalog assumption, images can be pulled at `config publish`. **V15**
measures `up -d` on the host.

**This revision does not adopt D22.** It is a candidate simplification for the
reviewer and the user to weigh against measured cost.

---

## 1. Problem

A request can wait out its admission timeout while every GPU is empty:

- idle keep-warm deployments share a hard desired set with live demand (R1);
- the planner ignores demand (C3);
- unapplied renders persist pins (C4) and publish partial projects (R3);
- the intended pressure release was never implemented (C2).

Underneath that:

- apply is not bound to its render (the round-4 race);
- many code paths mutate desired state outside any publication protocol;
- render inputs drift between commands;
- recreating containers can pin the gateway's traffic for one model onto another
  model's container (reproduced).

## 2. Goals and non-goals

**Goals**

1. Idle keep-warm residency never prevents admission.
2. Admission is lease-atomic, including renderability.
3. Every desired-state change is published crash-safely, whichever caller made it,
   and **publishes only what that change changed**.
4. An apply acts on exactly one published generation, including its auxiliary files
   and routes, and records it applied only once that is fully true.
5. infer-stack **never mistakes its own containers for orphans**, whether an apply
   failed, a process crashed, or a generation was superseded.
6. No GPU is handed to a container while a previous occupant is still there.
7. Nothing destructive is decided from a failed or ambiguous observation.
8. Traffic for a deployment reaches only that deployment's container.

**Non-goals:** FIFO or reserved admission; a durable `PENDING` state; preemption;
automatic re-pinning; automatic bundle GC; multi-node; changing `observe()`.

---

## 3. Invariants

I1-I10 and I12-I18 are as in revision 4, with I11 revised; I19 and I20 are new.

| id | invariant |
|---|---|
| **I1** | An `ACTIVE` lease is admitted, and its state is published or recoverably pending. From P5, a failed attempt has no lease. |
| **I2** | A `LIVE` deployment holds a hard committed allocation; an invalid one makes it DEGRADED; nothing creates a LIVE deployment without one. |
| **I3** | An `IDLE` keep-warm deployment is optional, and a candidate only while resident (`running`, `restarting`, `paused`). |
| **I4** | Admission is lease-atomic, including renderability. |
| **I5** | A waiting attempt writes nothing except discardable pre-commit staged bundles. |
| **I6** | Admission prepares and approves outside SQLite write transactions, then commits briefly after validating `admission_state_version`, all under one `_global_lock` hold. |
| **I7** | Every published generation contains every admitted LIVE deployment; DEGRADED deployments are excluded from runtime actions. |
| **I8** | Residency is a strict, project-scoped Docker snapshot. Failure means unknown; duplicates mean ambiguous. |
| **I9** | `current` is the single published authority. apply(G) materialises G's auxiliary files and routes and acts only on G. A missing or corrupt activated bundle fails closed. |
| **I10** | Barrier: nothing starts on a GPU while another container occupies it. An unmanaged container on the GPU, or holding the service's static address, blocks the apply. |
| **I11** | **Selective apply with content-addressed ownership.** A container's key is `(infer-stack.service, behavioural fingerprint)`, where the fingerprint hashes the canonical stanza **plus the generation-owned files that service consumes**. The container is **managed** if it was adopted at migration, or its key appears in **any retained bundle**. It **satisfies** G if its key is in G, it is the **unique** container with that key, and its state serves G. Apply(G) keeps satisfying containers. Managed containers that do not satisfy G are replaced or removed, except those of DEGRADED deployments. Duplicate wanted realisations are ambiguous and fail closed. Orphans are unmanaged project containers, reported and never removed implicitly. Ownership never depends on `applied_generation`. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`. |
| **I13** | Under unknown or ambiguous residency, only resource-neutral admission proceeds. |
| **I14** | A request routed for deployment D reaches only D's container. |
| **I15** | Recovery: every `_global_lock` entry and every apply first completes or repairs in-flight activations, and publishes pending state. |
| **I16** | Every desired-state mutator goes through the controller: preview then activation (admission class), or commit with a compare-and-clear pending marker (removal class). |
| **I17** | Read-only paths do not mutate. |
| **I18** | A generation is applied only if every part succeeded. In dynamic-routing mode that includes verified route reconciliation. |
| **I19** | **Frozen render profile.** Every generation carries the complete set of non-ledger render inputs, **including the catalog digest**. Admission-class and removal-class publications inherit `current`'s profile unchanged. The profile changes only through an explicit, preview-first `config publish`. **The catalog is configuration-time state:** acquiring an endpoint absent from the published profile fails with "publish the new configuration first". |
| **I20** | **Non-reversible compensation.** An activation whose staged bundle is lost is **reconstructed** from its immutable inputs (committed ledger, base generation, profile). It auto-activates only if the reconstruction's digest equals the approved digest; otherwise infer-stack fails closed into an explicit repair. A lease release is never used as compensation. |

---

## 4. Design

### 4.1 Strict residency → I3, I8, I11

**Implemented in P1** (`infer_stack/leasing/residency.py`,
`ComposeBackend.residency()`). It covers deployment-labelled containers: every match
is kept, the scope is the project label, GPUs are read from `DeviceRequests`,
unmappable reservations count as every GPU, and failure raises `ResidencyUnknown`.

**P2b extension:** list **all** containers carrying the project label, not only
deployment-labelled ones, and record for each:

- `infer-stack.service`, falling back to `com.docker.compose.service`;
- `infer-stack.config-hash`, when present.

Ownership (§4.6) and generation-0 adoption both need the infrastructure services
(Postgres, LiteLLM, Open WebUI, reverse proxy), which carry no deployment label.

### 4.2 Publication, profile, bundles, recovery → I1, I9, I15, I16, I19, I20

**Render profile (I19).** Every bundle includes `profile.json`, holding every
non-ledger render input:

- backend kind and Compose project name;
- LiteLLM on or off, master-key reference (a name, never a value), ports;
- UI on or off, and its port;
- dynamic routing on or off;
- reverse proxy on or off, its port, and its config source;
- **resolved image pins** (not package defaults);
- the catalog basis used for the static-superset route registry;
- network configuration (subnet, §4.7);
- for KubeAI: namespace, base URL, resource profile.

- **Admission-class and removal-class publications** read the profile from
  `current`'s bundle and apply only their ledger change.
- **Drift detection:** every command still resolves settings as today, compares them
  with `current`'s profile, and on a difference warns that local settings differ
  from the published profile, pointing at `infer-stack config publish`. It never
  applies them implicitly.
- **`infer-stack config publish`** is an admission-class (preview-first) publication
  of a new profile. A profile change can make state unpublishable, for example
  removing a route basis or disabling LiteLLM while dynamic routing is in use.
- **A package upgrade** that changes `PINNED_IMAGES` alters nothing until a
  profile is published.
- **Catalog:** configuration-time state (D19). `config publish` records the
  catalog's digest, and the endpoint definitions and route basis derived from it,
  in the profile. Acquire resolves endpoints **only from the published profile**.
  An endpoint added to the catalog afterwards is not acquirable until the next
  `config publish`, and the error says so. Editing the catalog during a workload
  changes nothing until it is published.
- **Explicit apply environment:** Compose is invoked with an environment built
  from the managed `.env` plus a fixed allow-list, never inherited from the
  caller's shell. Shell variables otherwise take precedence over `--env-file` in
  Compose interpolation (`${HF_TOKEN:-}`, the DB password, the master key), so the
  same bundle could mean different things from different shells.

**Store:**

- `generation_counter`;
- `applied_generation`;
- `admission_state_version`;
- `publication_pending(version)`;
- `activation(G, staged_id, lease_ids, base_generation, rendered_version,
  approved_digest, created_at)`;
- `repair_required(G, reason)`.

**Admission class** (acquire, renew re-admission, `config publish`): preview, stage,
approve, then a short commit that inserts the activation with
`base_generation = current` and `approved_digest = sha256(staged bundle)`. Activate
follows. Detailed in §4.5.

**Removal class** (release, sweep, evict, gc, rollback): one transaction applies the
mutation, bumps `admission_state_version`, and sets `publication_pending`. Then call
`publish_pending()`:

1. snapshot the ledger;
2. `prepare()` on `current`'s profile and route registry;
3. stage;
4. short commit inserting the activation (`base_generation = current`);
5. activate.

**Activate** (idempotent):

1. rename the staged directory to `gen-<G>`, then fsync;
2. atomically replace `current`;
3. short commit that deletes the activation and compare-and-clears the pending
   marker.

**Recovery (I15, I20)**, at every `_global_lock` entry and every apply:

1. **Activation row, directory present** (staged or `gen-<G>`): finish activation.
2. **Activation row, directory absent: reconstruct.**
   - `prepare()` from the committed ledger, the `base_generation` bundle's profile and
     route registry, and the activation's rendered version.
   - Stage it, and compute its digest.
   - If the digest equals `approved_digest`, activate.
   - Otherwise, or if the base bundle is also missing: write `repair_required`, keep
     the activation row, and **fail closed**.
3. **Stray staged directories** (not named by any row): delete.
4. **`publication_pending` set,** and no repair required: `publish_pending()`.

**While `repair_required` is set:**

- no admission-class publication proceeds;
- removal-class mutations still commit their ledger change and pending marker, but
  are **not published**, because any new generation would include the
  unverified committed change;
- `status` names the activation and the reason.

`infer-stack repair activation` re-renders from the committed ledger and base,
shows the diff against `base_generation`, takes approval, activates, and clears the
flag. **Determinism is required:** a render must be byte-identical given the same
ledger, profile and base registry. That extends the existing byte-stability goal,
and is tested.

**Bundle contents** (Compose): `docker-compose.yml`, `manifest.json`,
`profile.json`, and `aux/` holding `litellm_config.yaml`, `nginx.conf`,
`routes.json` and `route_registry.json`. Paths inside the compose file point at the
stable runtime paths (D9).

**apply(G)** runs under the apply lock:

1. recovery; `G := current`;
2. fail closed on a missing, corrupt or unvalidated `gen-<G>`;
3. materialise `aux/*`;
4. barrier and selective start (§4.6);
5. in dynamic mode, reconcile and verify routes; on failure, stop without advancing;
6. `applied_generation = G`.

Retention: keep every bundle. Content-addressed ownership (§4.6) depends on that, so
any future GC must preserve every (service, hash) pair still present on a container.

### 4.3 Planner → I2, I12

Unchanged: `required_ids`, `hard`, `optional_hints`. Hard allocations degrade, never
re-place; optional residents are never newly fit; defaults are unchanged.

### 4.4 Ledger → I2

Unchanged: `assigned_gpus` is set at admission commit and cleared with any LIVE→IDLE
change; IDLE→LIVE reuse adopts the unique resident's GPUs; the desired set splits
into required and optional.

### 4.5 Admission → I1, I4-I6, I13, I15

Unchanged from revision 4 in shape:

```text
recover; maintenance_sweep; residency; snapshot; overlay_acquire; prepare; stage; approve;
short commit (validate admission_state_version; commit overlay;
              record activation with base_generation and approved_digest); activate.
```

- **Profile:** `prepare()` uses `current`'s profile and route registry.
- **`admission_state_version`** bumps on:
  - demand-changing lease transitions;
  - claims;
  - deployment create, state, spec or `served` changes;
  - `assigned_gpus`;
  - reservation ownership;
  - route registry and profile publications.

  It does **not** bump on:
  - `applied_generation`;
  - activation or repair housekeeping;
  - the pending marker;
  - TTL-only renewals of all-LIVE leases;
  - history pruning.

### 4.6 Selective apply, ownership, barrier → I7, I10, I11

**Labels.** Every rendered service carries:

- `infer-stack.service=<service name>`;
- `infer-stack.fingerprint=<behavioural fingerprint>`.

The fingerprint is `sha256(canonical stanza, excluding this label, ‖ sha256 of each
generation-owned file the service consumes)`. The gateway's includes its
`litellm_config.yaml`, and nginx's includes `nginx.conf`. Any future bind-mounted
generated file follows the same rule. This generalises today's content-hash label
(`CONFIG_HASH_LABEL`), already stamped on the gateway and proxy for exactly this
reason (`compose.py:845-858`, `1030-1045`).

The fingerprint changes exactly when the service's behaviour changes, so an
unchanged service is not recreated. That was verified on a real daemon: an
unchanged label value keeps the container `Running`, and a changed value recreates
it.

**Three predicates:**

- `managed(c)`: `c.id` is in `adopted_ids`, or `(c.service, c.fingerprint)` is in
  `owned_keys`. `owned_keys` is built from every retained manifest.
- `wanted(c, G)`: `(c.service, c.fingerprint)` is in G, and `c`'s deployment is not
  displaced in G.
- `satisfies(c, G)`: `wanted(c, G)`, **and** `c` is the only container with that
  key, **and** its state serves G:
  - `running`: satisfies;
  - `paused`: required services are unpaused, optional ones stay paused (warm but
    not serving);
  - `restarting`: left to Docker's restart policy and treated as satisfying for
    now; readiness decides whether it serves;
  - `created`, `exited`, `dead`, `removing`: do **not** satisfy; replaced.

```text
m = manifest(G); res = residency()              # unknown -> abort; G stays unapplied
for key in m.keys: if count(res containers with key) > 1: abort("ambiguous: <key>")
keep      = [c : satisfies(c, G)]
unpause   = [c in keep : c.state == paused and c.service is required]
departing = [c : managed(c) and not satisfies(c, G) and c.deployment not in m.degraded]
            # includes stale realisations left by failed or crashed applies,
            # stopped containers with a wanted key, and adopted containers whose service changed
orphans   = [c : not managed(c)]                                     # report only
to_start  = [svc in m : not degraded, not optional, no container in keep carries its key]

for gpu in gpus(to_start):
    blockers = res.occupants(gpu) - keep
    if any not managed(b) for b in blockers: abort("unmanaged container <id> occupies GPU <gpu>")
    for b in blockers: docker stop b; wait for exit (bounded, else abort)
for svc in to_start with a static address: if an unmanaged container holds it: abort
for c in departing: docker rm -f c.id
for c in unpause: docker unpause c.id
add new-or-changed dependencies of to_start (e.g. postgres)
docker compose -p P -f gen-G/docker-compose.yml up -d --no-deps <to_start, dependency order>
     # explicit environment (§4.2)
```

**Partial applies.** A container started by a failed or crashed apply of G
carries G's key, so it is managed:

- if the next generation still wants it and it is running, it is kept;
- if it is stopped, it is replaced;
- if it is no longer wanted, it is removed.

No transition record is needed.

### 4.7 Gateway routing correctness → I14

As in revision 4:

- explicit network, append-only `service_addresses`, operator-run
  `infer-stack network migrate`, in-gateway `python3` upstream check;
- low `AIOHTTP_TTL_DNS_CACHE` as a mitigation only.

**D18, resolved:**

- **Persistence.** At the first address allocation, `network_config(subnet)` is
  written to the ledger and to the profile. The settings value is only the default
  for that first write.
- **Mismatch.** A later difference between settings and the persisted subnet is
  rejected: "subnet is persisted as X; changing it requires
  `infer-stack network migrate --subnet Y`". It is never reinterpreted.
- **Allocation.** Addresses are sequential, skipping the network address, the
  gateway address (`.1`), the broadcast address, and any address Docker reports as
  reserved.
- **Preflight** before the first allocation: the subnet must not overlap an existing
  Docker network or a host route.

### 4.8-4.13

As in revision 4:

- **4.8 backends:** KubeAI bundles; Null has none.
- **4.9 rollback and readiness timeout:** removal class.
- **4.10 renew:** a lock-free fast path; a slow path through admission, which loses
  to an EXPIRED lease.
- **4.11 observability:** plus `repair_required` and profile drift.
- **4.12 route registry and route commands:** generation-relative; commands publish.
- **4.13 callers:** routed through the controller; read paths non-mutating.

---

## 5. Migration

1. **Schema:** `assigned_gpus`, `generation_counter`, `admission_state_version`,
   `publication_pending`, `activation`, `repair_required`, `service_addresses`,
   `network_config`, `adopted_containers`.
2. **Generation rebase:** keep the legacy counters as diagnostics; set
   `generation_counter = 1` and `applied_generation = 0`.
3. **Secrets:** provisioned if missing.
4. **Initial profile:** resolved **once**, from the current settings and catalog, and
   written into `gen-0/profile.json`. From then on only `config publish` changes it.
5. **Hard allocations:** backfilled from strict residency only; unresolved entries,
   including LIVE reservations, block new allocation.
6. **Adoption covers all project containers.** Every container carrying the project
   label is recorded in `adopted_containers`, with its service name, when it matches
   either:
   - a backfilled LIVE deployment, by label and GPUs; or
   - an infrastructure service in `gen-0`, by service name (Postgres, LiteLLM, Open
     WebUI, reverse proxy).

   Adopted containers are managed without a config-hash label. They stay until their
   service changes, and the first recreation stamps the label. **No container is
   recreated by migration.** Every other project container is an orphan.
7. **First publication** is `gen-1`; the old compose file and sidecar are retired
   after it applies.
8. **Addresses and subnet:** not here; see `network migrate` (§4.7).

---

## 6. Tests

Tests 1-65 are as in revision 4. **P1 has landed with tests 39-41.** New in revision
5 (**R** = reviewer, **A** = author):

66. **R:** G starts B, and route verification fails. A later H drops B: B is
    removed as managed, never reported as an orphan.
67. **R:** G crashes halfway through service changes, and H supersedes it. Survivors
    from both F and G are managed, and H removes exactly those it does not want.
68. **R:** G completes its Docker actions and dies before advancing
    `applied_generation`. Recovery and the next apply keep G's containers without
    recreating them.
69. **A:** an unchanged service across generations G and G+1 has an identical
    config-hash label and is not recreated. Real-Docker test.
70. **R:** a removal publication inherits the profile. With local settings changed
    (UI, dynamic routing, proxy, catalog), a release publishes a generation whose
    `profile.json` equals `current`'s, and no unrelated service changes.
71. **A:** a package upgrade changing `PINNED_IMAGES` changes no image until
    `config publish`.
72. **A:** `config publish` is preview-first. An unpublishable profile commits
    nothing.
73. **A:** a lost staged bundle whose reconstruction digest matches is activated
    automatically.
74. **R, A:** the served-alias merge case. An acquire adds an alias to an existing
    shared deployment, and the staged bundle is deleted after commit. Recovery does
    **not** release the lease: it reconstructs, and on a digest mismatch sets
    `repair_required`; admissions are refused; releases commit but do not publish;
    `repair activation` re-renders with approval.
75. **A:** determinism: the same ledger, profile and base registry produce a
    byte-identical bundle, across processes.
76. **R:** adoption covers Postgres, LiteLLM, Open WebUI and nginx containers; none
    is recreated by migration.
77. **R:** a subnet settings change after allocation is rejected; allocation skips
    the network, gateway, broadcast and reserved addresses; an overlapping subnet
    fails preflight.
78. **A:** P2a's interim path: the pending marker is published through the existing
    render/apply path and compare-and-cleared; a mutation during that publication
    survives.
79. **A:** the residency extension lists infrastructure containers with their
    service name and config hash.
80. **R:** a gateway config content change with an identical stanza changes the
    fingerprint and recreates the gateway. An identical config does not.
81. **R:** a managed container with the wanted key in state `exited` or `dead` is
    replaced, not kept.
82. **R:** two containers with the same wanted key make the apply fail closed as
    ambiguous.
83. **A:** a required `paused` container is unpaused, not recreated; an optional
    one stays paused.
84. **R:** P2a exact-render binding: a concurrent mutation during a P2a publication
    cannot make the marker clear for a render that was not applied.
85. **R:** apply ignores the caller's shell: with a conflicting `HF_TOKEN` exported,
    the applied container matches the managed environment.
86. **R:** acquiring an endpoint absent from the published profile fails, naming
    `config publish`. An edited but unpublished catalog changes nothing.

---

## 7. Host verification

V1-V12 as in revision 4 (V7 done). Also:

- **V13:** on the host daemon, a changed label value recreates a service and an
  unchanged one does not. Verified on the guest daemon for this revision; confirm
  that the host's Compose version agrees.
- **V14:** reconstruction determinism on a real stack: render twice from the same
  ledger and profile and compare digests.
- **V15 (decides D22):** on the host with images already present, time
  `docker compose up -d` for three cases: a no-op, one added model service, and one
  removed model service. Repeat with several concurrent callers.

---

## 8. Implementation order

| step | content | status or behaviour |
|---|---|---|
| **P1** | Strict residency for deployment containers | **done** (`675315c`); V1, V2, V8 pending |
| **P2a** | Pending marker written atomically by every mutator; every caller through the controller; read paths non-mutating. Recovery publishes pending state by **rendering and applying synchronously under `_global_lock`** through the existing path, and compare-and-clears the marker **only after that exact apply succeeds**; the apply runs with the explicit environment (tests 51-54, 78, 84-85) | every mutation is published after a crash; the mutable-file race is closed by serialisation, transitionally (D22) |
| **P2b** | Render profile and `config publish`; bundles, `current`, activation with digest; reconstruction and repair; content-addressed labels and ownership; residency extended to all project containers; apply-exactly-G, fail-closed, route verification; KubeAI bundles; migration (rebase, profile, adoption) (tests 21-28, 55-60, 66-77, 79) | replaces the mutable publication path |
| **P3** | Planner keywords (tests 1-7) | none by default |
| **P4** | `assigned_gpus`, backfill, renew fast and slow paths (tests 42-47) | allocations become hard |
| **P5** | Admission preview; acquire leaves the removal class (tests 8-20) | failed attempts commit nothing |
| **P5b** | Explicit network, persisted subnet, addresses, `network migrate`, upstream check (tests 48-49, 61-65, 77) | I14 |
| **P6** | Selective apply with ownership and barrier, `gc --orphans` (tests 29-38) | no `--remove-orphans` |
| **P7** | Idle keep-warm becomes optional | **fixes the incident** |
| **P8** | DEGRADED end to end, observability (test 50) | |

---

## 9. Decisions

**Resolved:** D1-D18.

- D16: accepted, with the frozen profile.
- D17: dynamic mode gates "applied" on route verification; static-superset mode does
  not.
- D18: persisted subnet, explicit migration to change it.

**Resolved this round**

- **D19:** the catalog is configuration-time state, published before the leasing
  epoch. Acquire uses the published profile only.
- **D20:** P2a publishes, serialising render and apply under the lock, and clears
  the marker only after that apply.
- **D21:** content-addressed ownership, with a behavioural fingerprint and distinct
  `managed`, `wanted` and `satisfies` predicates.

**Open**

| id | question | author's lean |
|---|---|---|
| **D22** | Keep serialised render-and-apply under `_global_lock` permanently, and drop P2b's bundle, activation, reconstruction, repair and generation-relative registry machinery? | **Decide after V15.** If `docker compose up -d` in steady state (images pre-pulled at `config publish`) takes a few seconds, serialisation is the simpler design and its cost is small. If it routinely takes tens of seconds under concurrent per-shard leasing, keep P2b as specified. Either way, P1, P3, P4, P5, P5b, P6 and P7 are unaffected |

---

## 10. Out of scope

Documented for users and contributors in
[`docs/planning/known-limitations.md`](../../docs/planning/known-limitations.md#leasing-scope-boundaries-and-known-faults).
Do not approximate these through existing code paths; each needs its own design.

- **Hot catalog or configuration changes** during a leasing epoch (D19).
- **FIFO or reserved admission**; a large request can be starved.
- **Preemption** of admitted leases.
- **Automatic migration** of LIVE deployments between GPUs.
- **Automatic re-warming** of displaced keep-warm deployments.
- **Multi-node placement.**
- **Forged ownership labels** (threat model).
- **Changing `observe()`** to strict semantics.
- **Automatic bundle GC**, if P2b is kept. It must preserve every owned key still
  present on a container.
- **Whether an unswept past-TTL ACTIVE lease counts as demand.** Left as today.

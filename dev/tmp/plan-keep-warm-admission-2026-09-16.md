# Plan: admission-atomic leasing with optional keep-warm residency

- **Date:** 2026-09-16 · **Revision 5**
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

  Each revision's §0 records the review round it answers. This one answers the
  review of revision 4, which the reviewer scoped narrowly: runtime ownership
  during incomplete applies, what a removal-class publication inherits, and
  compensation for a lost activation.
- **Code baseline:** `dev/0.7.1` at `675315c`. That is P1 on top of `e751676`; the
  rest of the leasing code is unchanged.
- **Evidence:**
  - [`investigation-keep-warm-placement-starvation-2026-09-16.md`](investigation-keep-warm-placement-starvation-2026-09-16.md)
  - [`investigation-gateway-stale-upstream-after-recreate-2026-09-16.md`](investigation-gateway-stale-upstream-after-recreate-2026-09-16.md)
    (reproduced on the host)

---

## 0. Changes from revision 4

The reviewer did not send the plan back for broad redesign. It closed D16 (with a
condition), D17 and D18, and named three blocking gaps plus one mechanical
inconsistency. Every code claim below was re-verified.

| # | finding | re-verified | disposition |
|---|---|---|---|
| 1 | **A failed or partial apply orphans infer-stack's own containers.** `prev = manifest(applied_generation)` stays at F when G's apply fails after starting containers (`up` runs before route reconciliation: `compose.py:2062`, `2066`), so a later H classifies G's containers as orphans. The same happens on a crash mid-apply. | yes | **Accepted, with a different mechanism** from the one the reviewer sketched. **Ownership is content-addressed** (§4.6): a container is managed if its (service, config hash) appears in any retained bundle, or it was adopted. Ownership no longer depends on `applied_generation`. The reviewer's three tests are adopted |
| 1′ | The reviewer's sketch stamped `infer-stack.generation=G` on containers. | **verified against real Docker** | **Not adopted.** A label value is part of the service's config. On a real daemon, changing a label from `1` to `2` made `docker compose up` **recreate** the container, while an unchanged label left it `Running`. A per-generation label would therefore recreate every service on every apply: every model (a cold reload) and the gateway (whose no-recreation design is deliberate). The stable alternative is a hash of the service's rendered stanza, which changes only when the service does |
| 2 | **D16 is monotone only if non-ledger render inputs are frozen.** Each command re-resolves global configuration (`commands_leasing.py`: UI and LiteLLM `112-136`, dynamic routing `139-156`, proxy `159-178`, KubeAI `267-276`), and the catalog is reloaded **on every converge, including release and gc** (comment at `commands_leasing.py:236-244`). A release could therefore publish unrelated configuration changes. | yes | **Accepted.** Every generation carries an immutable **render profile** (I19, §4.2). Removal-class and admission-class publications inherit `current`'s profile; profile changes are explicit, preview-first publications. **Addition:** image pins default from the installed package (`PINNED_IMAGES`, `compose.py:799`), so a package upgrade silently changes render inputs; pins are part of the profile |
| 3 | **Admission compensation is not reversible.** Reuse calls `_merge_served`, which writes a new alias into an **existing** shared deployment (`ledger.py`, `_merge_served`); releasing the lease does not undo it. | yes | **Accepted.** No inverse rollback. The activation records the approved bundle's **digest** and its **base generation**. Recovery **reconstructs** the bundle deterministically and auto-activates only if the digest matches; otherwise it **fails closed** into an explicit repair (I20, §4.2) |
| 4 | P2a claimed to implement `publish_pending()`, whose storage protocol belongs to P2b. | — | **Accepted.** P2a is the marker, controller routing and non-mutating reads. Until P2b it publishes pending state through the **existing mutable render/apply path**, compare-and-clearing the marker. P2b swaps that path for bundles (§8, D20) |
| 5 | Generation-0 adoption covered only containers with `infer-stack.deployment`. Postgres, LiteLLM, Open WebUI and nginx carry only `infer-stack.engine` (`compose.py:807`, `874`, `969`, `1053`). | yes | **Accepted.** Adoption covers every project container. **Addition:** P1's `residency()` lists only deployment-labelled containers, so P2b extends it to all project containers (§4.1) |
| D16 | Accepted, conditional on a frozen profile. | — | **Resolved** with I19 |
| D17 | Author's lean accepted. | — | **Resolved:** dynamic mode gates "applied" on verified routes; static-superset mode does not |
| D18 | A fixed default with override, but the subnet must be **persisted** once addresses are allocated, a mismatch rejected, reserved addresses skipped, and conflicts preflighted. | — | **Resolved** as stated (§4.7) |
| P1 | Ready to implement. | — | **Done**, `675315c`. Host checks V1, V2 and V8 are still to run |

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
| **I11** | **Selective apply with content-addressed ownership.** A container is **managed** if it was adopted at migration, or if its `(infer-stack.service, infer-stack.config-hash)` pair appears in **any retained bundle**. Apply(G) keeps managed containers whose pair is in G and whose deployment is not displaced; every other managed container is **departing**, except those of DEGRADED deployments. **Orphans** are project containers that are not managed; they are reported and never removed implicitly. Ownership never depends on `applied_generation`. |
| **I12** | Established allocations ignore the caller's `allowed_gpus`. |
| **I13** | Under unknown or ambiguous residency, only resource-neutral admission proceeds. |
| **I14** | A request routed for deployment D reaches only D's container. |
| **I15** | Recovery: every `_global_lock` entry and every apply first completes or repairs in-flight activations, and publishes pending state. |
| **I16** | Every desired-state mutator goes through the controller: preview then activation (admission class), or commit with a compare-and-clear pending marker (removal class). |
| **I17** | Read-only paths do not mutate. |
| **I18** | A generation is applied only if every part succeeded. In dynamic-routing mode that includes verified route reconciliation. |
| **I19** | **Frozen render profile.** Every generation carries the complete set of non-ledger render inputs. Admission-class and removal-class publications **inherit `current`'s profile unchanged**. The profile changes only through an explicit, preview-first profile publication. |
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
- **Catalog use at acquire:** the endpoint spec a new deployment is created from is
  read from the live catalog and captured into the deployment's spec in the ledger,
  as today. Only the catalog-wide **route basis** is profile state (D19).

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
- `infer-stack.config-hash=<sha256 of its rendered stanza, excluding this label>`.

This extends `CONFIG_HASH_LABEL`, today set only on the gateway and proxy
(`compose.py:770`), to every service. The hash changes exactly when the service's
config changes, so an unchanged service is not recreated. That was verified: an
unchanged label keeps the container `Running`, and a changed value recreates it.

**Owned-pair index.** A map built from every retained `gen-*/manifest.json`, from
`(service, config_hash)` to the generations containing it, plus the adoption set of
container ids (§5).

```text
m    = manifest(G); res = residency()            # unknown -> abort; G stays unapplied
managed(c) = c.id in adopted_ids or (c.service, c.config_hash) in owned_pairs
orphans    = [c in res.project_containers if not managed(c)]          # report only
wanted(c)  = (c.service, c.config_hash) in m.services_by_hash
             and c.deployment not in m.displaced_optionals
keep       = [c managed, wanted(c)]
departing  = [c managed, not wanted(c), c.deployment not in m.degraded]
            # includes containers started by a failed or crashed apply of an
            # earlier generation, and adopted containers whose service changed
to_start   = services in m, not DEGRADED, not optional,
             with no kept container carrying their (service, hash)

for gpu in gpus(to_start):
    blockers = res.occupants(gpu) - keep
    if any not managed(b) for b in blockers: abort("unmanaged container <id> occupies GPU <gpu>")
    for b in blockers: docker stop b; wait for exit (bounded, else abort)
for svc in to_start with a static address:
    if an unmanaged container holds it: abort
for c in departing: docker rm -f c.id
add new-or-changed dependencies of to_start (e.g. postgres)
docker compose -p P -f gen-G/docker-compose.yml up -d --no-deps <to_start, dependency order>
```

**Why this survives partial applies.** Suppose G's apply started container B, then
failed on route verification or crashed:

- B's pair is in `gen-G`, a retained bundle, so B is managed.
- If a later H keeps B's service unchanged, B is kept; if H drops or changes it, B
  is departing and removed.
- A crash between the barrier and `up` leaves stopped managed containers that the
  next apply removes or restarts.

No transition record is needed, because ownership lives on the container and in the
retained bundles, and both survive a crash.

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

---

## 7. Host verification

V1-V12 as in revision 4 (V7 done). Also:

- **V13:** on the host daemon, a changed label value recreates a service and an
  unchanged one does not. Verified on the guest daemon for this revision; confirm
  that the host's Compose version agrees.
- **V14:** reconstruction determinism on a real stack: render twice from the same
  ledger and profile and compare digests.

---

## 8. Implementation order

| step | content | status or behaviour |
|---|---|---|
| **P1** | Strict residency for deployment containers | **done** (`675315c`); V1, V2, V8 pending |
| **P2a** | Pending marker written atomically by every mutator; every caller through the controller; read paths non-mutating; recovery publishes pending state **through the existing mutable render/apply path** and compare-and-clears (tests 51-54, 78) | every mutation is published after a crash; no bundle claims yet |
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

**Open, for the next review**

| id | question | author's lean |
|---|---|---|
| **D19** | Catalog split: endpoint specs for **new** deployments are read from the live catalog at acquire (captured into the ledger), while the catalog-wide static route basis is profile state. Is acquiring an endpoint added to the catalog after the last `config publish` correct? Its route is added because a live deployment now demands it, while the superset refresh waits for `config publish`. | Yes. Demand-driven routes follow admission; catalog-wide refresh is configuration |
| **D20** | P2a publishes pending state through the existing mutable path until P2b. Acceptable interim, or should P2a only *detect* the marker (the reviewer's wording)? | Publish. A detect-only marker would be re-rendered on every lock entry until P2b, and publishing restores crash recovery immediately |
| **D21** | Content-addressed ownership (I11) instead of a transition record or per-generation labels (the latter verified to force recreation). Any case where a (service, hash) pair in a retained bundle wrongly marks a foreign container as managed? | Only a container hand-built with forged labels, which is out of scope |

---

## 10. Deferred

- FIFO or reserved admission.
- Automatic re-warming of displaced keep-warm deployments.
- Explicit migration of LIVE deployments between GPUs.
- Automatic bundle GC. It must preserve every owned pair still present on a
  container.
- Any change to `observe()`.
- Whether an unswept past-TTL ACTIVE lease counts as demand.

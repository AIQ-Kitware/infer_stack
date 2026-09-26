# Backend parity roadmap: KubeAI as a superset of Compose

**Status:** proposed 2026-09-25 · **P0 done** 2026-09-24 on
`dev/backend-unification` · **P1–P4, P6 done** 2026-09-26 (P4's GPU run is
a handover) · P5 not started. Execution order: [../queue.md](../queue.md).
**Current state:** [../backend-parity.md](../backend-parity.md).
**Origin:** the scale-up run needs more than one workstation, and the
KubeAI backend had drifted from Compose for three months before the
2026-09-24 audit (`dev/tmp/plan-backend-unification-2026-09-24.md`).

---

## Objective (read this first; it outlives the plan)

A card, a catalog and an operator's habits move from one workstation to a
cluster with one change: naming what a GPU unit is. Concretely,

1. every row of the parity matrix reads *same*, *≈*, *n/a* or *boundary*;
   none reads *gap*;
2. a feature is written once. No code outside the backend modules and the
   CLI's construction of them branches on the backend kind.

Compose stays the fastest local path and loses nothing. KubeAI is Compose
plus a scheduler, with more information supplied and more tools installed;
it is not a sibling with its own habits.

## Principles

- **Superset, not sibling.** KubeAI gains what Compose has. Where a Compose
  feature has no cluster meaning (GPU indices, `network migrate`), the
  matrix says *n/a* and the CLI says why; it does not grow a second design.
- **The seam is the only place kind matters.** `isinstance(…, ComposeBackend)`
  in the CLI or TUI is a smell. Each phase removes some. The Compose-only
  escape hatches (`stack up`/`down` as raw compose) stay, and say so.
- **Verified on a real cluster.** A phase is done when its step in
  `dev/kubeai_e2e.sh` passes on k3s, not when its fakes pass. This is how
  the backend drifted last time.
- **One reviewable change per phase**, the Compose suite green throughout.

## Phases

### P0. One contract, one liveness view, one diagnosis — done

The 2026-09-24 audit landed: an e2e on k3s (K0); the LiteLLM gateway
fronting both backends, so the alias and key are the same everywhere (K1);
a strict `residency()` built from pods (K2); one startup diagnosis over
containers or pods (K4); one served-name rule, one GPU count, `runtime.env`
on KubeAI (K5); the gateway as its own module (G); and unleased keep-warm
yielding to leased demand on both backends.

### P1. One acquire path

**Closes:** the admission and `config publish` rows.

Today `Controller._admission_mode()` is true only for a backend with
`residency` + `preview` + `converge`, which is Compose. KubeAI takes the
pre-September branch in five places (render, acquire, `observe_state`,
`config publish`, `renew`).

Review 2026-09-25 against the code split this in two. The admission path
assumes GPU accounting in three places (`_backfill_allocations`,
`_unresolved_allocations`, `_admission_view` keeps a LIVE deployment only
with committed GPUs), so a `preview` alone would drop every KubeAI lease from
the render. And about fifteen test fakes drive the legacy path on purpose
(queue, lock, serialised publication), so deleting it is test work, not
controller work.

**P1a. KubeAI on the admission path.**

- One authority for "this backend allocates GPUs": `allocates_gpus`
  (Compose true, KubeAI false). The controller's accounting reads it
  through one helper; on KubeAI every deployment commits an empty
  allocation, so nothing is ever unresolved.
- `KubeaiBackend.preview(desired, placement, approve)`: render the Models
  to memory with a digest, write nothing. Placement inputs are accepted and
  ignored; the plan assigns every renderable deployment no GPUs.
  `plan_on_idle_host` is the same render, so a `--queue` acquire of an
  unrenderable endpoint fails at once instead of waiting out the timeout.
- `converge(..., placement=)` on KubeAI. Without it the controller's
  `TypeError` fallback would render **and apply** in one step.
- The approved-digest guard (`_planned_digest`, pre-approval,
  `last_planned_digest` / `last_preview_digest`) moves from `ComposeBackend`
  to `ConvergeScaffold`: one approval mechanism, not a copy.
- Compose's stable addresses and container adoption stay Compose-only by
  capability (`network`, `adopted`), not by joining KubeAI to them.
- `--queue` on KubeAI: admitted at once; the cluster is the queue.

**P1a done 2026-09-25.** Verified on k3s by `dev/kubeai_e2e.sh`, which
gained the step "an unrenderable endpoint is refused before anything is
written" (a `--queue` acquire, no lease, no Model), and still passes the
make-room step, which now runs on the admission path.

**P1b. Delete the legacy branch.**
`MemoryBackend` / `NullBackend` and the test fakes get a trivial
`residency` and `preview`; then the non-admission branches and
`_admission_mode()` go.

**P1b done 2026-09-26.** `SimpleAdmission` (in `leasing/backend.py`) gives
a backend with only `realize` / `teardown` the admission surface; a fake
that emulates capacity overrides its `plan`, one that emulates a render
failure its `refuse`. The controller has one acquire, render, renew and
publish path. Rollback after a commit is still reachable (the runtime can
change between preview and render) and keeps its tests, on a fake whose
render refuses what its preview admitted.

**Exit:** P1a: `_admission_mode()` is true for both real backends; a render
failure on KubeAI rolls the lease back before any `kubectl`; e2e passes.
P1b: `_admission_mode` is gone; `test_controller` runs its acquire scenarios
on all three fakes.
**Size:** P1a medium, P1b medium (mostly tests).

### P2. Day-2 verbs and the TUI through the seam

**Closes:** `logs` / `ps` / `stack`, the TUI's log follow, docker pane and
Up / Down, and `measure`.

- Two backend methods: `instances()` (from `residency()`: deployment, unit
  name, state, restarts, age) and `stream_logs(target, *, follow, tail)`
  (`docker logs -f` / `kubectl logs -f`).
- `ps` prints `instances()`. `logs` streams. `stack up` is `apply`; `stack
  down` is `backend.down()`, which both backends have. The raw compose form
  moves under `stack compose …` and stays Compose-only by name.
- TUI: the docker pane becomes an instances pane; the log follower uses
  `stream_logs`; Up / Down call `apply` / `down`.
- `measure`: lift the guard (`deployment_logs` already exists on KubeAI)
  after checking vLLM's memory line reaches the pod log.

**Exit:** on k3s the TUI follows a Model's log and lists its pod; `infer-stack
logs <alias>` and `ps` work on both backends with the same output shape.
**Size:** medium, mostly plumbing; no controller change.

**P2 done 2026-09-26.** `leasing/instances.py`: an `Instance` per container
or pod, built from residency by each backend's `instances()`, and one
`LogFollower` for the CLI and the TUI. `stack up` is `apply`, `stack down` is
`backend.down()`, and the raw Compose verbs (`stack compose …`) act on
whichever Compose project the backend has on this host, the gateway's on
KubeAI. Verified on k3s and on compose (the simulator catalog), in a real
terminal. `measure` was never refused on KubeAI (its guard was
`deployment_logs`, which KubeAI has), but it reads GPU memory-profiling lines
that CPU vLLM does not print: it moves to P4 with its GPU handover.

### P3. Gateway feature parity

**Closes:** `routes` on KubeAI, `dynamic_routing`, Open WebUI and the reverse
proxy in front of a cluster.

- The `routes` commands and `secrets rotate` resolve `controller.backend.gateway`
  on either backend instead of checking the backend kind.
- The KubeAI gateway project is rendered with the same `ui`, `reverse_proxy`
  and `dynamic_routing` settings as Compose. Postgres for dynamic routing is
  a gateway-project service, so nothing engine-side changes.

**Exit:** `routes list` on KubeAI; Open WebUI in front of a cluster; an e2e
step that acquires the same model `--dedicated` twice under dynamic routing
and gets two Models and two routes.
**Size:** small to medium; the `Gateway` class already owns the pieces.

**P3 done 2026-09-26.** The KubeAI gateway takes the compose settings (`ui`,
`reverse_proxy`, `dynamic_routing`) and records them in the recovery profile
as the gateway project's own profile. The kubeai backend hands the gateway
explicit render inputs (registry rows for the catalog and every Model, or
per-deployment dynamic routes) instead of writing rows into the registry
before approval, so the gateway's changes are previewed and approved with
the acquire's. Under dynamic routing each deployment is its own Model,
named with the deployment tail compose uses for its services. The `routes`
commands resolve the backend's Compose project and ask the backend for its
rows. Verified by `dev/kubeai_e2e.sh` on k3s: routes, Open WebUI, and two
`--dedicated` Models behind one alias.

### P4. Placement information parity (optional)

**Closes:** `min_vram_gib` / `measure`; less hand-written cluster
information.

- `catalog suggest` on KubeAI reads the device plugin's node labels
  (`nvidia.com/gpu.product`, `.memory`) and proposes `resourceProfiles`
  (today the README does this by hand).
- With several profile sizes, `min_vram_gib` picks the smallest profile
  whose GPU memory fits when an endpoint names none; otherwise the
  warn-and-ignore stays.

**Exit:** a catalog with `min_vram_gib` and no `resource_profile` lands on
the right profile on a cluster with two sizes.
**Size:** small. Shrinks "the information you add"; skip if nobody runs a
mixed-GPU cluster.

**P4 done 2026-09-26** on two k3s nodes (the second a container,
`dev/k3s_agent_container.sh`) with fake GPU labels: `min_vram_gib 40` got
the 80 GiB profile and scheduled on that node, `10` got the 24 GiB one and
answered, and `catalog suggest` proposed a profile per product
(`E2E_SIZED=1 dev/kubeai_e2e.sh`). A profile's size is its selected nodes'
`nvidia.com/gpu.memory`; the chart's generic profiles have no selector, so
no size. KubeAI keeps its own measurements overlay and fills a missing
`min_vram_gib` from it, as compose does, so `measure --record` feeds the
choice. Real GPU labels, a GPU serving the chosen profile, and `measure`
reading vLLM's memory lines are `dev/handover/p4_gpu_labels.sh`.

### P5. The multi-workstation shape

**Closes:** the gateway on one host, the port-forward default.

- Render the gateway in the cluster (a Deployment + Service behind an
  ingress) as the second gateway target; the Compose-project gateway stays
  the single-host default. `secrets rotate` becomes a Secret update and a
  rollout; `doctor` checks the ingress; `kubeai_base_url` defaults to it.
- A second workstation joins with `scripts/join_agent.sh`; resource
  profiles with node selectors say which GPUs live where. Write the "add a
  workstation" runbook into `kubeai-backend.md`.

**Exit:** on two nodes, a card on node A leases a model that lands on node
B and talks to it through the in-cluster gateway; `secrets rotate` works.
**Size:** medium to large. The only phase that adds a second renderer for
the gateway, and the only one that needs a second machine. Last.

### P6. One test surface (ongoing)

- Parametrize the controller tests over the Memory, fake-Compose and
  fake-KubeAI backends; `tests/test_parity.py` runs each *same* row's
  scenario on both fakes.
- `dev/kubeai_e2e.sh` gains one step per phase.
- Finishing a phase updates the matrix in `backend-parity.md`; the plan is
  done when the matrix has no *gap* row.

**P6 done 2026-09-26** as a suite that stays open: `tests/test_parity.py`
builds the same stack over a fake Docker and a fake kubectl and runs one
test per *same* row on both (21 rows, the TUI included). A row turned *same*
gets its test in the same change. The Memory and Null backends run the
controller's one path through `SimpleAdmission` (P1b), so the controller
suites cover them without a third parametrization. Found on the way: the
TUI replaced an injected Docker runner with the real one, so a TUI started
on a test backend talked to the host's Docker; it now wraps only the
default runner.

## Duplicate authorities

The rule while executing: a duplicate authority found on the way is
refactored when it is small, recorded here when it is not, and never made
worse. A blocker is fixed whatever its size.

| authority | where it was | status |
|---|---|---|
| "this backend allocates GPUs" | inferred from which methods a backend has | **fixed** (P1a): `allocates_gpus()` in `leasing/backend.py`, read by the controller and the CLI |
| approval digest and pre-approval | a copy inside `ComposeBackend` | **fixed** (P1a): `ConvergeScaffold`, shared |
| KubeAI render vs its plan | `converge` rendered inline | **fixed** (P1a): one `_render_documents` behind `converge`, `preview` and `plan_on_idle_host` |
| the acquire path | admission and a legacy branch in five places | **fixed** (P1b) |
| `_render`'s one-shot `converge(desired)` fallback, which applies | a second render contract for old backends | **fixed** (P1b): removed |
| "can anything be admitted while residency is unknown" | `_admit` admitted requests needing no new GPU; the render after the commit then failed without residency | **fixed** (P1b): nothing is admitted, and nothing is committed |
| the desired set | `desired_deployments()` beside the admission view; `routes prune` used the former | **fixed** (P1b): one view, `_admission_view` |
| a crashed acquire's placement scope | recorded in the marker and re-applied by a recovery render, although admission never records one | **fixed** (P1b): a stale scope is dropped |
| "is it running" for `status` | `docker compose ps` service names beside residency (so KubeAI read `unverified`) | **fixed** (P2): residency |
| engine vs gateway in the TUI | a `litellm` name hint (Open WebUI and Postgres counted as engines) | **fixed** (P2): an instance serves a deployment or it does not |
| where the day-2 verbs find the Compose project | a hard-coded path and project name | **fixed** (P2): the backend's `compose_project()` and `compose_argv()` |
| how the TUI runs a runtime command | it replaced the backend's runner with Docker's, whose allowlisted environment has no `KUBECONFIG`: every kubectl call from the TUI failed | **fixed** (P2): only `docker` commands are wrapped |
| the KubeAI gateway's approval | the gateway project asked its own diff approval at render, after the lease committed | **fixed** (P3): previewed and approved with the acquire's |
| KubeAI's route rows | written into the registry before any approval, and only for live Models (so every new Model recreated the gateway) | **fixed** (P3): render inputs, persisted after approval; the catalog's rows too, so no blip |
| a route's upstream | derived twice, in the render and in `routes list` (which showed `?` for a cluster route) | **fixed** (P3): `registry_route_entry` |
| a deployment's Model name | derived in five places | **fixed** (P3): `model_name()` |
| "is this the compose backend" in the CLI | `isinstance(…, ComposeBackend)` for `routes`, `gc --orphans`, `clean`, `network migrate`, `secrets rotate` | **fixed** (P3): the capability each needs (`compose_project()`, residency labels, `network`, `litellm`) |

## Not in scope

- more than one cluster, or scheduling across a Compose host and a cluster;
- KubeAI autoscaling beyond `min_replicas` / `max_replicas`: the lease
  decides residency;
- custom container launches and ollama daemons on KubeAI (a boundary in
  [known-limitations.md](known-limitations.md));
- Slurm, which is its own track ([../slurm-compatibility.md](../slurm-compatibility.md));
- moving the Compose backend onto Kubernetes primitives.

## Order and size

| phase | size | needs | gives |
|---|---|---|---|
| P1a | medium | a green e2e | atomic acquire on a cluster |
| P1b | medium | P1a; with P6 | one controller path |
| P2 | medium | P1 (instances from residency) | the TUI and day-2 verbs on a cluster |
| P3 | small–medium | none | dynamic routing, Open WebUI on a cluster |
| P4 | small | a mixed-GPU cluster to verify | less hand-written cluster information |
| P5 | medium–large | a second machine | the gateway off the operator's host |

P1 → P2 → P3 is single-host-cluster parity: k3s on one workstation, with
everything a Compose user has. P5 is the multi-workstation step.

# Backend parity roadmap: KubeAI as a superset of Compose

**Status:** proposed 2026-09-25 · **P0 done** 2026-09-24 on
`dev/backend-unification` · P1–P5 not started · P6 is ongoing.
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

- `KubeaiBackend.preview(desired, placement, approve)`: render the Models
  to memory with a digest, write nothing. Placement inputs are accepted and
  ignored; `unplaced` is exactly the unrenderable set (no profile, engine,
  custom launch).
- Admission accepts a backend with no GPU accounting: Compose reports
  assignments, KubeAI reports none, and the ledger's allocation table stays
  empty for it. A Pending pod is a wait reason, never an unplaced error.
- `--queue` on KubeAI: admitted at once; the cluster is the queue. Say so
  in the option's help.
- `MemoryBackend` / `NullBackend` get a trivial `residency` and `preview`
  so the tests run the one path.
- Delete the non-admission branches and `_admission_mode()`.

**Exit:** `_admission_mode` is gone; a render failure on KubeAI rolls the
lease back before any `kubectl`; e2e passes; `test_controller` runs its
acquire scenarios on all three fakes.
**Size:** medium. The controller is the most-changed module; start from a
green e2e and keep the diff to the five sites.

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
| P1 | medium | a green e2e | one controller; atomic acquire on a cluster |
| P2 | medium | P1 (instances from residency) | the TUI and day-2 verbs on a cluster |
| P3 | small–medium | none | dynamic routing, Open WebUI on a cluster |
| P4 | small | a mixed-GPU cluster to verify | less hand-written cluster information |
| P5 | medium–large | a second machine | the gateway off the operator's host |

P1 → P2 → P3 is single-host-cluster parity: k3s on one workstation, with
everything a Compose user has. P5 is the multi-workstation step.

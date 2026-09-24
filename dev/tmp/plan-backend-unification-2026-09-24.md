# Plan: one set of authorities for the Compose and KubeAI backends

Status: draft for review, 2026-09-24. Nothing here is implemented.

## Goal

Scaling past one host needs the KubeAI backend. It has not been run against a
cluster since it landed (2026-07-02); everything since then went into the
Compose backend. Make both backends go through the same controller paths and
the same client contract, so the next feature is written once, and remove
places where two pieces of code each decide the same thing.

Operating assumption: KubeAI runs **stock vLLM** Models. Custom container
launches (`runtime.command` / `mounts`) stay Compose-only.

## What the audit found

### A. Two client contracts (the scale-up blocker)

| | Compose | KubeAI |
|---|---|---|
| front door | LiteLLM gateway, one `base_url` | KubeAI's own gateway |
| request name | the **endpoint alias** | a DNS slug of the served name (`Qwen/Qwen3.8-27B` → `qwen-qwen3-8-27b`) |
| auth | managed master key | none (`EMPTY`) |

Cards and the pipeline send the endpoint alias as `model=`. Only one magnet
example reads the env file's per-endpoint names (`INFER_STACK_ENDPOINT_*`),
so an unchanged card gets a 404 on KubeAI. Routing
features built on the gateway also stop at Compose: `secrets rotate`, stable
upstream addresses, and the proposed external endpoints.

### B. Two acquire paths in the controller

The controller branches on `_admission_mode()` (it is on only when the
backend implements `residency` + `preview`) in five places: render, the
acquire itself, `observe_state`, `config publish` and `renew`. The admission
path carries the keep-warm fix, atomic acquire and serialised publication
review; the other path is the pre-September one. In production only KubeAI
takes it, and nothing has run it at scale.

### C. Two liveness authorities

Compose answers "what is running" with `residency()`: strict, and it raises
`ResidencyUnknown` rather than guess. KubeAI answers with `observe()`, which
returns the **empty set** when `kubectl` fails. Compose had that exact bug:
an unreadable Docker looked like "nothing running" and led to wrong decisions.

### D. Failure diagnosis only exists on Compose

These read Docker container state and exist nowhere else:

- crash-loop fail-fast (`startup_failure`, restart policy, the log classifier)
- image-pull progress
- engine logs in the TUI

KubeAI has the same failure modes (`CrashLoopBackOff`, `ImagePullBackOff`,
pod OOMKilled) and waits the full timeout on all of them.

### E. Small duplicated helpers

- served name: `kubeai._served_name` duplicates the rule inside
  `compose.vllm_service_dict`
- GPU count: `kubeai._gpu_count` duplicates `placement.required_gpu_count`
- launch-field support: KubeAI refuses `runtime.env`, although it already
  writes a Model's `spec.env` for the attention backend

Already shared, and fine: the vLLM argument pipeline (`vllm_service_dict` +
`vllm_args`), catalog resolution and identity, published profiles,
`ConvergeScaffold` (atomic writes, lock, diff approval).

## Plan

Each step is one reviewable change that leaves both backends working.

### K0. A real KubeAI to test against (prerequisite)

A `kind` cluster on the guest with the KubeAI chart and a `resourceProfile`
backed by the **vLLM simulator** (the image the mock catalog already uses).
No GPU needed. Add `dev/e2e_tests/kubeai_kind.sh`: acquire, `run` a
request, release, and verify the Model is pruned. Without it every step below
is fake-verified only, which is how KubeAI drifted.

**Done 2026-09-24**, with k3s instead of kind (k3s runs as a systemd service
on the guest; `--disable traefik --disable servicelb` keeps ports 80/443
free). KubeAI 0.23.4's `cpu` profile runs real vLLM 0.11.2 on CPU (needs
AVX-512), which is better than the simulator: KubeAI builds vLLM's own
command line, which the simulator's CLI would reject. Setup:
`dev/e2e_tests/kubeai-cpu-values.yaml`; check:
`dev/e2e_tests/kubeai_k3s.sh`, which passed. `doctor` passed its four checks;
acquire → ready took 143 s with an image pull, the full `run` took 86 s with
the image cached, a real completion came back, and release pruned the Model.
A Docker container on the node reaches KubeAI's gateway at its cluster IP,
which K1 relies on.

### K1. One client contract: the LiteLLM gateway fronts both backends

The gateway is infer-stack's and routes aliases to upstreams. Where the
engine runs is the backend's business. On KubeAI, each alias routes to the
KubeAI gateway URL with the Model's CR name as the upstream model. Then:

- the request name is the endpoint alias on every backend (A fixed);
- `secrets rotate`, dynamic routing and external endpoints work unchanged;
- KubeAI's `access()` returns the same front door as Compose.

**Decision needed: where the gateway runs for KubeAI.**

1. As today: a one-service Compose project on the operator host, pointing at
   the cluster's gateway. Least new code, but one host is in every request's
   path.
2. In the cluster, as a Deployment rendered next to the Models. Right for
   scale, but it is a second renderer for the gateway. Kubernetes has no
   per-service fingerprint recreate, so config changes need a rollout.

Recommendation: (1) first, since it reuses everything and unblocks cards
now; (2) when a real multi-node run needs it. Record (1)'s single host as a
known limitation.

**Done 2026-09-24 as (1).** The kubeai backend owns a gateway-only
`ComposeBackend` (its own state dir and project, `infer-stack-gateway`) and
feeds it one generic `upstream` route row per alias (`api_base` + the Model
name). The same row type is what external endpoints need. Verified on k3s
with `dev/kubeai_e2e.sh`: without the gateway, the alias returned 404; with
it, the same request answered, and `secrets rotate` accepted the new key and
rejected the old one. Static routes only; dynamic routing stays
compose-only.

### K2. One liveness authority

Generalise `residency()` into a backend-neutral snapshot: per deployment,
its instances (a container or a pod), state, restart count, last exit
reason, waiting reason, and GPUs when the backend knows them. Build KubeAI's
from `kubectl get pods -l model=<name> -o json`, raising `ResidencyUnknown`
when kubectl fails. `observe()` becomes a thin view of it on both backends.

### K3. One acquire path

Give placement to the backend:

- **Compose** keeps the local planner and GPU allocations.
- **KubeAI** always admits and records no GPUs, because the cluster
  schedules. A Model stuck Pending shows up as a waiting reason in K2's
  snapshot, not as an unplaced error.

With residency (K2) and a `preview` that renders without applying on both
backends, `_admission_mode()` is always true for real backends. The
non-admission branches are then deleted. Memory and Null backends get a
trivial residency so tests use the same path.

### K4. One failure diagnosis

Move `startup_failure` and the log classifier behind the K2 snapshot. Compose
supplies container state and logs; KubeAI supplies pod state
(`CrashLoopBackOff`, `ImagePullBackOff`, `OOMKilled`) and `kubectl logs`.
The fail-fast wait, the "likely cause" text and the TUI error path then work
on both. Pull progress stays Compose-only; Kubernetes pulls by itself, and
`ImagePullBackOff` is reported as a failure.

### K5. Small cleanups

- one served-name function
- one GPU-count function
- KubeAI accepts `runtime.env` (to `spec.env`, same reserved names)
- TUI engine logs through a backend `deployment_logs` stream, with
  `kubectl logs -f` on KubeAI

## Order and size

K0, then K1, is the minimum for a card to run unchanged on a cluster. K2 and
K3 are the refactor proper, where the duplicate authorities go away. K4 and
K5 are then small because they sit on K2.

Each step keeps the Compose suite green, and every KubeAI step adds a line
to the kind script.

## Not in scope

- multi-cluster, or scheduling across a mix of Compose hosts and a cluster
- KubeAI autoscaling policy beyond today's `min_replicas` / `max_replicas`
- custom container launches on KubeAI

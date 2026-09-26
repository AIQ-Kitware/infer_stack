# Compose and KubeAI: parity and deviations

infer-stack has one control plane and two realization layers. Everything
above the backend seam is shared: the catalog, the ledger (leases,
deployments, TTLs), the controller (`acquire` / `release` / `wait` / `evict`
/ `gc` / `clean` / `renew`, the admission queue, keep-warm reclaim), the
LiteLLM front door, the env-file contract, and the TUI. A backend only makes
the desired set real: Compose as containers on the local Docker daemon,
KubeAI as `Model` custom resources on a cluster.

The intended relationship is **KubeAI = Compose + a scheduler**. A catalog
that runs on Compose runs on KubeAI once you say what a GPU unit means on
the cluster and install the cluster tooling. A card does not change: the same
`OPENAI_BASE_URL`, the same managed key, the endpoint alias as the model
name.

This page records where that holds today (2026-09-25), where it does not,
and which gaps are deliberate. The plan for the rest is
[planning/backend-parity-roadmap.md](planning/backend-parity-roadmap.md);
cluster setup is [kubeai-backend.md](kubeai-backend.md).

## What you add to move from Compose to KubeAI

| | Compose | KubeAI adds |
|---|---|---|
| tools | Docker with the NVIDIA runtime | a cluster (`scripts/bootstrap_k3s.sh` for one host, `scripts/join_agent.sh` for another), `kubectl`, `helm`, the KubeAI chart (`scripts/install_kubeai.sh`), the NVIDIA device plugin on GPU nodes |
| information | none: GPUs are discovered with `nvidia-smi` | `resourceProfiles` in the chart values (what one GPU unit requests, and on which nodes), and per endpoint `runtime.resource_profile` or a default `kubeai_resource_profile` |
| settings | `backend compose` | `backend kubeai`; `kubeai_namespace`, `kubeai_base_url`, `kubeai_gateway_upstream` when the defaults (namespace `kubeai`, a port-forward on 8000, the Service's cluster IP) do not hold |
| preflight | none needed | `infer-stack doctor`: cluster → CRD → namespace → gateway |

The catalog, `acquire … --env-file`, `release`, the TUI and the env file a
card sources are the same on both.

## Layers

```
catalog ──► ledger ──► controller ──► gateway (LiteLLM: aliases, key, routes)
                          │
                          ▼  backend seam: leasing/backend.py
               ┌──────────┴───────────┐
          ComposeBackend          KubeaiBackend
          docker compose          kubectl apply of Model CRs
          local GPU planner       the cluster scheduler
          leasing/compose/        leasing/kubeai/ + leasing/kubeai-gateway/
```

On KubeAI the gateway is a Compose project with no engines
(`infer-stack-gateway`) on the host running infer-stack. It routes each
alias to the cluster's KubeAI Service under the Model's name, so `secrets
rotate` and the static superset route table work unchanged.

## Parity matrix

**same**: one code path, or verified equivalent. **≈**: the same outcome by
a different mechanism. **gap**: missing on one side and on the roadmap.
**n/a**: does not apply there. **boundary**: deliberately unsupported
(see [planning/known-limitations.md](planning/known-limitations.md)).

### Client contract

| | Compose | KubeAI |
|---|---|---|
| one `OPENAI_BASE_URL`, the managed key, the alias as model name | same | same through the gateway; with `litellm false`, Model names and no key |
| `secrets rotate` | same | same |
| env file (`INFER_STACK_*`, `OPENAI_*`) | same | same |
| readiness is a real generation through the front door | same | same |

### Lifecycle

| | Compose | KubeAI |
|---|---|---|
| `acquire` / `release` / `wait` / `evict` / `gc` / `clean` / `renew` / `run` / `test` | same | same |
| admission: lease and GPUs committed atomically; a `--queue`d acquire holds nothing | yes | **gap** (roadmap P1): the lease is committed first, then rendered; a Model the cluster cannot place waits out `--timeout`, then rolls back |
| `config publish` | pure preview, then commit | render, then commit (same gap) |
| `--no-apply` / `apply` / `render` | same | same |
| unleased keep-warm yields to leased demand | at placement | ≈ during the wait: `needs_room` evicts the longest-idle, one per 30 s |
| crash-loop fail-fast, the engine's error quoted | same | same (pods, `kubectl logs --previous`) |
| image-pull progress | yes | n/a: the kubelet pulls; `ImagePullBackOff` is a reported wait reason |
| strict `residency()` for decisions, lenient `observe()` for reports | same | same |
| recovery after an interrupted apply (settle check) | yes | ≈ `kubectl apply` is idempotent; nothing is left half-created |

### Placement

| | Compose | KubeAI |
|---|---|---|
| where a deployment lands | local planner over `nvidia-smi` | the cluster scheduler, via `resource_profile:<gpus>` |
| GPU count from TP × PP × DP | same | same |
| `placement.gpu_indices`, `allowed_gpus`, `skip_display_gpus` | yes | n/a: no host indices; node-scoped resource profiles are the equivalent |
| `placement.min_vram_gib`, `infer-stack measure` | yes | warned and ignored; `measure` refused (**gap**, P4) |
| GPU allocations recorded in the ledger | yes | none: the cluster owns them |
| more than one host | boundary | yes: the reason the backend exists |

### Catalog features

| | Compose | KubeAI |
|---|---|---|
| vLLM endpoints, `runtime.*` flags, `extra_args`, `env` | same | same (one argument pipeline) |
| served-name rules | same | same (`leasing/naming.py`) |
| `runtime.command`, `runtime.mounts` (custom launchers) | yes | boundary: a stock vLLM Model has no place for them; the render refuses loudly |
| ollama endpoints (`runtime_hosts`) | yes | boundary: daemon-shaped, not model-shaped |
| `min_replicas` / `max_replicas` | n/a | pass-through, default 1/1 |
| weight cache | the host HF cache, mounted | whatever the chart provides; cache profiles are not rendered |

### Gateway

| | Compose | KubeAI |
|---|---|---|
| static superset routes, no blip on model churn | same | same |
| `routes` inspect / seed / prune | yes | **gap** (P3): refused, although the registry exists |
| `dynamic_routing` (admin API + Postgres; distinct upstreams for same-model `--dedicated`) | yes | **gap** (P3) |
| Open WebUI (`ui`), reverse proxy | yes | **gap** (P3): the gateway project is rendered with `ui` off |
| `network migrate` / `network check` | yes | n/a: Service addresses are stable |
| where the gateway runs | this host | this host, so it is in every request's path (P5 moves it into the cluster) |

### Day-2 operations and the TUI

| | Compose | KubeAI |
|---|---|---|
| `doctor` | nothing to check | four checks |
| `logs`, `ps`, `stack up` / `stack down` | docker compose wrappers | **gap** (P2): they say so and point at `kubectl` |
| TUI: leases, deployments, catalog editing, acquire / release / evict, API tab, settings | same | same |
| TUI: engine log follow, the docker pane, the Up / Down buttons | `docker logs`, `docker compose ps` | **gap** (P2): empty, or "nothing rendered yet" |
| TUI GPU pane | `nvidia-smi` on this host | this host, not the cluster |
| `gc --orphans` (also inside `clean`) | yes | n/a: unlabeled Models are never touched |

### Testing

| | Compose | KubeAI |
|---|---|---|
| unit suite with fake runtimes | yes | yes |
| end to end | `dev/e2e_tests/run.sh` (tiers; `--gpu` for serving) | `dev/kubeai_e2e.sh` against a real cluster; k3s with CPU vLLM needs no GPU |

## Deviations that stay

- **Custom container launches and ollama daemons** do not run on KubeAI. A
  KubeAI Model is stock vLLM. Use `--backend compose` for those endpoints.
- **GPU indices** have no cluster meaning. Kubernetes places by resource, not
  by index; the per-node equivalent is a resource profile with a node
  selector.
- **One control plane never spans a Compose host and a cluster.** Pick a
  backend per data root.

## Where the gaps close

[planning/backend-parity-roadmap.md](planning/backend-parity-roadmap.md)
has the phases and their exit criteria. The audit that produced this page,
with what has already landed, is
`dev/tmp/plan-backend-unification-2026-09-24.md`.

# The KubeAI backend

`--backend kubeai` drives the same leasing verbs (`acquire` / `release` /
`wait` / `evict` / `gc` / the TUI) against a Kubernetes cluster running
[KubeAI](https://www.kubeai.org), instead of single-host docker compose. The
ledger, catalog, TTLs, admission queue, and env-file contract are identical —
only the realization layer changes:

| | compose backend | kubeai backend |
|---|---|---|
| unit of serving | docker compose service | KubeAI `Model` CR |
| GPU placement | planned locally (`plan_placement`) | cluster-scheduled via `resourceProfile` |
| front door | LiteLLM gateway | the same LiteLLM gateway, on this host, routing to KubeAI |
| request name | endpoint alias | endpoint alias |
| auth | managed `LITELLM_MASTER_KEY` | the same |
| state dir | `<data>/leasing/compose/` | `<data>/leasing/kubeai/` (`models.yaml` + sidecar), `<data>/leasing/kubeai-gateway/` |

Clients see the same contract on both backends: one `OPENAI_BASE_URL`, the
managed key, and the endpoint alias as the model name, so a card runs
unchanged. The gateway is a one-service Compose project
(`infer-stack-gateway`) that routes each alias to KubeAI under the Model's
name. `secrets rotate` works as on compose. With `--no-litellm` (or `config
set litellm false`) there is no gateway: clients talk to KubeAI directly and
must use the Model name from the env file (`INFER_STACK_ENDPOINT_*`); the
alias gets HTTP 404.

A deployment in the desired set renders as one `Model` CR labeled
`infer-stack/managed=true` + `infer-stack/deployment=<id>`; `apply` is
`kubectl apply` plus pruning of managed Models the render dropped. Reclaim
semantics carry over: an idle `keep-warm` deployment keeps its CR (model stays
resident); `stop` deployments are pruned on release; `evict`/`gc` free the
cluster. Hand-applied Models without the managed label are never touched.

Where the two backends match and where they still differ, row by row:
[backend-parity.md](backend-parity.md); the plan for the rest is
[planning/backend-parity-roadmap.md](planning/backend-parity-roadmap.md).

## One-time cluster setup

The normal setup path is now capability-driven and lives in the CLI:

```bash
# Existing cluster: just point kubectl at it. For a new local K3s server:
infer-stack kube k3s bootstrap --version=<exact-k3s-version>

# Planning is read-only; apply only after reviewing it.
infer-stack kube setup
infer-stack kube setup --apply
infer-stack kube nodes
```

`kube setup` checks the Kubernetes API, Ready nodes, GPU scheduling/discovery
facts, Helm, the KubeAI CRD/service, and discovered GPU resource profiles. It
is capability-based: an existing GPU Operator or externally-managed KubeAI is
accepted when it already supplies what infer-stack needs. It does not replace
working external components merely because they were installed differently.

The managed NVIDIA path begins only after Kubernetes exposes an `nvidia`
RuntimeClass. Installing NVIDIA drivers/container-runtime packages is host
administration and remains outside infer-stack. With that runtime present,
`kube setup --apply` may install/reconcile the pinned NVIDIA device plugin +
GPU Feature Discovery and then derive profiles from `nvidia.com/gpu.product`
and `nvidia.com/gpu.memory`.

KubeAI values are written to `<data>/generated/kube/kubeai-values.yaml`. Named
profiles already supplied by the operator win over generated profiles. Use an
operator values file for other chart customization:

```bash
infer-stack kube setup --values ./kubeai-values.local.yaml
infer-stack kube setup --apply --values ./kubeai-values.local.yaml
```

For an existing Helm-managed KubeAI release, apply preserves its installed
chart version by default; use `--kubeai-version=...` for an explicit change.
`scripts/install_kubeai.sh` remains only as a manual Helm escape hatch.

## Point infer-stack at it

```bash
infer-stack config set backend kubeai
# optional overrides (defaults shown):
infer-stack config set kubeai_namespace kubeai
infer-stack config set kubeai_base_url http://127.0.0.1:8000/openai/v1
# fallback profile for endpoints whose runtime omits resource_profile:
infer-stack config set kubeai_resource_profile nvidia-gpu-rtx-4090
# how the gateway reaches KubeAI (default: the kubeai Service's cluster IP,
# reachable from a cluster node; set an ingress URL when this host is not one):
infer-stack config set kubeai_gateway_upstream http://kubeai.example/openai/v1
```

Catalog endpoints opt into a specific profile per endpoint; the GPU count is
appended automatically from `tensor_parallel_size × pipeline_parallel_size ×
data_parallel_size`:

```yaml
endpoints:
  qwen-coder:
    model: qwen-coder-32b
    engine: vllm
    runtime:
      tensor_parallel_size: 2
      max_model_len: 32768
      resource_profile: nvidia-gpu-rtx-4090   # -> nvidia-gpu-rtx-4090:2
```

### Sizing: `min_vram_gib` picks a profile

An endpoint that names no `resource_profile` but declares
`placement.min_vram_gib` (or has one recorded by `infer-stack measure
--record`) gets the smallest resource profile whose GPUs are that large. A
profile's GPU size is the `nvidia.com/gpu.memory` label (set by GPU Feature
Discovery) of the nodes its `nodeSelector` selects; a profile without a
`nodeSelector`, like the chart's generic ones, has no size and is never
picked this way. With none large enough, the `kubeai_resource_profile`
default is used, or the acquire is refused with the sizes it found.
`infer-stack kube setup` is the setup authority for these profiles;
`catalog suggest --backend kubeai` consumes the same profile-generation shape
while suggesting model/catalog entries.

Verify the setup before the first acquire — `doctor` checks the chain in
dependency order (cluster reachable → CRD installed → namespace → KubeAI's API):

```bash
infer-stack doctor
```

Then the normal verbs just work:

```bash
infer-stack acquire qwen-coder --ttl 2h --env-file lease.env --yes
source lease.env            # OPENAI_BASE_URL + OPENAI_API_KEY -> the gateway
# request model=qwen-coder, the endpoint alias, as on the compose backend
infer-stack release --env-file lease.env
```

## The gateway inside the cluster

By default the gateway is a Compose project on the host running infer-stack,
so that host is in every request's path. For more than one workstation, put
it in the cluster:

```bash
infer-stack stack down                       # the host gateway (and any Models)
infer-stack config set kubeai_gateway cluster
infer-stack acquire <endpoint> --env-file lease.env --yes
# OPENAI_BASE_URL is now http://<a node's address>:30442/v1: any node answers
```

It is the same gateway (image, config, managed key, route registry) as a
Deployment and a NodePort Service in the KubeAI namespace, and it reaches
KubeAI by the Service's cluster DNS name, so no `kubectl port-forward` is
needed. `secrets rotate` updates its Secret and rolls it; `doctor` checks it;
`stack down` removes it. Settings: `kubeai_gateway_node_port` (30442) and
`kubeai_gateway_url` (an ingress URL clients should use instead). Static
routes only: dynamic routing and Open WebUI need the host placement.

## Add a workstation

On the server node, copy the K3s token into a protected file on the new host
through whatever secure channel you normally use, and record the server's K3s
version. Then on the new workstation (after NVIDIA driver/container-runtime
setup when it is a GPU node):

```bash
infer-stack kube k3s join \
  --server=https://<first-node-ip>:6443 \
  --token-file=~/.private/k3s-token \
  --node-name=<name> \
  --version=<same-k3s-version>
```

The token is read from a file so it does not land in shell history or the
installer argv. Between K3s nodes, allow the K3s-required cluster traffic
(6443/tcp to the server, the configured Flannel/backend traffic, and kubelet
traffic as appropriate for your network); expose the infer-stack NodePort only
on the trusted LAN/VPN.

Back on the control node, inspect and reconcile the newly visible hardware:

```bash
infer-stack kube nodes
infer-stack kube setup
infer-stack kube setup --apply
```

A new GPU product becomes a new discovered KubeAI resource profile without
rewriting any same-named operator profile. `dev/k3s_agent_container.sh` still
provides a synthetic second K3s node for development;
`dev/handover/p5_two_hosts.sh` is the real two-host evidence path.

## Semantics + limitations

- **Readiness is a real generation** through the gateway (same philosophy as
  compose): a Model CR existing — even with ready replicas — is not proof it
  can serve.
- **Admission is the same as on compose, minus GPU accounting.** An acquire
  is previewed in memory and commits its lease only if every deployment
  renders; a refused one (no resource profile, an ollama endpoint, a
  served-name collision) writes no lease and runs no `kubectl`. The cluster
  schedules, so capacity is never an admission reason and `--queue` is
  admitted at once. A Model the cluster cannot place sits not-ready until
  the acquire's `--timeout`, which rolls the lease back; the wait says why
  (`pod: Unschedulable`, `pod: ImagePullBackOff`).
- **An idle keep-warm Model stays only while its pod has started**, as an
  idle keep-warm container does on compose (a crash-looping pod counts, like
  a restarting container). One whose pod is gone, still Pending or finished
  is pruned at the next render.
- **A keep-warm model without a lease gives way to one with a lease.** When
  a leased Model's pod is `Unschedulable`, the wait evicts the longest-idle
  keep-warm deployment, one per 30 s, until it fits. Compose applies the same
  rule when placing. A leased model is never evicted this way. Verified on
  k3s: `E2E_MAKE_ROOM=1 dev/kubeai_e2e.sh` with the `cpu-half` profile.
- **An engine that cannot start fails the acquire at once**, as on compose:
  the crash diagnosis reads the pods (`kubectl get pods`, restart count, last
  exit) and the previous run's log, and quotes the engine's error with a
  likely cause. Verified on k3s: a rejected vLLM flag failed in 52 s instead
  of the 800 s timeout.
  Loud render failures do exist for: a missing `resource_profile` (no invalid
  CR is ever written), served-name collisions between simultaneously live
  deployments, and **ollama endpoints** — KubeAI serves models, not daemons,
  so the catalog's host-centric ollama endpoints don't map onto it yet; use
  `--backend compose` for those.
- `min_replicas` / `max_replicas` runtime keys pass through to the CR
  (default 1/1 — the lease lifecycle, not the autoscaler, decides residency).
- `dev/kubeai_e2e.sh` runs the full lifecycle (doctor → acquire → a request
  by alias, as a card makes it → release → prune verified) against a real
  cluster with an isolated config/data root; use it as the first smoke test
  on new setups. Without a GPU, install the chart with
  `dev/e2e_tests/kubeai-cpu-values.yaml` (real vLLM on CPU; needs AVX-512)
  and run it with `E2E_RESOURCE_PROFILE=cpu`. Verified on k3s 2026-09-24.
- `infer-stack ps`, `infer-stack logs [-f] <endpoint>`, `status` and the TUI's
  runtime pane read the pods (and the gateway's containers), in the same shape
  as on compose. `stack compose …` and `stack restart` act on the gateway's
  Compose project; the engines are pods, restarted by the kubelet.
- By default the gateway runs on the host running infer-stack, so that host is
  in every request's path (`kubeai_gateway cluster` moves it; see above). It
  takes the same settings as on compose: `ui` (Open WebUI,
  on by default), `reverse_proxy`, and `dynamic_routing`, under which each
  deployment is its own Model (`<name>-<id tail>`), so `--dedicated` twice
  gives two Models behind one alias. The gateway's config changes are shown
  and approved with the acquire's, before its lease commits.

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

For the control-plane/worker mental model and the full create/join runbook, see
[cluster-setup.md](cluster-setup.md). The generic integration is
distribution-neutral; K3s is only the first scoped provisioning target.

The normal setup path uses package commands:

```bash
infer-stack kube inventory             # works before KubeAI exists
infer-stack kube doctor                # detailed dependency-ordered readiness
infer-stack kube bootstrap --provider=k3s  # plan a new host or prerequisite repair
sudo -v
infer-stack kube bootstrap --apply     # K3s/Helm/NVIDIA device plugin 0.17.1 + GFD
infer-stack kube install               # inspect inferred resourceProfiles and values
infer-stack kube install --apply       # Helm upgrade/install, wait, post-install doctor
infer-stack kube status                # nodes, GPUs, plugin, pods/services/managed Models
```

Install host NVIDIA drivers/container toolkit before bootstrap. Inventory and
install are distribution-neutral. Bootstrap preserves working clusters and
existing kubeconfigs; it refuses to replace a selected unreachable context.
Use `--version` to pin K3s; active clusters are never implicitly upgraded.

`inventory`, `doctor`, and `status` accept `--json`. Errors belong to individual
probes, so a missing KubeAI ConfigMap cannot erase GPU node facts. A cluster with
a leftover CRD but no namespace/chart can be diagnosed and repaired with these
commands. Kubernetes GPU allocation and GFD readiness are separate from host
GPU visibility.

Mutation requires `--apply` (alias `--yes`). Both mutating commands support
`--dry-run` (alias `--plan`). Install derives one profile per GPU product with
GPU requests/limits, a product selector and `runtimeClassName=nvidia` when that
RuntimeClass exists. Kubernetes chooses placement; infer-stack assigns no host
GPU indices. Models request multiple GPUs via the existing profile/count syntax.

```bash
infer-stack kube install --values ./kubeai-values.local.yaml --namespace kubeai
infer-stack kube install --values ./kubeai-values.local.yaml --namespace kubeai --apply
```

Custom values extend the automatic common case. Existing/custom named profiles
win over generated profiles. The installed chart version is preserved unless
`--version` overrides it; `--chart` and `--release` select alternative settings.
`HF_TOKEN` is preserved using a temporary mode-0600 Helm values file, never
printed or written to the persistent generated public values. Inspect those
values at `<data>/generated/kube/kubeai-values.yaml`.

Installation waits for Helm readiness, then checks the configured
`kubeai_base_url`. If that URL is unavailable, installation reports a routing
failure even though the release is installed; set a reachable OpenAI URL with
`infer-stack config set kubeai_base_url <URL>` and repeat doctor. No automatic
background port-forward is created.

For temporary Compose/KubeAI testing, preserve the Compose authority as described
in [cluster setup](cluster-setup.md#temporarily-swap-compose-configured-workstations-into-a-cluster).
`kube setup` remains a compatibility plan/apply workflow, and the setup scripts
forward to the new commands.

For explicit CPU development, preserve the existing CPU chart workflow:

```bash
infer-stack kube install --gpu=none --values dev/e2e_tests/kubeai-cpu-values.yaml
infer-stack kube install --gpu=none --values dev/e2e_tests/kubeai-cpu-values.yaml --apply
infer-stack kube doctor --gpu=none
```

## Migrate an existing Compose recovery ledger

`stack down --backend compose` removes runtime objects but retains the Compose
recovery snapshot. Changing `settings.yaml` alone cannot reinterpret historical
Compose rows as Kubernetes Models. Use an explicit transition:

```bash
infer-stack release --all --backend compose
infer-stack stack down --backend compose
infer-stack config set backend kubeai
infer-stack ledger rotate               # preview; strict old-runtime verification
infer-stack ledger rotate --yes         # commit archived history and new epoch
infer-stack ledger archives
infer-stack acquire <endpoint> --yes
```

All active leases must be released and all old runtime objects removed first.
An unreachable runtime blocks rotation, rather than being interpreted as empty.
KubeAI transitions also check managed Model CRs whose pods may not yet exist.
Stopped Compose containers must be removed with `stack down` as well, so restart
policies cannot resurrect them after transition. Configuration/catalogs are
untouched. SQLite backups under `<ledger-directory>/archives/` retain the old
leases/deployments/profile, including committed WAL contents. The archive is
published before a single transaction resets live rows and installs the new
backend snapshot. Retrying after interruption is safe. The live DB stays at the
same path/inode; already-open old-backend controllers refuse their next
mutation against the new snapshot.

`status` names the configured and active recovery backends separately. `gc`
refuses backend mismatch cleanly with `ledger rotate` guidance. `gc --forget`
only prunes terminal historical rows; it never adopts the configured backend.

## Point infer-stack at it

For a Kubernetes-only control host:

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

For temporary testing while keeping an existing Compose recovery epoch, use a
separate KubeAI authority/data root:

```bash
export INFER_STACK_BACKEND=kubeai
export INFER_STACK_DATA_DIR="$HOME/.local/share/infer_stack-kubeai"
```

Unset those variables to return to that host's existing Compose authority after
the Kubernetes node has been detached as described in the cluster setup guide.

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
`infer-stack kube install` is the installation authority for these profiles;
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

## Add a workstation with the K3s provisioning integration

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
infer-stack kube inventory
infer-stack kube bootstrap --apply
infer-stack kube install --apply
infer-stack kube doctor
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

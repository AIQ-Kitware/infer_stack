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

## One-time cluster setup

```bash
# 1. A cluster. For a single GPU host, k3s works out of the box:
./scripts/bootstrap_k3s.sh

# 2. Resource profiles: name -> the requests/limits/nodeSelector that one
#    "GPU unit" means on your cluster. These names are what the catalog's
#    `runtime.resource_profile` refers to.
cat > kubeai-values.yaml <<'EOF'
resourceProfiles:
  nvidia-gpu-rtx-4090:
    limits:
      nvidia.com/gpu: "1"
EOF

# 3. Install the chart (HF_TOKEN, if exported, is passed to the chart secret):
./scripts/install_kubeai.sh kubeai-values.yaml kubeai

# 4. A route to the gateway. The default base_url assumes a port-forward:
kubectl -n kubeai port-forward svc/kubeai 8000:80 &
```

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

Verify the setup before the first acquire — `doctor` checks the chain in
dependency order (cluster reachable → CRD installed → namespace → gateway):

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

## Semantics + limitations

- **Readiness is a real generation** through the gateway (same philosophy as
  compose): a Model CR existing — even with ready replicas — is not proof it
  can serve.
- **Placement can only fail at admission.** The render never rejects for
  capacity (the cluster schedules); a Model the cluster cannot place sits
  not-ready until the acquire's `--timeout`, which then rolls the lease back.
  The wait says why (`pod: Unschedulable`, `pod: ImagePullBackOff`).
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
- The gateway runs on the host running infer-stack, so that host is in every
  request's path. Dynamic routing (`dynamic_routing`) is compose-only; the
  KubeAI gateway uses static routes.

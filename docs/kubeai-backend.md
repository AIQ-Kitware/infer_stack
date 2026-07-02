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
| front door | LiteLLM gateway | KubeAI's own OpenAI-compatible gateway |
| request name | endpoint alias (LiteLLM route) | Model CR name (dns-slug of the served name) |
| auth | managed `LITELLM_MASTER_KEY` | none (`api_key: EMPTY`) |
| state dir | `<data>/leasing/compose/` | `<data>/leasing/kubeai/` (`models.yaml` + sidecar) |

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
source lease.env            # OPENAI_BASE_URL -> the KubeAI gateway
# note: request the model by its CR name (the env-file's endpoint mapping
# carries it), not the endpoint alias — KubeAI has no alias layer.
infer-stack release --env-file lease.env
```

## Semantics + limitations

- **Readiness is a real generation** through the gateway (same philosophy as
  compose): a Model CR existing — even with ready replicas — is not proof it
  can serve.
- **Placement can only fail at admission.** The render never rejects for
  capacity (the cluster schedules); a Model the cluster cannot place sits
  not-ready until the acquire's `--timeout`, which then rolls the lease back.
  Loud render failures do exist for: a missing `resource_profile` (no invalid
  CR is ever written), served-name collisions between simultaneously live
  deployments, and **ollama endpoints** — KubeAI serves models, not daemons,
  so the catalog's host-centric ollama endpoints don't map onto it yet; use
  `--backend compose` for those.
- `min_replicas` / `max_replicas` runtime keys pass through to the CR
  (default 1/1 — the lease lifecycle, not the autoscaler, decides residency).
- The legacy profile-era renderer (`infer_stack/backends/kubeai_renderer.py`,
  `kubeai_ops.py`) is superseded by this backend and kept only for reference.
- `dev/kubeai_e2e.sh` runs the full lifecycle (doctor → acquire → generation
  through the gateway → release → prune-verified) against a real cluster with
  an isolated config/data root; use it as the first smoke test on new setups.

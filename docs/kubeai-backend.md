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
semantics carry over: an idle `keep-warm` deployment retains its warm CR or an
existing on-demand CR at zero replicas; `stop` deployments are pruned on release;
explicit eviction removes the logical Model and its replicas. Ordinary `gc`
expires leases and reconciles reclaim policy. Hand-applied Models without the
managed label are never touched.

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
install are distribution-neutral. `bootstrap --provider=k3s` always targets
the local K3s server, never an unrelated selected context (reachable or stale).
Root admin credentials remain `0600`. Bootstrap refreshes a private user copy at
`~/.kube/infer-stack-k3s.yaml`, preserving existing default configs. Explicitly
select it with `export KUBECONFIG=~/.kube/infer-stack-k3s.yaml` for raw shell
commands. Bootstrap also saves the `kubeconfig` setting for infer-stack's own
kubectl/Helm/backend operations, including the TUI. Change it deliberately with
`infer-stack config set kubeconfig /absolute/path/to/config`; explicit command
kubeconfig flags take precedence. Newly frozen recovery profiles retain that
path so later shell context changes cannot redirect an existing authority.
Use `--version` to pin K3s; active clusters are never implicitly upgraded.

`inventory`, `doctor`, and `status` accept `--json`. Errors belong to individual
probes, so a missing KubeAI ConfigMap cannot erase GPU node facts. A cluster with
a leftover CRD but no namespace/chart can be diagnosed and repaired with these
commands. Kubernetes GPU allocation and GFD readiness are separate from host
GPU visibility. A cluster-scoped RuntimeClass is reported separately from node
runtime evidence; missing handler evidence remains unknown, never an all-clear.
Apply bootstrap/install verifies unknown handlers on GPU nodes with small
`busybox:1.37.0` canaries using `runtimeClassName=nvidia`. These reserve no GPU
and retain completed pods in `kube-system` as diagnostic evidence. Inactive
0/0 NVIDIA DaemonSets are N/A, so an optional MPS component does not block readiness.

Mutation requires `--apply` (alias `--yes`). Both mutating commands support
`--dry-run` (alias `--plan`). Install derives one profile per GPU product with
GPU requests/limits, a product selector and `runtimeClassName=nvidia` when that
RuntimeClass exists. Kubernetes chooses placement; infer-stack assigns no host
GPU indices. Models request multiple GPUs via the existing profile/count syntax.

KubeAI Model names are limited to 40 characters. Short served names keep their
DNS slug; longer names use a readable prefix plus a digest of the full name.
Dynamic routes additionally reserve the deployment suffix, keeping dedicated
deployments distinct. Gateway routes use the same canonical name; endpoint
aliases and model source IDs remain unchanged.

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
`kube setup` is deprecated and delegates to `kube install`; it no longer
reconciles prerequisites independently. The bootstrap compatibility script
invokes the unambiguous `kube k3s bootstrap` local-server command.

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

Local `infer-stack tui` follows this same authority boundary. Run it on the
authority/admin host. With a configured `server_url`, it becomes an HTTP client
to the canonical authority instead; see [remote control service](#remote-control-service).
A worker needs only its Kubernetes agent/runtime.

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

After join, use `infer-stack kube k3s status` on the worker for local membership.
Joining does not change its admin kubeconfig or grant cluster-admin credentials;
verify it from the control plane with `infer-stack kube node status <name>`.
An already-running agent must match the requested server, node name and version
or join refuses.

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
- **An idle warm Model stays only while its pod has started**, as an
  idle keep-warm container does on compose (a crash-looping pod counts, like
  a restarting container). One whose pod is gone, still Pending or finished
  is pruned at the next render. An explicit `min_replicas: 0` endpoint instead
  retains an already-existing logical Model without pods; GC does not recreate
  a missing idle Model. A generation request wakes it.
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
- `min_replicas` / `max_replicas` default to 1/1. The catalog validates bounds
  and includes nondefault policy in deployment compatibility. See below for
  opt-in on-demand serving and load-based replication.
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

Acquire/wait logs changed readiness states once: pending/scheduler reasons,
scheduled node and image, container startup/model loading, replica readiness,
and finally generation verification. It does not log every unchanged poll.

## TUI cluster monitoring and control

Open `infer-stack tui`, expand runtime (`c`), and select **Cluster**. The cached
node table includes Ready, scheduling, GPU allocation/requests, product/memory,
GFD and observed runtime startup evidence. The pod table includes KubeAI and
NVIDIA components, their nodes, startup reasons, readiness, restarts and GPU
requests. **Refresh cluster** requests a new sample; **Doctor** runs the same
inventory/readiness policy as the CLI once, with actionable fixes.

Three batched list requests supply the cluster view. Automatic sampling occurs
at most every 15 seconds (or the longer configured observation interval), stops
when hidden, and never overlaps a previous sample. Ledger refreshes remain
independent. Partial RBAC/network failures retain other facts and show unknown
values rather than an empty healthy cluster. Scheduled GPU requests count all
namespaces and exclude completed pods; no remote GPU utilization is inferred.

Node detach/attach previews and confirmation reuse the CLI's ownership-aware
lifecycle operations. Detach warns about eviction and emptyDir loss; attach
requires the operator to stop local Compose GPU work first. Apply reconciles
through the controller even without a local gateway render; Down confirms
managed Model/gateway teardown and retains leases. Existing Acquire, Release,
Evict and pod log controls remain available. Endpoint editing exposes
`runtime.resource_profile` instead of physical host GPU indices.


Kubernetes readiness has separate stages: declared Model, scheduled pod,
running container, ready replica, and generation verified by an explicit API
request. CLI inventory and the Cluster summary read KubeAI's
`status.replicas.all/ready`, with fallback for legacy flat counters; missing
counts remain unknown. A CR alone never makes an endpoint eligible for the API
picker. Replica-ready candidates are labeled generation-unverified until
**Send** or **Test all** succeeds. Verification is last-request evidence for the
endpoint, not proof that every replica answered. Replacement, observed restarts,
readiness loss or a failed request invalidates that evidence.

The global ledger/API observation uses one cached Model-list read, reusing a
fresh Cluster sample when available. Full node/pod monitoring remains limited
to the visible Cluster tab. Passive refresh performs no generation requests.
Doctor reports retained terminated runtime evidence as a non-failing **WARN**,
with `kube install --apply` as the fresh-verification step.


The KubeAI endpoint editor exposes environment and resource profiles, with no
arbitrary container image/command controls. Custom images belong in installed
KubeAI resource/model-server profiles. `runtime.image` overrides are rejected
by rendering instead of ignored; command/mount overrides remain unsupported.
Editing an existing unsupported launch reports an error rather than dropping
those settings. Compose retains its image and command controls.

The editor's vLLM data-parallel size consumes GPUs inside one serving deployment;
it does not create independent Kubernetes replicas. KubeAI pod replication uses
`runtime.min_replicas` / `runtime.max_replicas`.

## Placement and replica policy

Administrators classify canonical Kubernetes node names and install selector
profiles using the existing Helm release:

```bash
infer-stack kube node classify namek.kitware.com --pool interactive --class rtx3090
infer-stack kube node classify namek.kitware.com --pool interactive --class rtx3090 --apply
```

Users can narrow an approved endpoint with `acquire`, `access`, or `run`
`--node namek.kitware.com` or `--pool interactive`. Node and pool constraints
intersect the catalog's node/pool constraints; they cannot broaden permission.
Kubernetes still assigns physical GPUs. Constraint equality is required for
sharing, and simultaneously live deployments with the same public alias but
different hard constraints are refused because alias routing would defeat the
placement promise. Compose rejects these cluster constraints.

An on-demand catalog endpoint can declare:

```yaml
runtime:
  resource_profile: nvidia-gpu-rtx-3090
  min_replicas: 0
  max_replicas: 2
  target_requests: 8
  scale_down_delay_seconds: 60
  load_balancing: LeastLoad
```

`PrefixHash` is the other supported load-balancing strategy. Minimum zero is
explicit opt-in; ordinary warm endpoints keep the 1/1 default. The lease protects
logical demand, while KubeAI owns replica count. Reclaim `stop` removes the Model
after the last lease; `keep-warm` retains its logical route even when the
autoscaler returns it to zero. Readiness always sends a real generation, which
can wake an on-demand Model. Passive status refresh does not wake it. Compose
refuses nondefault replica/autoscaling policy rather than silently ignoring it.

KubeAI's Helm-level `modelAutoscaling.timeWindow` defaults to ten minutes and
affects return-to-zero in addition to per-Model scale-down delay. The integration
harness uses a short test window; select production values from measured
traffic and startup latency. Status reports desired min/max, observed/ready
replica counts, individual nodes/reasons, and suggested actions. Replica
readiness alone does not establish generation readiness; unknown counts stay
unknown. Startup event history and latency distributions remain follow-up work.

## Remote control service

Run one canonical server on the machine holding the approved catalog, recovery
data, Kubernetes credentials and backend/gateway configuration. Preserve that
data root across restarts. The server reconciles TTLs periodically; clients do
not need kubeconfig or copies of its ledger.

Create a private control token on that host:

```bash
umask 077
python -c 'import secrets; from pathlib import Path; Path("control.token").write_text(secrets.token_urlsafe(32))'
infer-stack server --token-file ./control.token
```

The default listener is `127.0.0.1:14043`. Any off-host listener requires TLS:

```bash
infer-stack server --host 0.0.0.0 --token-file ./control.token \
  --tls-cert /absolute/path/control.crt --tls-key /absolute/path/control.key
```

Use a certificate valid for the hostname clients use. Tokens, identity files,
DSNs and TLS keys must be private and owned by the service user. TLS keys and
certificates load at server startup; token/identity files reload per request for
rotation and revocation. Configure clients with the securely distributed token
and CA (or system trust):

```bash
infer-stack config set server_url https://control.example:14043
infer-stack config set server_token_file /absolute/path/control.token
infer-stack config set server_ca_file /absolute/path/control-ca.pem
infer-stack acquire qwen-coder --ttl 2h --env-file lease.env --yes
infer-stack status
infer-stack operations
infer-stack operations --audit --limit 100
infer-stack kube nodes --json
infer-stack tui
infer-stack release --env-file lease.env
```

`--server-url`, `--server-token-file`, and `--server-ca-file` override those
settings; `INFER_STACK_SERVER` also selects the server URL. Remote lease verbs
include acquire, access, release, wait, renew, run, apply,
gc and evict. `operations OPERATION_ID` recovers a durable result. Mutations
return an operation before admission/readiness finishes; `--idempotency-key`
on acquire/access makes an exact request retry recover the same operation.
Connection retries keep their generated key. A generation timeout releases a
new acquisition's lease; a timeout on an explicit wait retains a held lease.

Use `access` or `run` for approved external endpoints or mixed bundles.
External-only access returns a descriptor with `lease_id: null`; it verifies
route publication and gateway readiness without sending inference upstream.
Managed bundle members still require a successful generation. Remote `run`
preserves the child exit status and releases only a lease it actually acquired.
Descriptors include the immutable catalog revision accepted by the operation.

The control credential authorizes resource operations. Inference credentials
are separately provisioned: connection descriptors and env files reference
`INFER_STACK_INFERENCE_KEY` without exposing the server's control token or
gateway master key. Set that variable from your inference credential before
sourcing `lease.env` or using `run`. The server does not yet mint user-scoped
inference keys. Only inference's actual base URL determines whether an API key
is required; a private development KubeAI service may have none.

For individual ownership and restricted users, replace `--token-file` with
`--identity-file /absolute/path/identities.json`. The private JSON registry is:

```json
{
  "identities": [
    {
      "principal": "alice",
      "role": "lease-user",
      "token_sha256": "<64 lowercase hex characters: SHA256 of token text without surrounding whitespace>",
      "allowed_endpoints": ["qwen-coder"],
      "max_active_leases": 2,
      "max_ttl_seconds": 7200
    }
  ]
}
```

Supported roles are `observer` (read-only), `lease-user` (own leases and allowed
endpoints), `operator`, and `catalog-admin`. Catalog administrators have operator
operations plus exclusive permission to inspect full catalog revisions and
approve new ones. A shared `--token-file` identity has operator capability;
catalog publication requires a named `catalog-admin` in the identity file. Lease-user
defaults are ten active/pending leases and a maximum one-day TTL. Endpoint
scopes apply after bundle expansion. Pending and active lease admission is
counted transactionally. Removing a credential revokes subsequent requests;
already accepted operations retain their durable intent. Observers/operators
see all state; lease users see their own leases, operations and related usage.

The listener bounds simultaneous connections, including TLS handshakes
(`--max-connections=32`), and requests per peer/principal
(`--requests-per-minute=600`; zero disables that rate limit). HTTP 429 reports
`Retry-After`. The API accepts typed catalog operations, not shell commands,
container launch specs or arbitrary Kubernetes YAML. `/v1/events` sends a
current SSE snapshot; polling operations is the durable recovery contract.
`/v1/audit` pages immutable transitions with `after` (sequence ID) and `limit`
(1..1000); lease users see their own records. `/v1/nodes` reports read-only
Kubernetes inventory, including per-device facts and GPU requests across
namespaces. Request counts are reserved capacity, not observed utilization.
Pod-list failure leaves requests unknown while preserving known node facts.
`kube nodes` uses that API when a control server is configured; the remote TUI
also displays it. Full local-dashboard parity, node administration and streaming
logs/events remain incomplete.

### Approved catalog revisions

The server seeds its approved catalog from `--catalog` or the configured local
catalog when first started. Subsequent starts keep the database's approved
revision even if that YAML changed. Local catalog commands continue to edit
files; they do not change a remote server's approved catalog.

Review a complete catalog file, then publish using a catalog-admin credential:

```bash
infer-stack catalog revisions
infer-stack catalog publish reviewed.yaml --dry-run
infer-stack catalog publish reviewed.yaml --base-revision CURRENT_SHA256 \
  --idempotency-key catalog-review-2026-10-08
infer-stack catalog revisions REVISION_SHA256
```

`catalog revisions` reports the active hash and retained revision provenance;
giving a hash retrieves that immutable source. Publication returns a durable
operation. The base hash is required: competing edits fail with a catalog
conflict instead of replacing a newer approval. Retry the exact request with
the same idempotency key to recover a lost response. After a conflict, review
the new current revision before submitting a new request/key. Activation and
its success/audit record are one database transaction.

Approval applies to future acquisitions. Existing leases and serving processes
keep their accepted launch specifications; redefining a resident alias can
still be refused by the publication compatibility rules. Queued operations
retain their original catalog snapshots. Full sources stay visible only to
catalog administrators. External keys must use `api_key_env` references; remote
catalog URLs cannot embed credentials, queries or fragments. Keep secrets in
the service's managed environment, outside reviewed catalog files.

### PostgreSQL authority

SQLite on the server's local disk is the default. PostgreSQL is optional for
the same single-authority service:

```bash
pip install 'infer-stack[postgres]'
infer-stack server --token-file ./control.token \
  --postgres-dsn-file /absolute/path/postgres.dsn
```

The DSN file contains a libpq DSN; off-host databases require
`sslmode=verify-full` with a trusted CA. The saved `postgres_dsn_file` setting
selects it for local controller commands too. Use a dedicated authority database
and retain the backend data root/configuration alongside it. PostgreSQL locks
serialize ledger transactions and whole publications and refuse a second
server. They do not provide safe automatic takeover after partitions; controller
HA is unavailable until destination-side fencing is implemented.

There is no automatic SQLite import. Pointing at an empty PostgreSQL database
starts a new authority, so do not do that against a namespace still managed by
another ledger. Explicit authority migration and restore acceptance remain
planned. Use native PostgreSQL backups; SQLite archive/rotation commands refuse
this store. Supported schema compatibility is checked at startup, including
refusal of future versions. Database failure prevents new mutations; existing
inference continues independently where its data plane remains available.

## Resource and energy observations

`infer-stack usage --json` reports immutable deployment/replica intervals,
GPU-hours and energy provenance. `--node` and `--deployment` filter records; a
remote client reads the same facts through the API. The remote TUI includes a
usage table. Import explicit collector facts on the authority host:

```bash
infer-stack usage --import-file collector-intervals.json --json
```

The file is a JSON list of `UsageInterval` mappings; its contract is in
`infer_stack/accounting.py`. Times are UTC epoch seconds and intervals are
half-open. Measured samples are total GPU watts for exactly the recorded UUID
set, with source and a maximum acceptable sample gap. Large gaps stay unknown;
an optional per-device proxy must name its method, source and version. Reports
distinguish measured, partial, proxy, mixed and resource-time-only results, and
show measurement coverage. Unknown energy is not zero. Shared leases do not
receive duplicate copies of deployment energy.

Imports are immutable and idempotent per interval ID and reject overlapping
replica/device intervals. A multi-record import can commit earlier records
before a later overlap fails; retrying unchanged records is safe. Querying an
empty store creates no database. Live NVML/DCGM collection, whole-node power,
reservation/activity aggregates and lease energy attribution are not yet
implemented; imported/synthetic evidence must not be described as live telemetry.

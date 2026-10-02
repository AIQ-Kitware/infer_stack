# Infer Stack

[![PyPI version](https://img.shields.io/pypi/v/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![Python versions](https://img.shields.io/pypi/pyversions/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![License](https://img.shields.io/pypi/l/infer-stack.svg)](https://github.com/AIQ-Kitware/infer_stack/blob/main/LICENSE)

Declare models in a catalog (`infer-stack catalog …`) and `acquire` or `run`
endpoints on demand. `infer-stack help tree` prints the whole command surface;
[docs/source/manual/](docs/source/manual/) has the Ollama + Open WebUI
tutorial and the leasing demo.

## Primary leasing workflow

The normal user path has three steps and no separate publication phase:

```bash
infer-stack config init
infer-stack catalog suggest --apply
infer-stack acquire <endpoint>
```

No GPU here? `infer-stack catalog suggest --simulator --apply` adds
`mock-smol`, a simulator that answers like vLLM with random text, so the same
three steps run on any host with Docker ([docs/mock-endpoints.md](docs/mock-endpoints.md)).

`settings.yaml` (`infer-stack config …`) and `catalog.yaml` are the user
configuration. Leasing keeps an
internal frozen recovery snapshot so a crash cannot re-render committed state
with different settings, but ordinary `acquire` advances that snapshot
automatically. Compatible catalog additions can be acquired while other models
are live; conflicting redefinitions or global setting changes take effect once
the affected stack is quiescent. `infer-stack config publish` is an advanced
pre-seeding/preview tool for multi-catalog operators, not a required fourth
step. See [ADR 0001](docs/adr/0001-user-config-is-authoritative.md).

## Related work

- [HyperQwen](https://github.com/syv-ai/HyperQwen) is a specialized model-preparation
  and serving stack for running Qwen3.8-27B efficiently on 24-GiB consumer GPUs,
  with its published tuning and measurements centered on the RTX 3090.
  infer-stack's suggestion for it delegates the requantization, patched-vLLM
  launcher, speculative decoding, and GPU-level tuning to HyperQwen's image,
  described entirely in catalog data (see "Images with their own launcher"
  below); infer-stack adds hardware discovery, catalog suggestions,
  exact GPU affinity, lease lifecycle, and routing around that serving stack.
  `catalog suggest` offers its fast endpoint on any Ampere-or-newer GPU with
  24 GiB; on a detected RTX 3090 it also offers explicit `-long` (150K) and
  `-huge` (245,760) endpoint variants from HyperQwen's measured single-user
  profiles. It has also been measured unchanged on an RTX PRO 6000 Blackwell.
  The base suggestion is
  named `qwen3.8-27b-dbirks-hyperqwen` because it starts from the
  `dbirks/Qwen3.8-27B-W4A16-AutoRound` derivative; the unsuffixed
  `qwen3.8-27b` identity is left available for the official
  `Qwen/Qwen3.8-27B` checkpoint.

## Supported platform

`infer-stack` supports **Linux hosts only**. Its process locking, container
runtime integration, GPU discovery, and deployment workflows are intentionally
Linux-oriented. Windows is not a supported or tested execution platform.

Operational and security constraints that are accepted during the current
planning-stage release are tracked in
[docs/planning/known-limitations.md](docs/planning/known-limitations.md).

`infer-stack` serves the endpoints declared in a catalog. `acquire` takes a
lease on an endpoint; the controller places its engine on free GPUs and
reconciles the backend to run it:

* **engines**: vLLM (one container per deployment) and Ollama (one daemon per
  `runtime_hosts` entry, serving many tags);
* **LiteLLM gateway**: one OpenAI base URL, `http://127.0.0.1:14042/v1`, in
  front of every endpoint alias. On by default; `config set litellm false`
  drops it;
* **Open WebUI**: on by default at `http://127.0.0.1:13000`;
  `config set ui false` or `acquire --no-ui` drops it;
* **reverse proxy**: an optional single-port nginx in front of both.

Two backends run this: **Compose** (single host; vLLM and Ollama) and
**KubeAI** (a Kubernetes cluster; vLLM only).

## Main commands

```bash
infer-stack config init               # data dir + default backend -> settings.yaml
infer-stack catalog suggest --apply   # seed catalog.yaml from this host's GPUs
infer-stack catalog show              # what can be acquired
infer-stack kube inventory           # Kubernetes/GPU/KubeAI facts, before or after install
infer-stack kube doctor              # detailed readiness + actionable fixes
infer-stack kube node detach NODE     # preview handing a cluster GPU host to Compose
infer-stack acquire <endpoint>        # lease, render, bring up, wait for a real generation
infer-stack access <endpoint> --env-file e.env  # reach it, managed or external; leases only what runs here
infer-stack test <endpoint>           # one generation through the gateway
infer-stack leases                    # desired vs running, per deployment
infer-stack status                    # paths, backend and a lease summary
infer-stack release --all             # drop every lease
infer-stack paths                     # where settings, catalog, ledger and caches live
infer-stack version
infer-stack help tree                 # the whole command surface
```

The CLI is built on [`kwconf`](https://github.com/Erotemic/kwconf),
so every subcommand is also importable as a Python class:

```python
from infer_stack.cli import AcquireCLI, TestCLI

AcquireCLI.main(argv=False, names=['smol135-1'], yes=True)
TestCLI.main(argv=False, name='smol135-1')
```

`manage.py` and `infer-stack` are aliases for the same entry point;
shell examples below use `infer-stack`.

## Operating the rendered Compose stack

`ps` and `logs` read the backend directly; `stack` wraps `docker compose` on
the rendered project, so you never `cd` into it or repeat `-f`/`--env-file`.

```bash
infer-stack ps                            # engines, gateway, UI
infer-stack ps -a                         # include exited instances
infer-stack logs -f <endpoint>            # follow whatever serves an endpoint
infer-stack logs --tail 200 litellm       # the gateway's backlog
infer-stack logs -f --raw litellm         # full LiteLLM tracebacks
infer-stack stack restart open-webui      # docker compose restart
infer-stack stack stop                    # stop everything (no remove)
infer-stack stack start                   # start it back up
infer-stack stack pull                    # refresh images
infer-stack stack compose -- ps --format json   # any other docker compose command
```

`logs` accepts a service or pod name, a container id prefix, a deployment id
or an endpoint alias. Interactive `infer-stack logs -f` compacts only
explicitly registered, known-noisy LiteLLM traceback shapes; unknown
tracebacks pass through unchanged. Redirected or piped output stays raw, and
`--raw` disables compaction in an interactive follow. `--no-color` drops the
name-prefix colors. The TUI uses the same compactor.

Ollama tags are pulled into the daemon on the first `acquire` of an endpoint
that serves them. `stack compose` reaches the daemon for anything else, e.g.
`infer-stack stack compose -- exec ollama-local-ollama ollama list`.

On the KubeAI backend `ps` and `logs` read pods, and the `stack` Compose verbs
act on the gateway's Compose project on this host.

## Inspect an endpoint before running it

```bash
infer-stack catalog show <endpoint>
infer-stack acquire <endpoint> --no-apply   # declare + write the compose project; start nothing
infer-stack paths leasing                   # where docker-compose.yml landed
infer-stack apply                           # start it (or `release --all` to discard)
```

## Catalog model

`catalog.yaml` is the one user-edited description of what can run. Its
sections:

* `models`: weight sources (`hf://org/name`, with optional `revision`,
  `quantization`, `dtype`);
* `endpoints`: served API names. Each picks an `engine` (`vllm` or `ollama`),
  a `model` (a `models` key, or an Ollama tag), and optionally `runtime`
  (vLLM settings), `protocol`, `placement`, `sharing` and `reclaim`;
* `runtime_hosts`: Ollama daemons, each with its GPUs and daemon settings;
* `bundles`: named lists of endpoints to acquire together.

An endpoint can instead name a server that already runs elsewhere, with
`external: {api_base, model, api_key_env}` in place of `engine`/`model`.
It has no lease: `infer-stack access` (or `run`) publishes its route on the
LiteLLM front door, and the client asks for the alias through the same base
URL as any other endpoint. Moving an alias between a managed runtime and an
external server changes neither the alias nor the workflow:

```bash
infer-stack env REMOTE_QWEN_KEY=sk-...          # the upstream's key, by name
infer-stack access qwen-remote --env-file qwen.env
source qwen.env                                 # OPENAI_BASE_URL, OPENAI_API_KEY, ...
```

`api_base` is resolved from inside the gateway's container or pod:
`localhost` there is the gateway itself, so a server on the same host is
reached through the Docker bridge address (e.g. `http://172.17.0.1:8000/v1`).
See [docs/planning/external-endpoints.md](docs/planning/external-endpoints.md).

```yaml
models:
  smol135:
    source: hf://HuggingFaceTB/SmolLM2-135M-Instruct

endpoints:
  smol135-1:
    engine: vllm
    model: smol135
    runtime: {max_model_len: 8192}
  chat:
    engine: ollama
    host: local-ollama
    model: qwen3.5:4b

runtime_hosts:
  local-ollama:
    engine: ollama
    placement: {gpu_indices: [0]}
    settings: {keep_alive: 30m}

bundles:
  both: [smol135-1, chat]
```

Edit it with `infer-stack catalog model|endpoint|host|bundle add …`, or by
hand with `infer-stack catalog edit`; `infer-stack catalog validate` checks it.
The schema reference is the docstring of `infer_stack/leasing/catalog.py`.

Shapes the stack renders:

```text
Open WebUI -> LiteLLM -> vLLM / Ollama     # the default
                     \-> an external OpenAI-compatible server (external:)
Open WebUI -> vLLM / Ollama                # config set litellm false
```

The named stack profiles of earlier releases (`setup`, `switch`,
`--profile`) are gone; see
[docs/stack-graph-profiles.md](docs/stack-graph-profiles.md).

## Where config and rendered artifacts live

`infer-stack` follows XDG basedir conventions, so the directory you invoke it
from never changes which config it reads or where it writes. There are two
roots:

| What | Default location | How to relocate |
| --- | --- | --- |
| `settings.yaml`, `catalog.yaml` | `~/.config/infer_stack/` (resp. `$XDG_CONFIG_HOME`) | `--config-dir` or `INFER_STACK_CONFIG_DIR` |
| **Everything generated**: `leasing/` (the ledger, the compose project, its `.env`) and the bind-mounted state (`hf-cache/`, `vllm-cache/`, `open-webui/`, `ollama/`, …) | `~/.local/share/infer_stack/` (resp. `$XDG_DATA_HOME`) | `config set data_dir <path>`, `--data-dir` or `INFER_STACK_DATA_DIR` |

`infer-stack paths` prints every resolved path and whether it exists.

The data dir **relocates the one infer-stack installation controlling a
host/backend; it does not create an isolated second installation**. Do not run
controllers from multiple config/data roots against the same Docker host or
Kubernetes namespace. See the
[single-owner limitation](docs/planning/known-limitations.md#one-control-plane-per-host-or-backend-namespace).

```bash
# Persist it once; later commands read it from settings.yaml.
infer-stack config set data_dir /data/service/docker/infer-stack

# Or keep both roots in a checkout for an ad-hoc experiment.
export INFER_STACK_CONFIG_DIR=$PWD/cfg INFER_STACK_DATA_DIR=$PWD/stack
infer-stack paths
```

`--config-dir` / `--data-dir` are accepted by every subcommand, after the
subcommand name.

## Constraining placement to specific GPUs

```bash
# Only place onto GPU 1 (e.g. GPU 0 is running a display).
infer-stack acquire <endpoint> --allowed-gpus 1

# Or confine a TP=2 endpoint to physical GPUs 1 and 3.
infer-stack acquire <tp2-endpoint> --allowed-gpus 1,3
```

`--allowed-gpus` (or `INFER_STACK_ALLOWED_GPUS=1,3`) filters the detected
inventory before placement for that call only. Real indices are preserved, so
the rendered compose stack pins `device_ids` to those exact GPUs. The durable
forms live in data:

* `placement: {gpu_indices: [1]}` on a vLLM endpoint pins it exactly (the list
  length must equal tp×pp×dp); Ollama daemons pin through their
  `runtime_hosts` entry;
* `placement: {min_vram_gib: 24}` makes smaller GPUs ineligible;
* `config set skip_display_gpus true` (or `--skip-display-gpus`) leaves the
  GPU driving a monitor free.

## Demos / integration recipes

The user manual under [docs/source/manual/](docs/source/manual/) has two
walkthroughs on the current CLI:
[the Ollama + Open WebUI tutorial](docs/source/manual/ollama-openwebui-tutorial.md)
and [the leasing demo](docs/source/manual/leasing-demo.md) (standing service,
Open WebUI, several models side by side).

---

## Backend 1: Compose

Use Compose for single-host serving. It runs vLLM and Ollama engines, mixed
freely, behind the optional gateway and UI.

### Getting started

Prerequisite: Docker and the `docker compose` plugin.

```bash
infer-stack config init --backend compose
infer-stack doctor --gpu
infer-stack catalog init

# A vLLM endpoint.
infer-stack catalog model add smol135 --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
infer-stack catalog endpoint add --model smol135    # -> smol135-1
infer-stack acquire smol135-1

# An Ollama endpoint, on a daemon pinned to GPU 0.
infer-stack catalog host add local-ollama --engine ollama --gpu 0
infer-stack catalog endpoint add chat --engine ollama --host local-ollama --model qwen3.5:4b
infer-stack acquire chat
```

`infer-stack catalog suggest --apply` fills the catalog with endpoints sized
for the detected GPUs instead.

### Test that it is responding

With LiteLLM enabled, every endpoint is reachable by its alias at:

```text
http://127.0.0.1:14042/v1
```

`acquire` already blocks until the endpoint returns a real generation through
that front door, which is stronger than Docker's container health. After
`acquire --no-wait`, block separately; to check again later, send one request:

```bash
infer-stack wait smol135-1
infer-stack test smol135-1
infer-stack test chat --prompt "Name three colors." --max-tokens 32
```

Clients read the front door and the managed key from the env file:

```bash
export OPENAI_BASE_URL=$(infer-stack env OPENAI_BASE_URL)
export OPENAI_API_KEY=$(infer-stack env LITELLM_MASTER_KEY)
```

or get both, plus per-endpoint names, from
`infer-stack acquire <endpoint> --env-file lease.env`.

To replace the gateway's master key (refused while leases are active; the
gateway restarts, and clients must fetch the key again):

```bash
infer-stack secrets rotate
```

### Stop it

```bash
infer-stack release --all            # drop every lease
infer-stack release --all --evict    # ...and stop the engines now
infer-stack clean -f                 # no leases, nothing on a GPU; the gateway stays
infer-stack stack down               # docker compose down, bypassing the ledger
```

After a plain `release`, `keep-warm` endpoints (the default `reclaim` policy)
stay loaded until another lease needs their GPUs or `infer-stack evict` stops
them. `stack down` releases no lease, so the next `apply` or `acquire` brings
leased models back.

All state is bind-mounted from the data dir, so none of these delete it,
including `stack down --volumes`. For a destructive reset, remove the
directories `infer-stack paths` lists.

### Open WebUI authentication

Open WebUI runs with `WEBUI_AUTH=False`: no login screen, and anyone who can
reach port 13000 gets the UI. No setting changes that. Keep the host on a
trusted network, or run without the UI (`config set ui false`).

### Reverse proxy

An optional nginx service publishes one HTTP port with the UI at `/` and the
API at `/v1`. It needs the LiteLLM gateway.

```bash
infer-stack config set reverse_proxy true                        # port 80
infer-stack config set reverse_proxy '{enabled: true, port: 8080}'
```

`acquire --reverse-proxy` turns it on for one call. Add `config_path:
/path/to/nginx.conf` to the block to mount your own config at
`/etc/nginx/conf.d/default.conf` instead of the generated one.

It does no TLS and no authentication. Only the one port is published and no
certificate is mounted, so terminate TLS in a proxy in front of it. The TLS
and LDAP settings of the pre-leasing profiles no longer exist.

### Persistent state and database layout

Everything lives under the data dir (`infer-stack paths`):

* `open-webui/`: Open WebUI's data directory (accounts, chats, settings),
  mounted at `/app/backend/data`;
* `postgres-litellm/`: LiteLLM's route store, rendered only with
  `config set dynamic_routing true`;
* `ollama/`: the Ollama model store, mounted at `/root/.ollama`;
* `hf-cache/`, `vllm-cache/`, `torch-cache/`, `triton-cache/`, `cuda-cache/`:
  vLLM weights and compile caches (see
  [docs/persistent-caches-and-warm-restarts.md](docs/persistent-caches-and-warm-restarts.md));
* `runtime/`: directories an endpoint's `runtime.mounts` asks for;
* `leasing/`: the ledger and the rendered compose project.

Open WebUI chat history is not tied to the models currently served, so old
chats may name aliases the gateway no longer advertises. That is expected.

### Custom .env values are preserved

The compose project's `.env` holds the managed secrets (`LITELLM_MASTER_KEY`,
`HF_TOKEN`, …). `infer-stack env KEY=VALUE` merges a value into it, and keys
you add are kept across renders. Compose uses the file for interpolation, so a
key reaches a container only when the rendered service references it (as
`HF_TOKEN` does for vLLM). Set `HF_TOKEN` before the first `acquire` of a
gated model.

### Switching models

There is no single active model to switch. `acquire` another endpoint and it
runs beside the first; release the first when you are done:

```bash
infer-stack acquire smol135-1
infer-stack acquire chat          # now both are served
infer-stack leases                # find smol135-1's lease id
infer-stack release <lease-id>
```

The gateway carries a route for every catalog endpoint, so adding or removing
a model does not recreate LiteLLM, and Open WebUI stays up. When GPUs are
short, `acquire` fails fast; `--queue` waits for a GPU instead, and idle
`keep-warm` deployments are evicted to make room. vLLM containers are named
after the served alias (`vllm-<alias>`), so `docker ps` and
`infer-stack logs vllm-<alias>` identify them.

### Protocol modes for base vs. instruct models

An endpoint's `protocol` is `chat` (the default) or `completions`. It decides
which surface the readiness probe and `infer-stack test` use, so a base model
without a chat template must declare `completions` or its `acquire` never sees
a ready generation:

```bash
infer-stack catalog model add pythia-160m --source hf://EleutherAI/pythia-160m
infer-stack catalog endpoint add --model pythia-160m --protocol completions
infer-stack acquire pythia-160m-1
infer-stack test pythia-160m-1              # hits /v1/completions
```

The gateway forwards `/v1/completions` unchanged, so evaluation clients that
need exact prompt control should call it directly. Open WebUI is a chat UI and
sends chat requests, which a base model cannot answer.

### Images with their own launcher

Some images wrap vLLM in their own launcher and are configured through
environment variables rather than `vllm serve` flags. Describe that in the
endpoint's `runtime`; infer-stack has no model-specific code for it:

```yaml
runtime:
  image: example.org/my-vllm-launcher:1.0
  max_model_len: 65536
  gpu_memory_utilization: 0.93
  command: [single]              # replaces `vllm serve MODEL <flags>`
  env:                           # container environment
    PORT: '{port}'
    MAX_LEN: '{max_model_len}'   # filled from the field above
    GPU_UTIL: '{gpu_memory_utilization}'
    EXTRA_ARGS: '--served-model-name={served_model_name}'
  mounts:                        # persisted under the runtime data dir
    /app/models: my-launcher/models
```

Changing the launcher's mode is a data edit, in the catalog or the TUI's
endpoint editor. The HyperQwen suggestion (see "Related work") is a worked
example: on an RTX 3090, `catalog suggest` emits its measured context
profiles as separate `-long` and `-huge` endpoints. The hardware check
happens only while suggesting; the resulting catalog holds ordinary explicit
runtime data, so `apply`/`acquire` never retunes an endpoint after the fact.

- `{max_model_len}`, `{gpu_memory_utilization}`, `{served_model_name}` and
  `{port}` are filled in from the endpoint, so a launcher that takes them
  through its own variables stays in step when the fields change.
- Env values are written as strings (`true`/`false` for booleans), and `$`
  is literal. `HF_TOKEN`, `VLLM_ATTENTION_BACKEND`, `CUDA_VISIBLE_DEVICES` and
  `NVIDIA_VISIBLE_DEVICES` are infer-stack's and are refused.
- `extra_args` stay what they were: flags appended to the stock `vllm serve`
  command, after infer-stack's own, so vLLM keeps the extra value for a
  repeated flag. Repeating a flag infer-stack acts on (served name, parallel
  sizes, `--max-model-len`) is refused; with `command`, pass flags through the
  launcher instead.
- All of these are deployment identity: endpoints that launch differently
  never share a process.
- `env` works on both backends (on KubeAI it becomes the Model's `spec.env`).
  `command` and `mounts` are Compose only; KubeAI refuses an endpoint with
  either rather than serve stock vLLM in its place.

### Reasoning / thinking models

vLLM separates a reasoning trace from the answer when it is started with a
reasoning parser. Pass the flag through `runtime.extra_args`:

```yaml
endpoints:
  qwen3-think:
    engine: vllm
    model: qwen3-0.6b
    runtime:
      extra_args: [--reasoning-parser=qwen3]
```

The parser name depends on the model family and the vLLM version (`vllm serve
--help` lists them). To test end to end:

```bash
infer-stack test qwen3-think --prompt "Think step by step: 17*23" --max-tokens 512

# Streaming, through the gateway:
curl -N "$(infer-stack env OPENAI_BASE_URL)/chat/completions" \
  -H "Authorization: Bearer $(infer-stack env LITELLM_MASTER_KEY)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-think","stream":true,
       "messages":[{"role":"user","content":"Think step by step: 17*23"}]}'
```

In Open WebUI, reasoning shows up best with streaming enabled in the
chat settings.

---

## Backend 2: KubeAI

`--backend kubeai` runs the same leasing verbs against a Kubernetes cluster
running [KubeAI](https://www.kubeai.org): the same catalog, ledger, TTLs,
env file and TUI, with `Model` custom resources in place of compose
services and the cluster scheduler in place of the local GPU planner. The
LiteLLM gateway still fronts everything, so a card sees one `OPENAI_BASE_URL`,
the managed key and the endpoint alias on either backend.

* Creating/joining a cluster and the distribution boundary:
  [docs/cluster-setup.md](docs/cluster-setup.md).
* KubeAI setup, settings and semantics:
  [docs/kubeai-backend.md](docs/kubeai-backend.md).
* What matches Compose, what does not yet, and what is deliberate:
  [docs/backend-parity.md](docs/backend-parity.md); the plan to close the
  rest: [docs/planning/backend-parity-roadmap.md](docs/planning/backend-parity-roadmap.md).

The short version:

```bash
infer-stack kube inventory
infer-stack kube doctor
# New local cluster / NVIDIA plugin + GFD prerequisites (review before applying):
infer-stack kube bootstrap --provider=k3s
sudo -v
infer-stack kube bootstrap --provider=k3s --apply
# Automatic profiles from Kubernetes node labels; no temporary values file needed:
infer-stack kube install
infer-stack kube install --apply
infer-stack kube doctor
infer-stack catalog suggest --backend kubeai
infer-stack config set backend kubeai
infer-stack config set kubeai_gateway cluster
infer-stack doctor --backend kubeai
infer-stack acquire <endpoint> --ttl 2h --env-file lease.env --yes
```

If the workstation already has a Compose ledger/configuration that you plan to
return to, keep that authority intact rather than switching backend kinds in
the same ledger. The cluster setup guide documents the temporary test pattern:
use `INFER_STACK_BACKEND=kubeai` with a separate KubeAI data root, and use
`infer-stack kube node detach/attach` to hand each physical GPU host between
Kubernetes scheduling and direct Compose use without uninstalling or rejoining
the node. See [docs/cluster-setup.md](docs/cluster-setup.md).

### Switching an existing recovery ledger to KubeAI

Stopping the Compose realization retains its recovery backend and historical
ledger. Backend kinds require separate recovery epochs. After releasing every
old lease and stopping the old runtime:

```bash
infer-stack release --all --backend compose
infer-stack stack down --backend compose
infer-stack config set backend kubeai
infer-stack status                     # configured kubeai / active recovery compose
infer-stack ledger rotate              # verify quiescence and preview the archive
infer-stack ledger rotate --yes        # archive old history; initialize KubeAI epoch
infer-stack ledger archives            # archive paths + history inspection commands
infer-stack acquire <endpoint> --yes
```

Rotation refuses active leases, remaining old runtime objects, and unknown
runtime state. It does not change configuration/catalogs or silently tear down
workloads. Old leases, deployments and the recovery profile remain in SQLite
archives under `<ledger-directory>/archives/`. Rotation is transactional and
safe to retry; a matching recovery backend makes the command a no-op. `gc`
reports backend mismatches with transition instructions; `gc --forget` remains
a history-only operation.

### Kubernetes cluster setup and KubeAI integration

[docs/cluster-setup.md](docs/cluster-setup.md) describes server/worker setup.
`kube inventory` and `kube status` report structured facts; `--json` includes
probe errors without discarding other discoveries. Inventory and catalog
suggestions work before the KubeAI chart or namespace exists.

`kube doctor` checks Kubernetes, NVIDIA scheduling/discovery, and KubeAI in
dependency order. Top-level `doctor --backend kubeai` remains the operational
preflight for acquiring work. If Helm installation succeeds but the configured
KubeAI API URL is unavailable, doctor reports that remaining routing issue.
Set `kubeai_base_url` to a reachable OpenAI URL for the installed service.

`kube bootstrap` defaults to a read-only plan; `--apply` (or `--yes`) authorizes
host/cluster changes. K3s is the implemented provisioning provider; inventory,
installation and the backend work with any Kubernetes distribution. Bootstrap
always provisions/reconciles the local K3s server, preserves unrelated selected
contexts and kubeconfigs, installs Helm if missing,
and reconciles NVIDIA device plugin **0.17.1** with GPU Feature Discovery.
Install the host NVIDIA driver and container toolkit first. K3s discovers the
installed runtime on startup; bootstrap restarts local K3s only when runtime
discovery needs repair. It waits for Ready nodes, GPU allocation and GFD labels, and verifies unknown
GPU-node runtime handlers with small node-specific canary pods. Root admin
credentials stay `0600`; a private `0600` copy lives at
`~/.kube/infer-stack-k3s.yaml`. Existing default configs are preserved. Select the
local server explicitly with `export KUBECONFIG=~/.kube/infer-stack-k3s.yaml`
before inventory/install; rerun bootstrap to refresh copied certificates.

`kube install` shows inferred Helm values; `--apply` uses Helm upgrade/install
and checks readiness afterward. `--dry-run`/`--plan` forces read-only behavior.
`--namespace`, `--release`, `--chart`, `--version` and `--values` allow overrides.
Existing named profiles and custom values are preserved; `HF_TOKEN` overrides
the chart token using a temporary protected file. Generated public values live
at `<data>/generated/kube/kubeai-values.yaml`.

`kube setup` is a deprecated alias of `kube install`; it no longer owns a
separate prerequisite workflow. `kube k3s bootstrap` provisions only the local
server, `join` verifies requested worker membership, and `status` reports local
membership independently of any stale selected admin context. Setup scripts are compatibility wrappers around the
package commands.

### Debugging serving failures

`infer-stack kube status` summarizes live Kubernetes state. `infer-stack acquire`
reports pod failures such as `ImagePullBackOff`, `Unschedulable`, and engine
crashes. Use `infer-stack ps` and `infer-stack logs -f <endpoint>` to inspect
model serving. If `/models` works but completions return 404, check that clients
use the gateway alias; direct KubeAI requests use the Model name from the env
file (`INFER_STACK_ENDPOINT_*`). Missing `libcuda.so.1` usually indicates that
the model profile did not request GPUs or select the correct runtime.

---

## Which backend should I start with?

**Compose** when one workstation is enough: it is the fastest path to a
working server, everything it renders is a file you can read, and it needs
only Docker.

**KubeAI** when the models must run on more than one machine, or a cluster
already exists. It is Compose plus a scheduler: the same catalog, verbs and
env file, with a cluster and `resourceProfiles` supplied. Expect more
first-request overhead (pod creation, image pull, model load).

A catalog written for Compose runs on KubeAI, with two exceptions: ollama
endpoints and custom container launches (`runtime.command` / `mounts`),
which stay Compose-only. [docs/backend-parity.md](docs/backend-parity.md)
has the full matrix.

## vLLM startup caches

Generated Compose mounts persist Hugging Face, vLLM, PyTorch/TorchInductor,
Triton, and CUDA JIT caches. Warm starts avoid redownloading and redoing many
compile/JIT steps, but a vLLM model swap still creates a new engine process and
must reload weights into GPU memory.

### Diagnosing readiness

`docker compose` health only means a container-level healthcheck passed. It
is not "the routed model answers a request through the front door", which is
what `acquire` and `wait` check: a model swap starts a new engine process,
and LiteLLM stays up while returning upstream connection errors until vLLM
has loaded the weights.

```bash
infer-stack acquire <endpoint> --yes     # waits for a real generation
infer-stack wait <endpoint>              # after acquire --no-wait
infer-stack test <endpoint>              # one generation through the gateway
infer-stack status                       # desired vs running, per deployment
infer-stack logs <service> --tail 80     # the engine or gateway log
```

A crash-looping engine fails the acquire at once with its error quoted, so
the timeout is only for a model that is loading. Reading Docker's own
signals: `litellm exited with code 137` is a SIGKILL (an OOM kill or a forced
replacement), whereas LiteLLM returning HTTP 500 with `Cannot connect to host
vllm-*` means LiteLLM is running and its upstream vLLM is not ready yet.

### Kubernetes dashboard

`infer-stack tui` provides a **Cluster** tab in the expanded runtime pane on the
KubeAI backend. It shows Ready/scheduling state, GPU allocation and scheduled
requests across namespaces, GFD labels, runtime evidence, managed Models and
KubeAI/NVIDIA pod startup/restart state. GPU requests are scheduling facts, not
live GPU utilization; the system pane explicitly describes the local host.

Cluster monitoring uses three batched API reads, cached for at least 15 seconds
and polled only while the tab is visible. Slow probes cannot overlap or block
keyboard input; **Refresh cluster** forces a sample, and **Doctor** runs the
shared detailed checks on demand. Pod samples also feed the Instances view.

Select a node for **Detach node** or **Attach node**. Each previews its workload
and scheduling state and requires confirmation; detach drains workloads and
emptyDir data, while attach confirms local Compose workloads have stopped.
These reuse `kube node detach/attach` and preserve cluster membership. The
Control tab provides **Apply** through the leasing controller and confirmed
**Down** for managed Models and the gateway. Leases remain after Down. The
KubeAI endpoint editor selects resource profiles; Kubernetes chooses node/GPU
placement.

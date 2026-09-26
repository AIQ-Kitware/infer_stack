# Infer Stack

[![PyPI version](https://img.shields.io/pypi/v/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![Python versions](https://img.shields.io/pypi/pyversions/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![License](https://img.shields.io/pypi/l/infer-stack.svg)](https://github.com/AIQ-Kitware/infer_stack/blob/main/LICENSE)

> **Heads up — the leasing model is now the primary workflow.** Declare models
> in a catalog (`infer-stack catalog …`) and `acquire`/`run` endpoints
> on demand; see `docs/source/manual/` (the Ollama + Open WebUI tutorial and the
> leasing demo) and `infer-stack help tree`. The
> **named stack profiles** documented below (`setup`/`render`/`up`/`switch`/…)
> are the pre-leasing model and now live under **`infer-stack legacy <command>`**
> (e.g. `infer-stack legacy render`). This README still describes that legacy
> flow; a leasing-oriented rewrite is pending.

## Primary leasing workflow

The normal user path has three steps and no separate publication phase:

```bash
infer-stack config init
infer-stack catalog suggest --apply
infer-stack acquire <endpoint>
```

`config.yaml` and `catalog.yaml` are the user configuration. Leasing keeps an
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

`infer_stack` manages **named stack profiles** for local and Kubernetes-backed inference.

A stack profile is a small graph made from:

* **providers** — inference runtimes such as vLLM and Ollama
* **gateways** — optional API routers such as LiteLLM
* **frontends** — optional UIs such as Open WebUI
* **routes** — optional public model aliases exposed through a gateway

This repo can render those profiles through two backends:

* **Compose** for local single-host serving. Compose supports vLLM, Ollama, optional LiteLLM, and optional Open WebUI.
* **KubeAI** for Kubernetes-backed vLLM serving. KubeAI support is vLLM-only for now.

The direct Ollama path can run without LiteLLM and without predeclaring models. vLLM profiles still use explicit runtimes, placement, and runtime settings.

## Main commands

```bash
infer-stack setup --backend compose --profile ollama-direct
# or: infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
infer-stack list-profiles
infer-stack describe-profile <profile>
infer-stack validate
infer-stack render
infer-stack up -d
infer-stack deploy
infer-stack switch <profile> --apply  # re-render and converge; no separate up needed
infer-stack status
infer-stack smoke-test
infer-stack version                   # print the installed version
infer-stack config paths              # show where config / artifacts / caches live
```

The CLI is built on [`kwconf`](https://github.com/Erotemic/kwconf),
so every subcommand is also importable as a Python class — useful for
notebooks, tests, and other scripts:

```python
from infer_stack.cli import RenderCLI, SmokeTestCLI

RenderCLI.main(argv=False, profile="qwen2-5-7b-instruct-turbo-default", yes=True)
SmokeTestCLI.main(argv=False, model="qwen/qwen2.5-7b-instruct-turbo")
```

`manage.py` and `infer-stack` are aliases for the same entry point;
shell examples below use `infer-stack`.

## Operating the rendered Compose stack

Once the stack is up, common docker compose operations are available as
`infer-stack` subcommands so you don't have to `cd` into the rendered
output directory or repeat the `-f docker-compose.yml --env-file .env`
flags. They all resolve the rendered location via the same
`output.generated_dir` chain as the rest of the CLI.

```bash
infer-stack ps                              # docker compose ps
infer-stack ps -a                           # include stopped
infer-stack logs -f open-webui              # follow one service
infer-stack logs --tail=200 litellm vllm-*  # tailored backlog
infer-stack logs -f --raw litellm            # full LiteLLM tracebacks
infer-stack restart open-webui              # restart specific services
infer-stack stop                            # stop everything (no remove)
infer-stack start                           # start back up
infer-stack pull                            # refresh images
```

Interactive ``infer-stack logs -f`` compacts only explicitly registered, known-noisy
LiteLLM traceback shapes; unknown tracebacks pass through unchanged. Redirected or
piped output stays raw, and ``--raw`` disables compaction in an interactive follow.
The compacted CLI path preserves Compose ANSI service colors when attached to a TTY;
``--no-color`` still disables them. The TUI uses the same conservative compactor when
LiteLLM logs are visible.

For Ollama model management inside the rendered Ollama service, prefer the
CLI wrappers:

```bash
infer-stack ollama-pull smollm2:135m
infer-stack ollama-list
infer-stack ollama-ps
```

For other interactive one-shot commands inside a container, use
`infer-stack logs`, `infer-stack ps`, `infer-stack restart`, or fall back to raw
Compose only when no wrapper exists.

On the KubeAI backend these wrappers raise ``NotImplementedError`` —
use the equivalent ``kubectl`` commands in the meantime.

## Inspect a profile before running it

```bash
infer-stack describe-profile qwen2-5-7b-instruct-turbo-default --format yaml
```

## Stack profile model

Profiles are written as stack graphs. The main sections are `providers`, `gateways`, `frontends`, and `routes`. For details and examples, see [docs/stack-graph-profiles.md](docs/stack-graph-profiles.md).

Common shapes:

```text
Open WebUI -> Ollama                         # ollama-direct, no LiteLLM
Open WebUI -> LiteLLM -> vLLM                # classic vLLM compose profiles
Open WebUI -> LiteLLM -> Ollama              # Ollama with stable aliases
Open WebUI -> LiteLLM -> Ollama + vLLM       # mixed migration / test stacks
Ollama API + vLLM API directly               # raw backend profiles
```

Custom provider models and custom profiles live in the configured `catalog.user_models_file`, which defaults to `~/.config/infer_stack/models.yaml`. New files should prefer provider-specific top-level keys:

```yaml
vllm_models:
  my-vllm-model:
    hf_model_id: org/model

ollama_models:
  my-ollama-model:
    tag: qwen3.5:4b

profiles:
  my-stack:
    providers: {}
    gateways: {}
    frontends: {}
    routes: {}
```

`models:` is still interpreted as a vLLM model catalog for convenience, but new docs and recipes use `vllm_models:` / `ollama_models:`.

## Where config and rendered artifacts live

`infer-stack` follows XDG basedir conventions, so where you invoke it
from never changes which config it reads or where it writes rendered
artifacts:

There are exactly two path roots:

| What | Default location | How to relocate |
| --- | --- | --- |
| `config.yaml`, `models.yaml`, `kubeai-values.local.yaml` | `~/.config/infer_stack/` (resp. `$XDG_CONFIG_HOME`) | `--config-dir` (or `INFER_STACK_CONFIG_DIR`) |
| **Everything generated** — `generated/` (docker-compose.yml, .env, plan.yaml, kubeai/*) **and** `state/` (hf-cache, postgres volumes, Ollama store, runtime bind mounts) | `~/.local/share/infer_stack/` (resp. `$XDG_DATA_HOME`) | `--data-dir` (or `INFER_STACK_DATA_DIR`) |

`--data-dir` is the single knob for "put everything I generate in one
directory." It **relocates the one infer-stack installation controlling a
host/backend; it does not create an isolated second installation**. Do not run
controllers from multiple config/data roots against the same Docker host or
Kubernetes namespace. See the
[single-owner limitation](docs/planning/known-limitations.md#one-control-plane-per-host-or-backend-namespace).

Set it once at `setup`; it is baked into the absolute
`state.*` and `output.generated_dir` paths written to `config.yaml`, so
later commands don't need it again:

```bash
# All rendered artifacts and bind-mount state land under one directory.
infer-stack setup \
  --backend compose \
  --profile ollama-direct \
  --data-dir /data/service/docker/vllm-stack

infer-stack render --yes
```

```bash
# Keep config.yaml in a checkout for ad-hoc experiments.
infer-stack setup --config-dir $PWD --backend compose --profile <p> --data-dir $PWD/stack
```

`--config-dir` / `--data-dir` live on every subcommand, so they appear
**after** the subcommand name. For "set once for the whole shell" use the
env vars instead. For a bespoke split layout (e.g. big `state/` on a data
disk, artifacts elsewhere), edit `state.*` / `output.generated_dir` in
`config.yaml` directly.

## Constraining placement to specific GPUs

If some of your GPUs are tied up by other work, restrict the planner
(and the rendered ``device_ids``) to the subset you want it to use:

```bash
# Only place onto GPU 1 (e.g. GPU 0 is running a display).
infer-stack render --yes --profile test-single-11gb --allowed-gpus 1

# Or pin a TP=2 profile to physical GPUs 1 and 3.
infer-stack render --yes --profile test-multi-gpu --allowed-gpus 1,3
```

``--allowed-gpus`` (or ``INFER_STACK_ALLOWED_GPUS=1,3``) filters the
detected inventory before placement — real indices are preserved, so
the rendered compose stack pins ``device_ids: ["1", "3"]`` to those
exact physical GPUs. Useful for integration tests that need to share a
host with other jobs.

## Demos / integration recipes

End-to-end examples under [docs/demos/](docs/demos/) are written as
markdown tutorials. The CI smoke test is runnable with pytest-codeblocks:

```bash
pytest --codeblocks docs/demos/ci_smoke_test.md
```

Each ``bash`` block is a self-contained shell snippet you can also
copy-paste into a terminal. See
[docs/demos/ci_smoke_test.md](docs/demos/ci_smoke_test.md) for the
``setup → describe → validate → render`` flow on the smallest test
profiles.

For a real running vLLM stack on a workstation, see
[docs/demos/quickstart.md](docs/demos/quickstart.md). For direct Ollama on a dual GTX 1080 Ti style host, see
[docs/demos/ollama_direct_quickstart.md](docs/demos/ollama_direct_quickstart.md). For a focused GPU-1 backend switch test, see [docs/demos/smollm2_gpu1_backend_switch.md](docs/demos/smollm2_gpu1_backend_switch.md).

User-supplied paths on the CLI (`--file`, `--from-file`,
`--resource-profiles-file`, `--output-dir`) still resolve against the
current working directory — they're meant to behave as typed.

---

## Backend 1: Compose

Use Compose for local single-host deployments. It can render direct Ollama stacks, vLLM stacks, mixed Ollama+vLLM stacks, and raw backend-only stacks.

### Getting started

Prerequisite: Docker and the `docker compose` plugin must be installed.

```bash
# Direct Ollama, no LiteLLM and no predeclared models.
infer-stack setup --backend compose --profile ollama-direct
infer-stack validate --simulate-hardware 2x11
infer-stack render --yes --simulate-hardware 2x11
infer-stack up -d

# Classic vLLM through LiteLLM/Open WebUI.
infer-stack setup --backend compose --profile qwen2-5-7b-instruct-turbo-default
infer-stack validate
infer-stack render
infer-stack up -d
```

### Test that it is responding

When LiteLLM is enabled, the default Compose front door is:

```text
http://127.0.0.1:14042/v1
```

When using `ollama-direct`, Open WebUI talks to Ollama directly and the Ollama API is available at:

```text
http://127.0.0.1:11434
http://127.0.0.1:11434/v1
```

unless you changed the relevant ports in config.

Wait until the active profile can serve a real request through its resolved default endpoint:

```bash
infer-stack wait-ready
```

`wait-ready` is stronger than Docker Compose health: it probes the user-facing
LiteLLM, Ollama, or direct vLLM access surface and, by default, requires a tiny
generation/completion to succeed. The smoke test runs this readiness probe by
default before issuing its normal test request:

```bash
infer-stack smoke-test
```

For direct Ollama profiles, pull a model first and then smoke-test that model:

```bash
infer-stack ollama-pull qwen3.5:4b
infer-stack ollama-list
infer-stack smoke-test --model qwen3.5:4b
```

For LiteLLM profiles, `smoke-test` reads the rendered `.env` automatically and
uses the active profile's resolved OpenAI-compatible front door. You can inspect
individual secrets when needed:

```bash
infer-stack env LITELLM_MASTER_KEY
infer-stack env VLLM_BACKEND_API_KEY
```

To replace the gateway's master key (refused while leases are active; the
gateway restarts, and clients must fetch the key again):

```bash
infer-stack secrets rotate
```

When you intentionally want the old quick behavior, skip the readiness wait:

```bash
infer-stack smoke-test --no-wait --model gpt2
```

### Stop it

```bash
infer-stack down
```

`down` never removes named volumes. The Postgres data directory and the
Open WebUI volume are preserved across `down`, `up`, `switch`, and `render`.

### Open WebUI authentication

By default Open WebUI runs with `WEBUI_AUTH=False` — no login screen,
anyone who can reach the port gets straight into the UI. This is the
expected behavior for a local dev box. To re-enable login/signup, set
in `config.yaml`:

```yaml
open_webui:
  auth: true
```

and re-render. Existing accounts stored in the `postgres-open-webui`
volume are preserved across the toggle.

### Reverse proxy (TLS) and LDAP

Open WebUI can be fronted by an opt-in nginx TLS reverse proxy, and its
login can be backed by an LDAP directory. Both are off by default and
configured as ordinary config fields. The built-in `openwebui-tls-ldap`
profile wires them together as a worked example (Ollama + Open WebUI
behind nginx, no public Open WebUI/Ollama ports); see
[`examples/openwebui-tls-ldap/`](examples/openwebui-tls-ldap/).

```bash
infer-stack setup --backend compose --profile openwebui-tls-ldap
```

**Reverse proxy.** Enable it under `frontends.reverse_proxy`. It renders
an nginx service plus a generated `state.runtime/nginx.conf`:

```yaml
frontends:
  reverse_proxy:
    enabled: true
    target: open_webui        # or litellm / ollama / a custom upstream
    server_name: host.example.com
    ssl:
      enabled: true
      certificate: ./certs/site.crt
      certificate_key: ./certs/site.key
      dhparam: ./dhparam.pem   # optional
```

When `ssl.enabled` is true, port 80 redirects to HTTPS (`force_https`)
and the cert/key/dhparam host paths are bind-mounted read-only. When
`ssl.enabled` is false, only HTTP is published (HTTPS publishing is
gated on TLS so you never get a `:443` mapping with nothing listening).

> **Path caveat.** Relative `certificate`/`certificate_key`/`dhparam`/
> `config_path` values are written verbatim into the generated
> `docker-compose.yml`, so Docker Compose resolves them **relative to the
> generated directory** (where the compose file lives), not your CWD.
> `infer-stack render` warns when a referenced cert or config file is not
> found. Use absolute paths if you want to avoid the ambiguity.

**LDAP.** Enable it under `frontends.open_webui.ldap`. The directory
settings render as Open WebUI `LDAP_*` environment variables, and
secrets/site-specific values are emitted as `.env` placeholders
(`LDAP_HOST`, `LDAP_PASSWD`, `LDAP_SEARCH_BASE`, …) so you can fill them
in after the first render without re-touching the compose YAML:

```yaml
frontends:
  open_webui:
    ldap:
      enabled: true
      env_defaults:
        LDAP_PORT: '636'
        LDAP_USE_TLS: 'true'
        LDAP_ATTRIBUTE_FOR_USERNAME: uid
```

**Manual escape hatches.** When the typed renderer is not enough, drop
to manual control without leaving infer-stack:

* `frontends.reverse_proxy.config_path` — mount an existing nginx config
  file instead of rendering one.
* `frontends.reverse_proxy.extra_config` — inject extra directives into
  the rendered HTTPS `server` block.
* Every rendered service (`ollama`, vLLM runtimes, `litellm`,
  `open_webui`, `reverse_proxy`) accepts generic overrides:
  `extra_env`, `env_file`, `extra_volumes`, `extra_hosts`, `labels`,
  `additional_ports`, and `gpus` (scalar `all`/count or a structured
  device-request list).

Field precedence (lowest to highest) is: top-level config section
(`reverse_proxy:` / `open_webui:` / `ollama:`) → the matching
`frontends.*` / `providers.*` / `gateways.*` section → the active
profile. Newer configs should prefer the `frontends.*` / `providers.*`
form shown above.

### Persistent state and database layout

Compose renders stateful services only when their components are enabled:

* `postgres-open-webui` — rendered only when Open WebUI is enabled. It stores chats, accounts, and settings in `state.postgres_open_webui`.
* `postgres-litellm` — rendered only when LiteLLM is enabled. It stores router state in `state.postgres_litellm`.
* `ollama` — rendered only when the Ollama provider is enabled. Its model store is `state.ollama`, mounted at `/root/.ollama`.
* vLLM runtimes mount `state.hf_cache` for Hugging Face weights and `state.vllm_cache` for compiled artifacts.

Each Postgres container has its own `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD`, sourced from component-specific `.env` keys. There is no shared Postgres instance and no `postgres-init` bootstrap service.

Open WebUI chat history is **not** tied to the model currently being served, so after a profile switch old chats may reference model IDs the current gateway no longer advertises — that is expected.

### Operational tips

Prefer scoping commands to specific services rather than relying on
container names. Use only the services rendered by the active profile:

```bash
# LiteLLM gateway profile
infer-stack logs -f litellm

# Direct Ollama profile
infer-stack logs -f ollama

# Ollama model store helpers
infer-stack ollama-list
infer-stack ollama-ps
```

You do not need to delete any volume during normal operation. If you
ever want a destructive reset, do it explicitly with
`docker compose down -v` against `generated/docker-compose.yml` — the
toolchain itself never does this.

### Custom .env values are preserved

`generated/.env` is rewritten non-destructively. Any `KEY=value` pair
you add manually (for example `VERBOSE=1`, `HF_HOME=/data/hf`, or any
key this program does not yet know about) is preserved across
`render`, `setup`, `switch`, `up`, and `deploy`. Comments and the order
of existing lines are preserved where practical.

### Switching profiles

```bash
infer-stack switch <profile> --apply
```

`switch --apply` re-renders from the updated `config.yaml`, then brings the
stack up convergently with `--remove-orphans` so a separate `infer-stack up` is
not needed. Components/runtimes that are no longer in the rendered compose file
are dropped. Compose preserves existing containers whose service definitions did
not change. For vLLM-to-vLLM profile switches, unchanged Open WebUI stays up;
LiteLLM is refreshed through its admin API when possible. The live refresh
path treats LiteLLM's "model not found in db" response for config-backed
models as non-fatal, so switching aliases can add the new route without
tearing LiteLLM down. That can temporarily leave stale config-backed aliases
in `/v1/models`; restart LiteLLM manually only when you want to clean those
up. If Compose already created or recreated LiteLLM while converging the new
stack, no extra router refresh is attempted because the new container has
already loaded the freshly rendered YAML. Profiles that do not render
LiteLLM, such as direct Ollama profiles, skip the router refresh path even if
an old `runtime/litellm_config.yaml` file remains from a previous profile.
Switches that change Open WebUI's provider wiring, such as
`Open WebUI -> LiteLLM` to `Open WebUI -> Ollama`, necessarily recreate
Open WebUI because its environment changes. Postgres volumes and provider
caches are left untouched. vLLM runtime containers are named after their
Compose service, for example `vllm-chat`, so `docker ps` and
`infer-stack logs vllm-chat` clearly identify them as vLLM containers.

### Protocol modes for base vs. instruct models

Profiles declare a `protocol_mode` (`chat` or `completions`) that the
served model must support. Models also declare which protocols they
support via `supported_protocols`. Validation runs before render and
fails with an actionable message if a profile asks for `chat` on a
completions-only model.

Practical guidance:

* Instruct/chat models (with a chat template) can use either, but
  default to `chat`.
* Base models like Pythia, Llama-2 base, Mistral-v0.1 base, and Falcon
  base do not define a chat template. Their HELM profiles use
  `protocol_mode: completions` and the `smoke-test` command will
  exercise `/v1/completions` for them.
* The rendered LiteLLM config uses `text-completion-openai/<served>`
  as the upstream provider for completions-only services. That means
  even chat-shaped requests sent through Open WebUI to a Pythia model
  get translated by LiteLLM into upstream `/v1/completions` calls — no
  second vLLM container is needed to support Open WebUI for a
  completions-only model.
* Open WebUI is still a chat UI, so prompt formatting matters.
  HELM/eval clients should call `/v1/completions` directly for exact
  prompt control rather than going through the chat frontend.

### Chat-shaped clients on top of completions models

Some clients (e.g. InspectAI / Inspect Evals stock MMLU tasks) only
speak `/v1/chat/completions` and cannot be reconfigured. For those
cases, profiles can opt into a LiteLLM-only adapter:

```yaml
chat_compat:
  enabled: true
  strategy: flat_messages
```

When set on a `protocol_mode: completions` service, the rendered
LiteLLM config keeps the `text-completion-openai/<served>` upstream
and adds LiteLLM's documented prompt-template fields
(`initial_prompt_value` / `roles` / `final_prompt_value`) so chat
messages get flattened into a plain prompt — no role labels, messages
joined by `\n` — before being forwarded to vLLM `/v1/completions`.

This is **not** a chat tune; the model is still a base model and
prompt formatting still matters for evaluation. Use it only when a
chat-shaped client cannot be changed. The vLLM container is not
restarted, no `--chat-template` is rendered, and the adapter takes
effect after a `litellm`-only restart:

```bash
infer-stack render
infer-stack restart litellm
```

The built-in `pythia-inspect-mmlu-compat` profile is a ready-made
example; see
[`recipies/compose_pythia_inspect_mmlu_compat.md`](recipies/compose_pythia_inspect_mmlu_compat.md).

### Images with their own launcher

Some images wrap vLLM in their own launcher and are configured through
environment variables rather than `vllm serve` flags. Describe that in the
endpoint's `runtime`; infer-stack has no model-specific code for it:

```yaml
runtime:
  image: ghcr.io/syv-ai/hyperqwen:sha-684e927
  max_model_len: 65536
  gpu_memory_utilization: 0.93
  command: [single]              # replaces `vllm serve MODEL <flags>`
  env:                           # container environment
    PORT: '{port}'
    SPEC: dflash2
    CTX: fast
    PREFIX_CACHE: 1
    MAX_LEN: '{max_model_len}'   # filled from the field above
    GPU_UTIL: '{gpu_memory_utilization}'
    EXTRA_ARGS: '--served-model-name={served_model_name}'
  mounts:                        # persisted under the runtime data dir
    /app/models: hyperqwen/qwen3.8-27b/models
    /cache: hyperqwen/qwen3.8-27b/cache
```

Switching that image to another context profile is a data edit, in the catalog
or the TUI's endpoint editor. HyperQwen's measured 3090 profiles are:

- fast/default: `max_model_len: 65536`, `SPEC: dflash2`, `CTX: fast`;
- long: `max_model_len: 150000`, `SPEC: mtp`, `CTX: long`;
- huge: `max_model_len: 245760`, `SPEC: dflash2`, `CTX: huge`.

On an RTX 3090, `catalog suggest` emits the latter two as `-long` and `-huge`
endpoint variants. The hardware check happens only while suggesting: the
resulting catalog contains ordinary explicit runtime data, so `apply`/`acquire`
never silently retunes an endpoint after the fact.

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

Models can declare reasoning support in the catalog:

```yaml
reasoning:
  enabled: true
  parser: qwen3
  expose_to_openwebui: true
```

Profiles can override or set the same field per service. When a
service has `reasoning.enabled: true` and a `parser`, the renderer
adds `--reasoning-parser <parser>` to that vLLM container's command
line — that flag alone enables reasoning extraction in the current
vLLM CLI. You do not need to repeat it by hand in `extra_args`.

Open WebUI sees reasoning content via two paths:

1. Inline `<think>...</think>` tags emitted by the model.
2. Structured `reasoning_content` fields when LiteLLM normalizes them.

The LiteLLM template keeps `merge_reasoning_content_in_choices: true`
on chat-mode entries so Open WebUI can display reasoning in the
streamed response. To test reasoning end-to-end:

```bash
# Non-streaming CLI smoke test:
infer-stack smoke-test \
  --model qwen3.6-35b-a3b \
  --prompt "Think step by step: 17*23"

# For streaming inspection, read the key with the CLI wrapper:
LITELLM_MASTER_KEY=$(infer-stack env LITELLM_MASTER_KEY)
curl -N http://127.0.0.1:14042/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","stream":true,
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

* Setup, settings and semantics: [docs/kubeai-backend.md](docs/kubeai-backend.md).
* What matches Compose, what does not yet, and what is deliberate:
  [docs/backend-parity.md](docs/backend-parity.md); the plan to close the
  rest: [docs/planning/backend-parity-roadmap.md](docs/planning/backend-parity-roadmap.md).

The short version:

```bash
./scripts/bootstrap_k3s.sh                         # a one-host cluster (k3s + helm)
./scripts/install_kubeai.sh kubeai-values.yaml     # the chart, with your resourceProfiles
kubectl -n kubeai port-forward svc/kubeai 8000:80 &
infer-stack config set backend kubeai
infer-stack doctor                                 # cluster -> CRD -> namespace -> gateway
infer-stack acquire <endpoint> --ttl 2h --env-file lease.env --yes
```

### KubeAI prerequisites

You need a Kubernetes cluster, `kubectl` and Helm. `scripts/bootstrap_k3s.sh`
installs k3s and helm on one host; the steps it runs, by hand:

```bash
curl -sfL https://get.k3s.io | sh -
# or pin
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='v1.34.3+k3s1' sh -
```

Make `kubectl` usable without `sudo`:

```bash
sudo mkdir -p /etc/rancher/k3s/config.yaml.d
printf 'write-kubeconfig-mode: "0644"\n' | \
  sudo tee /etc/rancher/k3s/config.yaml.d/10-kubeconfig-mode.yaml >/dev/null
sudo systemctl restart k3s
kubectl get nodes
```

Install Helm:

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/83a46119086589a593a62ca544982977a60318ca/scripts/get-helm-4
chmod 700 get_helm.sh
./get_helm.sh
helm version
```

### NVIDIA GPU support

Install the NVIDIA device plugin and GPU Feature Discovery so Kubernetes can expose GPU resources and labels:

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --version 0.17.1 \
  --namespace nvidia-device-plugin \
  --create-namespace \
  --set gfd.enabled=true \
  --set runtimeClassName=nvidia
```

Check that GPU support is working:

```bash
kubectl -n nvidia-device-plugin get pods
kubectl get node "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')" \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
kubectl get nodes --show-labels | tr ',' '\n' | grep 'nvidia.com/' || true
```

You want a non-empty `nvidia.com/gpu` count and `nvidia.com/*` labels such as product and memory.

### Resource profiles

A `resourceProfile` is what one "GPU unit" means on your cluster; the
catalog's `runtime.resource_profile` (or the `kubeai_resource_profile`
setting) names one, and infer-stack appends the GPU count. Include GPU
`requests`, GPU `limits` and `runtimeClassName: nvidia`: without them the
pod can land on the GPU node and still start without `libcuda.so.1`.

```bash
PRODUCT="$(kubectl get nodes -o jsonpath='{.items[0].metadata.labels.nvidia\.com/gpu\.product}')"

cat > kubeai-values.yaml <<EOF
resourceProfiles:
  nvidia-gpu:
    runtimeClassName: nvidia
    requests:
      nvidia.com/gpu: "1"
    limits:
      nvidia.com/gpu: "1"
    nodeSelector:
      nvidia.com/gpu.product: "${PRODUCT}"
EOF
./scripts/install_kubeai.sh kubeai-values.yaml kubeai
```

If a `kubeai` release already exists, reuse its namespace
(`helm list -A | grep kubeai`) and set `kubeai_namespace` to match.

### Debugging checks

`infer-stack acquire` reports pod-level failures itself (`ImagePullBackOff`,
`Unschedulable`, a crash with the engine's error quoted). `infer-stack ps` lists
the pods and `infer-stack logs -f <endpoint>` follows one, as on compose. For
anything else, with `NS` the namespace and `MODEL` the Model's name
(`kubectl -n $NS get models`):

```bash
kubectl -n "$NS" describe model "$MODEL"
kubectl -n "$NS" get pods -l model="$MODEL"
kubectl -n "$NS" logs -f -l model="$MODEL" -c server            # the engine
kubectl -n "$NS" logs -l model="$MODEL" -c server --previous     # after a restart
kubectl -n "$NS" logs deploy/kubeai --tail=200 -f                # the KubeAI controller
kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 40
```

Common bad states:

* `libcuda.so.1: cannot open shared object file`: the pod did not request a
  GPU; fix the resource profile (requests, limits, `runtimeClassName`).
* startup probe fails with `connection refused`: still pulling the image,
  loading the model or warming up; the acquire's wait reports which.
* `/models` works but completions 404: the request bypassed the gateway and
  used the alias; through the gateway the alias is the model name, directly
  against KubeAI the Model name is (`INFER_STACK_ENDPOINT_*` in the env file).

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

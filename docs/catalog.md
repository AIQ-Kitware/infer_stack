# Catalog and custom endpoints

[Documentation index](index.md) · [Quickstart](../README.md#quickstart)

The catalog declares what can run, independently of which endpoints currently
have leases. Start with [model discovery](model-profiles.md) to import a known
configuration rather than building one by hand.

## Catalog schema

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
It has no local serving lease: `infer-stack access` (or `run`) publishes its route on the
LiteLLM front door, and the client asks for the alias through the same base
URL as any other endpoint. Moving an alias between a managed runtime and an
external server changes neither the alias nor the workflow:

After declaring an external endpoint (e.g. `qwen-remote`), and using trusted
local configuration:

```bash
infer-stack env REMOTE_QWEN_KEY=sk-...          # the upstream's key, by name
infer-stack access qwen-remote --env-file qwen.env
source qwen.env                                 # OPENAI_BASE_URL, OPENAI_API_KEY, ...
```

`api_base` is resolved from inside the gateway's container or pod:
`localhost` there is the gateway itself, so a server on the same host is
reached through the Docker bridge address (e.g. `http://172.17.0.1:8000/v1`).
See [docs/planning/external-endpoints.md](planning/external-endpoints.md).

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
The [schema reference](../infer_stack/leasing/catalog.py) is the module docstring.

Shapes the stack renders:

```text
Open WebUI -> LiteLLM -> vLLM / Ollama     # the default
                     \-> an external OpenAI-compatible server (external:)
Open WebUI -> vLLM / Ollama                # config set litellm false
```

The named stack profiles of earlier releases (`setup`, `switch`,
`--profile`) are gone; see
[docs/stack-graph-profiles.md](stack-graph-profiles.md).

## Constraining placement to specific GPUs

```bash
# Only place onto GPU 1 (e.g. GPU 0 is running a display).
infer-stack acquire NAME --allowed-gpus 1

# Or confine a TP=2 endpoint to physical GPUs 1 and 3.
infer-stack acquire TP2_NAME --allowed-gpus 1,3
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
* `placement: {weight_floor: false}` disables the cached-weight size heuristic;
  use only for a runtime with justified offloading or different weight storage.
  The explicit VRAM floor still applies.

## Protocol modes for base vs. instruct models

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

## Images with their own launcher

Some images use their own launcher (including native engines without vLLM)
and are configured through environment variables rather than `vllm serve` flags.
Compose supports these OpenAI-compatible launchers under `engine: vllm` via
the endpoint's `runtime`:

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
endpoint editor. The [HyperQwen suggestion](model-profiles.md) is a worked
example: `catalog suggest` uses compute capability and VRAM to emit the serving
profiles appropriate to the detected hardware class (and exact name gates only
for measured exceptions). The hardware check happens only while suggesting;
the resulting catalog holds ordinary explicit runtime data, so `apply`/`acquire`
never retunes an endpoint after the fact.

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

## Reasoning / thinking models

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

Declare actual supported per-request effort levels with
`catalog endpoint add --reasoning-efforts` (for example `medium high` if the
model supports those values). This lets LiteLLM forward `reasoning_effort` for
custom aliases. Preserve the declaration when recreating an endpoint. For
HyperQwen and pi, see [reasoning-effort.md](reasoning-effort.md).

## Inspect an endpoint's serving identity

```bash
infer-stack catalog endpoint provenance NAME
```

The result is credential-free JSON describing configured model, image and
runtime identity. Unpinned artifacts are reported as uncertain. This describes
the launch specification, not attestation of the weights actually loaded.

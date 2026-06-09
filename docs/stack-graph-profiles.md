# Stack graph profiles

Profiles describe an LLM deployment as a small graph rather than as a single
vLLM-shaped service list.

The four graph sections are:

- `providers`: inference servers that can answer model requests, currently
  `vllm` and `ollama`.
- `gateways`: optional API routers/proxies, currently `litellm`.
- `frontends`: optional user-facing UIs, currently `open_webui`.
- `routes`: optional public model aliases exposed through a gateway.

This separation keeps the simple cases simple:

- Ollama can run as one direct daemon with no predeclared model list.
- vLLM can still run one or more explicit model runtimes.
- LiteLLM is only needed when you want a unified `/v1` model namespace.
- Open WebUI can point either at LiteLLM or directly at Ollama.

## Mental model

```text
providers  -> raw inference endpoints
routes     -> optional public model names
gateways   -> optional route surface, usually LiteLLM
frontends  -> optional UI, usually Open WebUI
```

A profile may have providers without routes. For example, `ollama-direct`
starts Ollama and Open WebUI directly, and you pull models with the Ollama CLI
or from Open WebUI. A mixed profile with both Ollama and vLLM usually enables
LiteLLM so clients have one stable OpenAI-compatible endpoint.

## Direct Ollama

`ollama-direct` starts one Ollama daemon and Open WebUI connected directly to
it. It does not render LiteLLM, does not render `postgres-litellm`, and does
not require model declarations.

```bash
infer-stack setup --backend compose --profile ollama-direct
infer-stack render --yes --simulate-hardware 2x11
infer-stack up -d

infer-stack ollama-pull qwen3.5:4b
```

Rendered shape:

```text
Open WebUI -> Ollama
```

The profile owns daemon settings such as `keep_alive`, `context_length`, and
`max_loaded_models`; the model store itself lives in `state.ollama` and is
mounted at `/root/.ollama`.

## Ollama through LiteLLM

Use this when you want a stable OpenAI-compatible route name that can later be
moved from Ollama to vLLM without changing clients.

```yaml
profiles:
  ollama-qwen3.5-4b-gateway:
    providers:
      ollama:
        enabled: true
        gpu_indices: [0, 1]
        keep_alive: 2m
        context_length: 4096
    gateways:
      litellm:
        enabled: true
    frontends:
      open_webui:
        enabled: true
        provider: litellm
    routes:
      home-assistant-local:
        provider: ollama
        model: qwen3.5:4b
```

LiteLLM renders that route as `ollama_chat/qwen3.5:4b` with
`api_base: http://ollama:11434`.

Rendered shape:

```text
Open WebUI -> LiteLLM -> Ollama
```

Routes may reference an entry in `ollama_models`, or a raw Ollama tag such as
`qwen3.5:4b` directly.

## vLLM through LiteLLM

vLLM runtimes are explicit because each runtime starts a model-serving process.

```yaml
vllm_models:
  smollm2-135m-instruct:
    hf_model_id: HuggingFaceTB/SmolLM2-135M-Instruct
    served_model_name: smollm2-135m
    supported_protocols: [chat]
    min_vram_gib_per_replica: 4
    preferred_gpu_count: 1
    defaults:
      max_model_len: 2048
      gpu_memory_utilization: 0.5

profiles:
  smollm-vllm-compose:
    providers:
      vllm:
        runtimes:
          chat:
            model: smollm2-135m-instruct
            placement:
              strategy: first_fit
              gpu_count: 1
    gateways:
      litellm:
        enabled: true
    frontends:
      open_webui:
        enabled: true
        provider: litellm
    routes:
      smollm2:
        provider: vllm
        runtime: chat
```

Rendered shape:

```text
Open WebUI -> LiteLLM -> vLLM runtime
```

## Mixed Ollama + vLLM

`mixed-ollama-smollm` demonstrates one shared Ollama daemon plus one vLLM
runtime behind LiteLLM.

```yaml
profiles:
  mixed-local:
    providers:
      ollama:
        enabled: true
        gpu_indices: [0, 1]
      vllm:
        runtimes:
          smollm:
            model: smollm2-135m-instruct
            placement:
              strategy: first_fit
              gpu_count: 1
    gateways:
      litellm:
        enabled: true
    frontends:
      open_webui:
        enabled: true
        provider: litellm
    routes:
      home-assistant-local:
        provider: ollama
        model: qwen3.5:4b
      smollm2-135m:
        provider: vllm
        runtime: smollm
```

Rendered shape:

```text
                 -> Ollama
Open WebUI -> LiteLLM
                 -> vLLM
```

Mixed routes need a gateway for one unified client namespace. Without LiteLLM,
Ollama and vLLM can still run as raw servers, but clients must address them
separately.

## Raw servers

`raw-ollama-vllm` starts backend servers without Open WebUI or LiteLLM. This is
useful for debugging direct provider endpoints.

```yaml
profiles:
  raw-ollama-vllm:
    providers:
      ollama:
        enabled: true
        publish_port: true
      vllm:
        runtimes:
          smollm:
            model: smollm2-135m-instruct
            publish_port: true
    gateways:
      litellm:
        enabled: false
    frontends:
      open_webui:
        enabled: false
    routes: {}
```

Rendered shape:

```text
Ollama API directly
vLLM API directly
```

## Backend support

Compose supports Ollama, vLLM, LiteLLM, and Open WebUI in valid combinations.
KubeAI currently supports only vLLM runtimes; profiles that enable Ollama,
LiteLLM, or Open WebUI are rejected for `--backend kubeai`.

## Configuration files

Custom provider models and stack profiles can come from the configured
`catalog.user_models_file`, which defaults to `~/.config/infer_stack/models.yaml`,
from additional paths in `catalog.model_path` / `catalog.model_paths`, or from
`INFER_STACK_MODEL_PATH` for per-shell overlays. Use provider-specific top-level
keys:

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

`models:` is still interpreted as a vLLM model catalog for convenience, but new
examples should use `vllm_models:` and `ollama_models:`.

`INFER_STACK_MODEL_PATH` behaves like a PATH variable: entries are separated by
`:` on POSIX systems and by `;` on Windows. Each entry can be a YAML file or a
directory. Directory entries are scanned non-recursively for `models.yaml` / `models.yml`,
`*.models.yaml` / `*.models.yml`, and `*.profiles.yaml` / `*.profiles.yml`. Later files override earlier files, so a
repo-local experiment overlay can temporarily override or extend the normal user
catalog without editing `config.yaml`:

```bash
export INFER_STACK_MODEL_PATH="${INFER_STACK_MODEL_PATH:+$INFER_STACK_MODEL_PATH:}$PWD/infer_stack_profiles"
infer-stack describe-profile my-experiment-profile
```

For persistent extra catalogs, put the same directory or file path in
`config.yaml`:

```yaml
catalog:
  model_path:
    - /srv/experiments/infer_stack_profiles
```

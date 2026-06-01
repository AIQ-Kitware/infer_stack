# Quickstart — direct Ollama on a dual GTX 1080 Ti host

This quickstart brings up the simplest local stack for Pascal-era GPUs:

```text
Open WebUI -> Ollama
```

There is no LiteLLM router and no predeclared model registry. Ollama manages
its own model store under `state.ollama`, and models are pulled with the
Ollama CLI or through Open WebUI.

## Prerequisites

- Docker with the NVIDIA container runtime.
- `nvidia-smi` works on the host and from GPU-enabled containers.
- `infer-stack` installed in your Python environment.

This path is useful for GTX 1080 Ti systems because Ollama supports Pascal
GPUs, while current official vLLM images require newer CUDA compute capability.

## 1. Set up the profile

```bash
infer-stack setup \
  --backend compose \
  --profile ollama-direct \
  --data-dir /data/service/docker/infer-stack
```

For a two-card 1080 Ti machine, the built-in profile pins Ollama to GPUs
`[0, 1]` and uses conservative daemon settings:

```yaml
keep_alive: 2m
context_length: 4096
num_parallel: 1
max_loaded_models: 1
max_queue: 8
```

## 2. Render and start

```bash
infer-stack render --yes --simulate-hardware 2x11
infer-stack up -d
```

Expected rendered services:

```text
postgres-open-webui
ollama
open-webui
```

Not expected in this profile:

```text
postgres-litellm
litellm
vllm-*
```

Open WebUI is available at <http://127.0.0.1:13000>. Ollama is published on
<http://127.0.0.1:11434> because the direct profile sets `publish_port: true`.

## 3. Pull a model

Start with a small model before trying larger ones:

```bash
infer-stack ollama-pull qwen3.5:4b
```

Other good initial candidates for 11 GiB cards:

```bash
infer-stack ollama-pull qwen3.5:2b
infer-stack ollama-pull qwen3.5:9b-q4_K_M
infer-stack ollama-pull gemma4:e2b
```

Run a direct API check through the CLI smoke test. It uses Ollama's native API
for direct-Ollama profiles and reads rendered runtime settings automatically:

```bash
infer-stack smoke-test --model qwen3.5:4b
```

## 4. Day-to-day operations

```bash
infer-stack ps
infer-stack logs -f ollama
infer-stack down
infer-stack up -d
```

Inspect loaded models and where they live:

```bash
infer-stack ollama-list
infer-stack ollama-ps
```

## 5. When to add LiteLLM

Stay direct when the machine is just a local Ollama server or Home Assistant
backend. Switch to an Ollama gateway profile when you want a stable OpenAI
`/v1` route name that can later move to vLLM:

```bash
infer-stack switch ollama-qwen3.5-4b-gateway --apply
```

That changes the shape to:

```text
Open WebUI -> LiteLLM -> Ollama
```

For mixed Ollama+vLLM routing, use a profile such as `mixed-ollama-smollm` as
a template.

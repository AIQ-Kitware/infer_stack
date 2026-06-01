# Quickstart — vLLM + LiteLLM on an RTX 3090 workstation

This guide brings up the traditional vLLM-backed stack on a single-GPU
workstation where GPU 0 may be reserved for the desktop and GPU 1 is free for
inference. It starts with the tiniest vLLM model to prove the plumbing works,
then switches to progressively larger vLLM profiles.

For the simpler Pascal/1080 Ti path that runs **Open WebUI -> Ollama** with no
LiteLLM and no predeclared model registry, use
[`ollama_direct_quickstart.md`](ollama_direct_quickstart.md).

Rendered shape for this guide:

```text
Open WebUI -> LiteLLM -> vLLM
```

Default ports for this profile family:

- LiteLLM: <http://127.0.0.1:14042/v1>
- Open WebUI: <http://127.0.0.1:13000>

After setup, all state and generated artifacts live under
`/data/service/docker/infer-stack/`:

```text
/data/service/docker/infer-stack/
  generated/            <- docker-compose.yml, .env, plan.yaml
  hf-cache/             <- downloaded Hugging Face weights
  vllm-cache/           <- compiled vLLM artifacts
  open-webui/           <- Open WebUI state
  postgres-open-webui/  <- Open WebUI database
  postgres-litellm/     <- LiteLLM database, only when LiteLLM is enabled
  runtime/              <- runtime bind-mount configs such as litellm_config.yaml

~/.config/infer_stack/
  config.yaml           <- active profile, backend, paths, ports
  models.yaml           <- optional custom vllm_models, ollama_models, profiles
```

## Prerequisites

- Docker with the NVIDIA container runtime (`nvidia-smi` visible inside containers).
- `infer-stack` installed in your Python environment.
- A Hugging Face token in your shell for gated models:

```bash
export HF_TOKEN=hf_...
```

The first profile in this guide, `gpt2-single`, is public and does not need
`HF_TOKEN`.

## 1. First-time setup with GPT-2

Start with `gpt2-single`: GPT-2 124M, ~250 MB download, no auth,
completions-only. The goal is to validate Docker, GPU placement, LiteLLM, and
Open WebUI before committing to a large model.

```bash
infer-stack setup \
  --backend compose \
  --profile gpt2-single \
  --data-dir /data/service/docker/infer-stack
```

Render and start:

```bash
infer-stack render --yes
infer-stack up -d
```

vLLM downloads GPT-2 and compiles a small graph. Expect roughly 30-60 seconds
on first start and a few seconds on warm restart.

Watch it come up:

```bash
infer-stack ps
infer-stack logs -f
```

Once all containers are healthy, smoke-test the API:

```bash
infer-stack smoke-test
```

Manual completion check:

```bash
curl -s http://127.0.0.1:14042/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(infer-stack env --key LITELLM_MASTER_KEY)" \
  -d '{
    "model": "gpt2",
    "prompt": "Once upon a time, ",
    "max_tokens": 32
  }' | python3 -m json.tool
```

Open WebUI is up at <http://127.0.0.1:13000>, but GPT-2 is a base model, so
switch to a chat-capable profile before using it interactively.

## 2. Switch to SmolLM2 135M, chat-capable

```bash
infer-stack switch smollm2-135m-single --apply
```

`switch --apply` updates `config.yaml`, re-renders the stack, removes orphaned
vLLM containers, and refreshes LiteLLM/Open WebUI in place. Cached weights stay
on disk.

Smoke test again:

```bash
infer-stack smoke-test
```

Then open <http://127.0.0.1:13000>. Open WebUI now has a chat-capable route.

## 3. Switch to a workstation-sized vLLM model

After the plumbing is proven, switch to a real workstation profile. The
`workstation-safe` profile uses first-fit placement so it avoids display GPUs
when the policy reserves them:

```bash
infer-stack switch workstation-safe --apply
```

First start downloads the model and warms the vLLM cache; subsequent restarts
reuse `hf-cache/` and `vllm-cache/`.

```bash
infer-stack smoke-test
```

If you want a profile that explicitly pins a runtime to a particular GPU, copy
one of the built-in profile definitions into `~/.config/infer_stack/models.yaml`
and change `providers.vllm.runtimes.<name>.placement.gpu_indices`.

## Day-to-day workflow

Stop without deleting state:

```bash
infer-stack down
```

Start after a reboot:

```bash
infer-stack up -d
```

Tail logs and inspect status:

```bash
infer-stack logs
infer-stack ps
```

Run the smoke test:

```bash
infer-stack smoke-test
```

Read the LiteLLM key from `.env`:

```bash
infer-stack env --key LITELLM_MASTER_KEY
```

## Troubleshooting

### Required host ports are already bound

`infer-stack up` does a pre-flight check on enabled component ports. For this
profile family, the usual ports are 14042 for LiteLLM and 13000 for Open WebUI.
Direct Ollama profiles may also publish 11434.

Find what is holding a port:

```bash
ss -tlnp 'sport = :14042'
sudo lsof -nP -iTCP:14042 -sTCP:LISTEN
docker ps --filter publish=14042
```

Common fixes:

- Stop the old stack with `infer-stack down`.
- Stop a stale container, for example `docker stop litellm && docker rm litellm`.
- Change ports with setup flags such as
  `infer-stack setup --litellm-port 14001 --open-webui-port 13001`, then render
  and start again.

### Smoke test errors

- Could not connect: give the stack more time; check `infer-stack ps`.
- Connection closed before a response: the router is up but the upstream model
  may still be loading; inspect `infer-stack logs vllm-*`.
- 401/403: the key in `.env` does not match the running LiteLLM container;
  restart with `infer-stack down && infer-stack up -d`.
- 503: vLLM is still loading; inspect `infer-stack logs vllm-*`.

### Wiping state

Delete databases, Open WebUI state, and runtime configs while keeping model
caches:

```bash
infer-stack purge --yes
```

Delete everything, including model caches:

```bash
infer-stack purge --yes --delete-cache
```

# SmolLM2 backend switch test on GPU 1

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../../README.md).

This is a copy/paste smoke path for a workstation where GPU 1 is available.
It uses only built-in profiles and `infer-stack` runtime commands.

The two profiles are:

- `smollm2-135m-vllm-gpu1`: Open WebUI -> LiteLLM -> vLLM on physical GPU 1.
- `smollm2-135m-ollama-gpu1`: Open WebUI -> Ollama on physical GPU 1.

Switching profiles is non-destructive for state. `infer-stack switch --apply`
re-renders and runs Compose with `--remove-orphans`, but it does not delete
state directories or model caches. Containers whose definitions are unchanged
stay up. In this specific vLLM-to-Ollama switch, Open WebUI is recreated because
its provider changes from LiteLLM to Ollama; LiteLLM and vLLM are removed as
orphans. You can switch back to the vLLM profile later.

`--apply` already runs the converge step. You only need a separate
`infer-stack up -d` after plain `switch` without `--apply`, or when you edited
rendered files/configuration by hand and want to reconcile them later.

`infer-stack wait-ready` is the command to use after a switch when you care
about the model being fully servable, not merely the Docker container being
started or healthcheck-passing. It polls the same access surface that clients
use and requires a tiny generation/completion to succeed.

## 1. vLLM on GPU 1

```bash
cd /home/joncrall/code/helm_audit/submodules/infer_stack

infer-stack switch smollm2-135m-vllm-gpu1 --apply --yes
infer-stack ps
infer-stack wait-ready --model smollm2-135m
infer-stack smoke-test --model smollm2-135m
```

Open WebUI should be available at:

```text
http://127.0.0.1:13000
```

In this profile, Open WebUI talks to LiteLLM, and LiteLLM routes the
`smollm2-135m` model alias to the vLLM runtime.

Useful checks:

```bash
infer-stack logs --tail=120 litellm vllm-chat
infer-stack env LITELLM_MASTER_KEY
```

## 2. Switch to direct Ollama on GPU 1

```bash
infer-stack switch smollm2-135m-ollama-gpu1 --apply --yes
infer-stack ps
```

Expected services now:

```text
postgres-open-webui
ollama
open-webui
```

LiteLLM and vLLM containers are removed as orphans, but their persistent state
and caches are not deleted.

Pull the comparable Ollama model through the CLI wrapper:

```bash
infer-stack ollama-pull smollm2:135m
infer-stack ollama-list
infer-stack wait-ready --model smollm2:135m
infer-stack smoke-test --model smollm2:135m
```

Open WebUI is still available at:

```text
http://127.0.0.1:13000
```

In this profile, Open WebUI talks directly to Ollama, so it should discover
`smollm2:135m` from Ollama after the pull.

Useful checks:

```bash
infer-stack ollama-ps
infer-stack logs --tail=120 ollama open-webui
```

## 3. Switch back to vLLM

```bash
infer-stack switch smollm2-135m-vllm-gpu1 --apply --yes
infer-stack wait-ready --model smollm2-135m
infer-stack smoke-test --model smollm2-135m
```

## 4. Stop without deleting state

```bash
infer-stack down
```

`down` does not remove volumes or state directories. Use `infer-stack purge`
only when you intentionally want to delete Docker-written state.

## Diagnosing readiness during switches

When switching vLLM profiles, the vLLM runtime process may be replaced even if
Open WebUI and LiteLLM stay up.  During that window LiteLLM can list the new
model alias but return upstream connection errors until the vLLM process is
actually serving requests.

Use:

```bash
infer-stack diagnose --model smollm2-135m --generation
infer-stack diagnose --logs --tail 80
```

The first command prints active profile, compose service state, LiteLLM route
state, and provider probes.  The second adds recent logs for LiteLLM, Open
WebUI, vLLM, and Ollama services.

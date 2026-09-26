# Compose recipe: Pythia 6.9B + Pythia 2.8B + Qwen3.6-35B-A3B on a 4x96GB host

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../README.md).

> Schema note: built-in profiles now resolve to stack-graph profiles. vLLM runtimes live under `providers.vllm.runtimes`, LiteLLM routes live under `routes`, and direct Ollama profiles can run without LiteLLM.

This recipe uses the built-in `pythia-qwen3.6-mixed-4x96` profile to serve
three models behind a single LiteLLM router on a machine with **4 x 96GB
GPUs**:

| Service       | Model                       | GPUs   | Protocol     |
|---------------|-----------------------------|--------|--------------|
| `qwen36-35b`  | `Qwen/Qwen3.6-35B-A3B`      | 0, 1   | chat (TP2, reasoning) |
| `pythia-69b`  | `EleutherAI/pythia-6.9b`    | 2      | completions  |
| `pythia-28b`  | `EleutherAI/pythia-2.8b-v0` | 3      | completions  |

The Qwen service has `--reasoning-parser qwen3` auto-rendered by the
profile (vLLM enables reasoning whenever a parser is set). The two Pythia services are routed
through LiteLLM as `text-completion-openai/...` so chat-shaped requests
from Open WebUI are translated into upstream `/v1/completions` calls.

---

## 1. Setup, render, run

```bash
cd /path/to/infer_stack
python manage.py setup --backend compose --profile pythia-qwen3.6-mixed-4x96
python manage.py render
python manage.py up -d
```

If you already have the stack running on a different profile:

```bash
python manage.py switch pythia-qwen3.6-mixed-4x96 --apply
```

`switch --apply` re-renders, brings the stack up convergently, and
recreates `litellm` + `open-webui` in place so the new alias list takes
effect. Postgres volumes are not touched.

### LiteLLM-only chat compatibility for the Pythia services

The two Pythia services in this profile carry
`chat_compat: { enabled: true, strategy: flat_messages }`. That makes
LiteLLM advertise `eleutherai/pythia-6.9b` and `eleutherai/pythia-2.8b-v0`
as chat-capable to clients that only speak `/v1/chat/completions`
(notably InspectAI / Inspect Evals stock MMLU tasks). LiteLLM flattens
the chat messages into a plain prompt — concatenated `content` joined
by `\n`, no role labels — and forwards a `/v1/completions` call to vLLM.

This is **LiteLLM-only**:

- vLLM is not restarted, no `--chat-template` is rendered.
- No second/duplicate Pythia container.
- Pythia stays `protocol_mode: completions`; HELM-style direct
  `/v1/completions` clients still get exact-prompt control.
- The Qwen3.6 entry is unchanged — it remains a normal chat route.

To roll out a LiteLLM config change (e.g. after editing chat-compat
fields) without disturbing any of the running vLLM containers:

```bash
python manage.py render

infer-stack restart litellm
```

`--no-deps` keeps `vllm-qwen36-35b`, `vllm-pythia-69b`, and
`vllm-pythia-28b` running. Confirm with:

```bash
infer-stack ps
```

Avoid `docker compose down`, `down -v`, or a profile-wide
`up --force-recreate`.

### Restarting only the Qwen container

If you change Qwen settings (for example, the tool-call parser) and
you do not want to disturb the Pythia containers — they may be in
active use — re-render and recreate only the Qwen vLLM service:

```bash
python manage.py render
infer-stack restart vllm-qwen36-35b
```

`--no-deps` prevents Compose from touching `postgres-litellm`,
`postgres-open-webui`, `litellm`, `open-webui`, or the Pythia vLLM
services. To confirm the rendered service name first:

```bash
infer-stack ps
```

Avoid `docker compose down`, `down -v`, or a profile-wide
`up --force-recreate` for this — they will disrupt the Pythia
containers you are trying to keep running.

### Tool calling

Per the `Qwen/Qwen3.6-35B-A3B` model card, the recommended vLLM flags
for this model are:

```text
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--language-model-only
```

(also: `vllm >= 0.19.0`, which the pinned image satisfies).

The mixed profile renders all four automatically from the Qwen
service's `reasoning`, `tool_calling`, and `extra_args` metadata, so
clients like Kilo Code that send `tool_choice: "auto"` work out of the
box.

If `qwen3_coder` parses tool calls poorly against your specific prompts,
`qwen3_xml` is a manual fallback worth testing — but it is not what the
model card recommends. To try it, change the `parser:` value in
`default-profiles.yaml` (or in a local stack profile under `models.yaml`) and rerun
the targeted Qwen restart above.

---

## 2. Set your Hugging Face token (Qwen3.6 is gated)

```bash
grep '^HF_TOKEN=' generated/.env
```

Edit `generated/.env` and set `HF_TOKEN=...`. Unknown `.env` keys
(yours and ours) are preserved across re-renders.

---

## 3. Verify the routes

```bash
eval "$(infer-stack env --export)"
curl -s http://127.0.0.1:14042/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
```

Expected aliases:

- `qwen3.6-35b-a3b`
- `eleutherai/pythia-6.9b`
- `eleutherai/pythia-2.8b-v0`

Smoke-test each (the smoke-test picks `/v1/chat/completions` or
`/v1/completions` based on the profile's protocol):

```bash
python manage.py smoke-test --model qwen3.6-35b-a3b
python manage.py smoke-test --model eleutherai/pythia-6.9b
python manage.py smoke-test --model eleutherai/pythia-2.8b-v0
```

---

## 4. Open WebUI

Open `http://127.0.0.1:13000`. All three model aliases will appear.
Reasoning streams from the Qwen3.6 service and is displayed inline
when chat streaming is enabled. The two Pythia models will respond,
but remember they are base/completions models — prompt formatting
matters and chat-style multi-turn use will not produce great results
without explicit prompt strategies.

For exact prompt control on the Pythia models, prefer direct
`/v1/completions` against LiteLLM:

```bash
curl -s http://127.0.0.1:14042/v1/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"eleutherai/pythia-6.9b","prompt":"The capital of France is","max_tokens":16}'
```

---

## Summary

- Three vLLM containers, one shared LiteLLM router, separate Postgres
  containers for Open WebUI and LiteLLM.
- Reasoning flags are emitted from profile metadata, not hand-written.
- Completions-only models are usable from chat clients via LiteLLM's
  `text-completion-openai` provider, with no extra vLLM containers.

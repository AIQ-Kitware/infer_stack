# Leasing demo — deploy a model behind Open WebUI (yardrat)

A walkthrough, not a test plan. Where `leasing-test-plan.md` pokes at edges,
this shows the **happy path a real user takes**: set up your config + storage
once, deploy a chat model as a standing service, talk to it from Open WebUI,
then switch models around — all through the leasing CLI.

Validated against **yardrat**: GPU 0 Quadro RTX 8000 (48 GiB, free), GPU 1
Quadro RTX 5000 (16 GiB, display-attached). Both Turing (sm_75): **fp16 only**,
so every vLLM endpoint pins `--dtype=half`.

Setup is now durable config, not shell exports: `config set` persists your
storage location and default backend, and `catalog` edits the model list — so
the per-command flags stay short and every block is copy-paste-independent.

> The point of infer-stack is that you switch freely between models and configs.
> "demo settings" and "real settings" are the same settings — we write to the
> real config + real storage and can re-point either at any time.

Run `infer-stack help tree` any time to see the whole command surface.

---

## 1. One-time setup

### 1a. Choose where docker-mounted state lives, and the default backend

The vLLM containers bind-mount a Hugging Face weight cache (a few GiB for these
SmolLM2 models; tens of GiB for a large model). That cache + Open WebUI history
live under your data dir. Persist it once — no shell export needed afterward —
and set Compose as the default backend so you never repeat `--backend`:

```bash
# It is a good idea to setup the data-dir beforehand with appropriate permissions
export INFER_STACK_DATA_DIR=/data/service/docker/infer-stack
mkdir -p "$INFER_STACK_DATA_DIR"
chown "$USER" "$INFER_STACK_DATA_DIR"

# Interactive: prompts for the data dir + default backend, shows them, confirms.
infer-stack config init
# ...or non-interactively (scripts / CI):
infer-stack config init --yes --data-dir "$HOME/infer-stack" --backend compose
infer-stack config show
```

(Point the data dir at a disk with room — `/data/infer-stack` etc. for large
models. You can also change either later: `infer-stack config set backend …`.)

### 1b. Build your catalog (your durable model list)

No YAML by hand — the `catalog` editor validates as it writes:

```bash
infer-stack catalog init
infer-stack catalog model add smol17b --source hf://HuggingFaceTB/SmolLM2-1.7B-Instruct
infer-stack catalog model add smol135 --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
# the main endpoint (Turing => --dtype=half), kept warm across releases:
infer-stack catalog endpoint add chat --engine vllm --model smol17b \
    --max-model-len 8192 --gpu-mem 0.4 --extra-args='--dtype=half' --reclaim keep-warm
# a small, fast one that frees its GPU on release:
infer-stack catalog endpoint add chat-fast --engine vllm --model smol135 \
    --max-model-len 4096 --gpu-mem 0.2 --extra-args='--dtype=half' --reclaim stop
infer-stack catalog show
```

> Gated model? Set the token once (stored in the managed `.env` that Compose
> auto-loads), no shell export: `infer-stack secret set HF_TOKEN=hf_…`.

---

## 2. Deploy the model as a standing service

`serve` is an infinite lease (no TTL) — "deploy this and keep it up".
`--require-generation` makes readiness honest (waits for a real token). The first
run downloads the weights (a few GiB), so allow time. No `--backend` — it comes
from your settings.

```bash
infer-stack serve chat --require-generation --timeout 1200
```

Watch it come up from another shell:

```bash
infer-stack leases          # one live group, one standing (manual) lease
infer-stack stack ps        # the vllm-… + litellm services  (alias: infer-stack ps)
infer-stack stack logs -f   # follow startup (Ctrl-C to stop) (alias: infer-stack logs -f)
```

---

## 3. Talk to it (the stable front door)

One base URL (`http://127.0.0.1:14042/v1`), and infer-stack owns the API key —
fetch it inline, never export it by hand:

```bash
KEY="$(infer-stack secret get LITELLM_MASTER_KEY)"
curl -s http://127.0.0.1:14042/v1/models \
  -H "Authorization: Bearer $KEY" | python -m json.tool
curl -s http://127.0.0.1:14042/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

You ask for the **endpoint alias** (`chat`), not the HF model id — the gateway
routes it. That indirection lets you swap the model behind an alias without
changing any client.

---

## 4. Put Open WebUI in front of it

The leasing stack serves the OpenAI-compatible gateway; Open WebUI is a separate
container pointed at it. Its chat history persists under your data dir, so it
survives restarts and model switches.

```bash
DATA="$(infer-stack config get data_dir)"
mkdir -p "$DATA/open-webui"
docker run -d --name open-webui --restart unless-stopped \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:14042/v1 \
  -e OPENAI_API_KEY="$(infer-stack secret get LITELLM_MASTER_KEY)" \
  -e WEBUI_AUTH=False \
  -v "$DATA/open-webui:/app/backend/data" \
  ghcr.io/open-webui/open-webui:main
echo "Open WebUI -> http://$(hostname):3000  (model: chat)"
```

Browse to `http://<yardrat>:3000`. `chat` is in the model picker.
(`WEBUI_AUTH=False` skips login for a single-user demo — don't expose that port
publicly. If you rotate the key, `docker rm -f open-webui` and re-run.)

---

## 5. Switch models around (the whole point)

The gateway stays put; what's behind it is yours to change. Add the small model
alongside the main one — Open WebUI's model list updates automatically:

```bash
infer-stack serve chat-fast --require-generation --timeout 600
infer-stack leases          # two live groups now; refresh Open WebUI's model list
```

Drop a standing service when you're done. A `serve` lease has no env-file, so
release it by its session id (copy it from `leases`):

```bash
infer-stack leases          # note the session id of the lease to drop
infer-stack release <SESSION_ID>
# chat is reclaim:keep-warm (stays resident); chat-fast is reclaim:stop
# (its container is torn down once no lease protects it).
```

To change the model itself: `infer-stack catalog model add smol17b --source
hf://other/Model --force` (or edit runtime via `catalog endpoint add chat …
--force`), release the old lease, and `serve chat` again. Same alias, new model —
clients and Open WebUI don't change.

---

## 6. Teardown

```bash
# stop Open WebUI
docker rm -f open-webui 2>/dev/null
# release all standing leases, then down the leasing stack
infer-stack leases --json | python -c 'import json,sys;[print(l["id"]) for l in json.load(sys.stdin)["leases"] if l["state"]=="active"]' | xargs -r -n1 infer-stack release
infer-stack stack down
# weights cache + chat history under your data dir are kept (re-serving is then
# fast). Delete the data dir to reclaim disk.
echo "done"
```

---

## Ergonomic notes

The earlier draft of this demo flagged five ergonomic smells; building the
`config`/`catalog`/`secret` submodals resolved the first cluster:

- ✅ **Default backend** — `infer-stack config set backend compose`; the leasing
  verbs default to it, so no more `--backend` on every call.
- ✅ **Storage in durable config** — `infer-stack config set data_dir <path>`;
  `data_root()` honors it (override > `$INFER_STACK_DATA_DIR` > setting > XDG), so
  no `~/.bashrc` export.
- ✅ **Managed `HF_TOKEN`** — `infer-stack secret set HF_TOKEN=…` writes the
  managed `.env` Compose auto-loads, set once before `serve`.

Still open (tracked in `dev/leasing-followups.md`):

- **No endpoint-addressed teardown for standing services.** Stopping a `serve`
  means copying a session id out of `infer-stack leases`. Want
  `infer-stack release --endpoint chat` / `unserve chat`.
- **Open WebUI is unmanaged** — a hand-run `docker run` (arguably correct
  separation, but costs the one-command UX). Maybe an `infer-stack ui up`.

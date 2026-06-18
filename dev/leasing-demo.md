# Leasing demo — deploy a real model behind Open WebUI (yardrat)

A walkthrough, not a test plan. Where `leasing-test-plan.md` pokes at edges,
this shows the **happy path a real user takes**: set up your config + storage
once, deploy a chat model as a standing service, talk to it from Open WebUI,
then switch models around — all through the leasing CLI.

Validated against **yardrat**: GPU 0 Quadro RTX 8000 (48 GiB, free), GPU 1
Quadro RTX 5000 (16 GiB, display-attached). Both Turing (sm_75): **fp16 only**,
so every vLLM endpoint pins `--dtype=half`.

Every block re-exports the one env var it needs (`INFER_STACK_DATA_DIR`) so you
can copy-paste any block on its own — nothing depends on a variable set in an
earlier block. The catalog and that one export are the *only* setup; after that
the commands are short.

> The point of infer-stack is that you switch freely between models and configs.
> So "demo settings" and "real settings" are the same settings — there's no
> throwaway scratch dir here. We write to the real config + the real storage
> location, and we can re-point either at any time.

---

## 1. One-time setup

### 1a. Choose where docker-mounted state lives

The vLLM containers bind-mount a Hugging Face weight cache (a few GiB for these
SmolLM2 models; tens of GiB if you later serve a large model). That cache (and
Open WebUI's chat history) lives under `INFER_STACK_DATA_DIR`. Pick a real,
persistent location — **not** `/tmp` — and make it the default for every future
shell:

```bash
# Point this at a disk with room. $HOME is fine for the demo; use something
# like /data/infer-stack if you'll serve larger models.
echo 'export INFER_STACK_DATA_DIR="$HOME/infer-stack"' >> ~/.bashrc
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
mkdir -p "$INFER_STACK_DATA_DIR"
echo "infer-stack state -> $INFER_STACK_DATA_DIR"
```

### 1b. Write your catalog (your "user config")

The catalog is the durable list of models/endpoints you serve. Put it at the
default config location and every `serve`/`acquire` finds it with no `--catalog`
flag. This is the file you edit over time as you add models.

```bash
mkdir -p ~/.config/infer_stack
cat > ~/.config/infer_stack/catalog.yaml <<'YAML'
# Your standing model catalog. Turing GPUs => --dtype=half on every vLLM model;
# SmolLM2 is ungated and tiny, so the demo runs end-to-end in minutes. Swap the
# `source` for a bigger model when you're ready — nothing else changes.
models:
  smol17b: {source: hf://HuggingFaceTB/SmolLM2-1.7B-Instruct}   # the main model
  smol135: {source: hf://HuggingFaceTB/SmolLM2-135M-Instruct}   # a fast/tiny one
endpoints:
  chat:
    engine: vllm
    model: smol17b
    runtime:
      max_model_len: 8192
      gpu_memory_utilization: 0.4
      extra_args: ['--dtype=half']
    reclaim: {policy: keep-warm}      # stay resident across releases
  chat-fast:
    engine: vllm
    model: smol135
    runtime:
      max_model_len: 4096
      gpu_memory_utilization: 0.2
      extra_args: ['--dtype=half']
    reclaim: {policy: stop}
YAML
echo "catalog -> ~/.config/infer_stack/catalog.yaml"
infer-stack paths        # sanity: shows config + leasing artifact locations
```

---

## 2. Deploy the model as a standing service

`serve` is an infinite lease (no TTL) — the right verb for "deploy this and keep
it up". `--require-generation` makes readiness honest (waits for a real token,
not just the model being listed). The first run downloads the weights from HF
(a few GiB for SmolLM2-1.7B), so give it a generous timeout.

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
infer-stack serve chat --backend compose --require-generation --timeout 1200
```

Watch it come up from another shell:

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
infer-stack leases          # one live group, one standing (manual) lease
infer-stack ps              # the vllm-… + litellm services
infer-stack logs -f         # follow startup (Ctrl-C to stop)
```

> Gated model instead? vLLM reads `HF_TOKEN` from the environment at container
> start, so `export HF_TOKEN=hf_…` in the shell *before* `serve`. (See the
> ergonomics notes — there's no managed slot for it yet.)

---

## 3. Talk to it (the stable front door)

There is one base URL (`http://127.0.0.1:14042/v1`) and infer-stack owns the API
key — fetch it inline, never export it by hand:

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
KEY="$(infer-stack secrets LITELLM_MASTER_KEY)"
curl -s http://127.0.0.1:14042/v1/models \
  -H "Authorization: Bearer $KEY" | python -m json.tool
curl -s http://127.0.0.1:14042/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

You ask for the **endpoint alias** (`chat`), not the HF model id — the
gateway routes it. That indirection is what lets you swap the model behind an
alias without changing any client.

---

## 4. Put Open WebUI in front of it

The leasing stack serves the OpenAI-compatible gateway; Open WebUI is a separate
container you point at it. Its chat history persists under your data dir, so it
survives restarts and model switches.

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
mkdir -p "$INFER_STACK_DATA_DIR/open-webui"
docker run -d --name open-webui --restart unless-stopped \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:14042/v1 \
  -e OPENAI_API_KEY="$(infer-stack secrets LITELLM_MASTER_KEY)" \
  -e WEBUI_AUTH=False \
  -v "$INFER_STACK_DATA_DIR/open-webui:/app/backend/data" \
  ghcr.io/open-webui/open-webui:main
echo "Open WebUI -> http://$(hostname):3000  (model: chat)"
```

Browse to `http://<yardrat>:3000`. `chat` is in the model picker. (`WEBUI_AUTH=False`
skips login for a single-user demo — don't expose that port publicly. The key is
passed at container start; if you rotate it, `docker rm -f open-webui` and re-run.)

---

## 5. Switch models around (the whole point)

The gateway stays put; what's behind it is yours to change. Add the small model
alongside the main one — Open WebUI's model list updates automatically (it reads
`/v1/models`):

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
infer-stack serve chat-fast --backend compose --require-generation --timeout 600
infer-stack leases          # two live groups now; refresh Open WebUI's model list
```

Drop a standing service when you're done with it. A `serve` lease has no
env-file, so release it by its session id (copy it from `leases`):

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
infer-stack leases          # note the session id of the lease you want gone
infer-stack release <SESSION_ID>
# chat is reclaim:keep-warm (stays resident); chat-fast is reclaim:stop
# (its container is torn down once no lease protects it).
```

To change the model itself, edit `~/.config/infer_stack/catalog.yaml`
(e.g. bump `max_model_len`, or point `smol17b` at a different `source`), release
the old lease, and `serve chat` again. Same alias, new model — clients and
Open WebUI don't change.

---

## 6. Teardown

```bash
export INFER_STACK_DATA_DIR="$HOME/infer-stack"
# stop Open WebUI
docker rm -f open-webui 2>/dev/null
# release all standing leases, then down the leasing stack
infer-stack leases --json | python -c 'import json,sys;[print(l["id"]) for l in json.load(sys.stdin)["leases"] if l["state"]=="active"]' | xargs -r -n1 infer-stack release
C="$INFER_STACK_DATA_DIR/leasing/compose"
docker compose -p infer-stack -f "$C/docker-compose.yml" down --remove-orphans 2>/dev/null
# weights cache + chat history under $INFER_STACK_DATA_DIR are kept (re-serving
# is then fast). Delete the dir to reclaim disk.
echo "done"
```

---

## Ergonomic smells found writing this demo

Where a step felt clunkier than it should, it's flagged here as a design
prompt — not necessarily a bug. (Cross-ref: this is the "if it isn't ergonomic,
that's a smell" check.)

1. **`--backend compose` on every command.** A user who always uses Compose
   repeats it constantly. The leasing verbs default to `--backend null`.
   *Proposal:* honor a persisted default (an `INFER_STACK_BACKEND` env var, or a
   `backend:` key in the catalog/user config) so a configured host can just say
   `infer-stack serve chat`.

2. **Storage location is env-only for the leasing path.** Legacy `setup` baked
   `state.*` paths into `config.yaml`; the leasing Compose backend ignores that
   and reads `data_root()` (env/XDG) directly. So "where my weights live" can
   only be set via `INFER_STACK_DATA_DIR` / `--data-dir`, hence the `~/.bashrc`
   line in §1a. *Proposal:* a first-class `infer-stack config set data_dir <p>`
   (or have leasing honor `config.yaml`'s storage root) so storage is part of
   the durable user config, not a shell export.

3. **No endpoint-addressed teardown for standing services.** `serve` has no
   env-file, so stopping it means copying a session id out of `infer-stack
   leases`. *Proposal:* `infer-stack release --endpoint chat` (or
   `infer-stack unserve chat`) to release standing leases by the name you
   served them under.

4. **Open WebUI is unmanaged.** The legacy stack rendered Open WebUI + its
   postgres; the leasing model dropped the UI, so it's a hand-run `docker run`
   here. That's arguably correct separation (leasing serves the API; the UI is a
   client), but it costs the one-command UX. *Proposal:* an optional
   `infer-stack ui up` (or a `ui` entry the catalog can render) that wires the
   base_url + managed key automatically.

5. **`HF_TOKEN` has no managed slot.** Gated models need it in the shell env at
   `serve` time (Compose interpolates `${HF_TOKEN:-}`); there's nowhere durable
   to put it next to the managed `LITELLM_MASTER_KEY`. *Proposal:*
   `infer-stack secrets --set HF_TOKEN=…` writing the managed `.env` (which
   Compose already auto-loads).

None of these block the demo; they're where the next ergonomic polish would pay
off. Smells 1–3 are the cheapest wins.

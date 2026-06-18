# Leasing demo — deploy a model behind Open WebUI (yardrat)

A walkthrough, not a test plan. Where `leasing-test-plan.md` pokes at edges,
this shows the **happy path a real user takes**: set up your config + storage
once, deploy a chat model as a standing service, talk to it from Open WebUI,
then run several models side by side — all through the leasing CLI.

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
# Omit the endpoint NAME and it defaults to the model, so what you serve and see
# in Open WebUI *is* the model (Turing => --dtype=half). The main one is kept warm:
infer-stack catalog endpoint add --model smol17b \
    --max-model-len 8192 --gpu-mem 0.4 --extra-args='--dtype=half' --reclaim keep-warm
# the small one frees its GPU on release:
infer-stack catalog endpoint add --model smol135 \
    --max-model-len 4096 --gpu-mem 0.2 --extra-args='--dtype=half' --reclaim stop
infer-stack catalog show
```

A defaulted name gets an auto-incrementing `-N` suffix, so this gives two
endpoints named after their models — **`smol17b-1`** and **`smol135-1`** — which
is what we use throughout. (Add another `--model smol17b` endpoint and it becomes
`smol17b-2`, never clobbering the first.) `infer-stack catalog show` lists them.

> Want a stable alias *decoupled* from the model (a `chat` you later re-point at
> whatever's current)? Name the endpoint explicitly instead:
> `infer-stack catalog endpoint add chat --model smol17b`. The demo stays
> model-centric so the name in Open WebUI tells you which model you're talking to.

> Gated model? Set the token once (stored in the managed `.env` that Compose
> auto-loads), no shell export: `infer-stack env HF_TOKEN=hf_…`.

---

## 2. Deploy the model as a standing service

`serve` is an infinite lease (no TTL) — "deploy this and keep it up".
`--require-generation` makes readiness honest (waits for a real token). The first
run downloads the weights (a few GiB), so allow time. No `--backend` — it comes
from your settings. A managed **Open WebUI comes up alongside the gateway by
default** (disable with `--no-ui`, or globally `infer-stack config set ui false`).

On a terminal, `serve`/`acquire` **show you the diff** of the compose project
they're about to change and ask you to confirm (so you see exactly which
services are added/removed); `--yes` (or `-y`) skips the prompt, and it's skipped
automatically when output isn't a terminal (scripts/CI). infer-stack also
narrates what it's doing (placement, `docker compose up`, readiness) on stderr.

```bash
infer-stack serve smol17b-1 --require-generation --timeout 1200
# shows the compose diff, asks to apply (or pass --yes), then:
#   open webui: http://127.0.0.1:13000
```

Watch it come up from another shell:

```bash
infer-stack leases          # one live group, one standing (manual) lease
infer-stack stack ps        # vllm-… + litellm + open-webui  (alias: infer-stack ps)
infer-stack stack logs -f   # follow startup (Ctrl-C to stop) (alias: infer-stack logs -f)
```

---

## 3. Talk to it (the stable front door)

The quickest check is the built-in smoke test — it sends a real generation to
the endpoint through the gateway and prints latency + the reply:

```bash
infer-stack test smol17b-1
# smol17b-1: ok (0.42s) 'ready'
```

Under the hood that is one HTTP call to a single base URL
(`http://127.0.0.1:14042/v1`), with infer-stack owning the API key. You can of
course do it by hand — fetch the key inline, never export it:

```bash
KEY="$(infer-stack env LITELLM_MASTER_KEY)"
curl -s http://127.0.0.1:14042/v1/models \
  -H "Authorization: Bearer $KEY" | python -m json.tool
curl -s http://127.0.0.1:14042/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"smol17b-1","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

You ask for the **endpoint name** (`smol17b-1`, your catalog's short name for the
model), not the full HF id — the gateway routes it.

---

## 4. Open WebUI (managed, already running)

Because `serve` brought Open WebUI up with the stack, just browse to it:

```bash
echo "Open WebUI -> http://$(hostname):13000  (model: smol17b-1)"
```

`smol17b-1` is in the model picker. Its chat history persists under your data dir
(`$(infer-stack config get data_dir)/open-webui`), so it survives restarts and
model switches — and crucially the UI container is **not** recreated when you
add/remove/switch models (only the gateway is), so it never blinks. `WEBUI_AUTH`
is off for a single-user workstation — don't expose port 13000 publicly.

> Don't want it? `infer-stack serve smol17b-1 --no-ui` (once) or
> `infer-stack config set ui false` (always). Turning it back on re-renders it
> on the next `serve`/`acquire`.

---

## 5. Run several models at once (the whole point)

The gateway stays put; add or drop models freely and Open WebUI's picker
follows. Bring the small model up alongside the big one — each is addressable by
its own name:

```bash
infer-stack serve smol135-1 --require-generation --timeout 600
infer-stack leases            # two live groups now
infer-stack test smol135-1    # confirm the new model serves
```

Both models now show in Open WebUI by name (`smol17b-1`, `smol135-1`); the UI
container itself is untouched by the change.

Drop a standing service when you're done. A `serve` lease has no env-file, so
release it by its session id (copy it from `leases`):

```bash
infer-stack leases          # note the session id of the lease to drop
infer-stack release <SESSION_ID>
# smol17b-1 is reclaim:keep-warm (stays resident); smol135-1 is reclaim:stop
# (its container is torn down once no lease protects it).
```

A `keep-warm` model stays resident after release (no cold-start next time) — but
it holds its GPU. To free that GPU now, **evict** it (overrides keep-warm):

```bash
infer-stack evict smol17b-1  # tear down the idle `smol17b-1` group now (by name)
infer-stack evict --all      # ...or every idle/released model at once
# or do it in one step at release time:
infer-stack release <SESSION_ID> --evict
```

> Prefer one **stable name that outlives the model behind it** (a `chat` you
> re-point)? Use an explicit endpoint name and swap its `--model` with `--force`:
> `infer-stack catalog endpoint add chat --model smol17b`, later
> `infer-stack catalog endpoint add chat --model smol17b-v2 --force`; clients and
> Open WebUI keep asking for `chat` while the model behind it changes.

---

## 6. Teardown

```bash
# release every active lease in one shot, then down the leasing stack (Open
# WebUI, the gateway, and the model containers all come down together)
infer-stack release --all
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
- ✅ **Managed `HF_TOKEN`** — `infer-stack env HF_TOKEN=…` writes the managed
  `.env` Compose auto-loads, set once before `serve`.
- ✅ **Managed Open WebUI** — bundled into the leasing stack and on by default
  (`--no-ui` / `config set ui false` to opt out). Stable across model switches
  (the UI container isn't recreated when routing changes), so it never blinks.
- ✅ **Concise smoke test** — `infer-stack test smol17b-1` instead of
  hand-rolled `curl`.

Still open (tracked in `dev/leasing-followups.md`):

- **No endpoint-addressed teardown for standing services.** Stopping a `serve`
  means copying a session id out of `infer-stack leases`. Want
  `infer-stack release --endpoint smol17b-1` / `unserve smol17b-1`. (`evict
  smol17b-1` already tears the deployment down, but doesn't release the lease.)

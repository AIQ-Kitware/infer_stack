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
small SmolLM2 / Qwen models; tens of GiB for a large model). That cache + Open
WebUI history live under your data dir. Persist it once — no shell export needed afterward —
and set Compose as the default backend so you never repeat `--backend`:

```bash
# It is a good idea to setup the data-dir beforehand with appropriate permissions
export INFER_STACK_DATA_DIR=/data/service/docker/infer-stack
mkdir -p "$INFER_STACK_DATA_DIR"
chown "$USER" "$INFER_STACK_DATA_DIR"

# Interactive: prompts for each setting (data dir, backend, Open WebUI,
# display-GPU skipping), shows them, confirms. Re-run to edit; --fresh to reset.
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
# Two from the SmolLM2 family...
infer-stack catalog model add smol17b --source hf://HuggingFaceTB/SmolLM2-1.7B-Instruct
infer-stack catalog model add smol135 --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
# ...and two from the Qwen family, so the catalog spans more than one model
# lineage and you can run them side by side (see §5):
infer-stack catalog model add qwen15 --source hf://Qwen/Qwen2.5-1.5B-Instruct
infer-stack catalog model add qwen05 --source hf://Qwen/Qwen2.5-0.5B-Instruct
# Omit the endpoint NAME and it defaults to the model, so what you serve and see
# in Open WebUI *is* the model (Turing => --dtype=half). The main one is kept warm:
infer-stack catalog endpoint add --model smol17b \
    --max-model-len 8192 --gpu-mem 0.4 --extra-args='--dtype=half' --reclaim keep-warm
# the small one frees its GPU on release:
infer-stack catalog endpoint add --model smol135 \
    --max-model-len 4096 --gpu-mem 0.2 --extra-args='--dtype=half' --reclaim stop
# the two Qwen endpoints, both reclaim:stop (they free their GPU on release):
infer-stack catalog endpoint add --model qwen15 \
    --max-model-len 8192 --gpu-mem 0.4 --extra-args='--dtype=half' --reclaim stop
infer-stack catalog endpoint add --model qwen05 \
    --max-model-len 4096 --gpu-mem 0.2 --extra-args='--dtype=half' --reclaim stop
infer-stack catalog show
```

A defaulted name gets an auto-incrementing `-N` suffix, so this gives four
endpoints named after their models — **`smol17b-1`** / **`smol135-1`** (SmolLM2)
and **`qwen15-1`** / **`qwen05-1`** (Qwen) — which is what we use throughout. (Add
another `--model smol17b` endpoint and it becomes `smol17b-2`, never clobbering
the first.) `infer-stack catalog show` lists them.

**Or name endpoints explicitly.** Give a NAME positional when you want a stable
alias *decoupled* from the model — e.g. a `chat` you later re-point at a
different model, so clients and Open WebUI keep asking for `chat` while the model
behind it changes (see §5). The NAME is just the first argument:

```bash
# explicit name `chat` -> smol17b (everything after NAME is the same flags)
infer-stack catalog endpoint add chat --model smol17b \
    --max-model-len 8192 --gpu-mem 0.4 --extra-args='--dtype=half' --reclaim keep-warm
# a couple more, to show the pattern:
infer-stack catalog endpoint add coder    --model smol17b --extra-args='--dtype=half'
infer-stack catalog endpoint add fast-chat --model smol135 --extra-args='--dtype=half' --reclaim stop
infer-stack catalog show
```

Mix freely: model-centric names (`smol17b-1`) tell you *which model* in Open
WebUI; explicit names (`chat`) give you a stable handle that outlives the model.
The rest of this demo uses the model-centric ones.

> Gated model? Set the token once (stored in the managed `.env` that Compose
> auto-loads), no shell export: `infer-stack env HF_TOKEN=hf_…`.

---

## 2. Deploy the model as a standing service

`serve` is an infinite lease (no TTL) — "deploy this and keep it up".
`--require-generation` makes readiness honest (waits for a real token). The first
run downloads the weights (a few GiB), so allow time. No `--backend` — it comes
from your settings. A managed **Open WebUI comes up alongside the gateway by
default** (disable with `--no-ui`, or globally `infer-stack config set ui false`).

On a terminal, **every verb that changes the compose project** — `serve`,
`acquire`, `release`, `evict`, `apply` — **shows you the diff** and asks you to
confirm before it touches `docker-compose.yml` or runs docker (so you see exactly
which services are added/removed/recreated); `--yes` (or `-y`) skips the prompt,
and it's skipped automatically when output isn't a terminal (scripts/CI).
Declining a `serve`/`acquire` rolls the lease back; declining a `release`/`evict`
records the change but leaves docker as-is until you `infer-stack apply`.
infer-stack also narrates what it's doing (placement, `docker compose up`,
readiness) on stderr.

```bash
infer-stack serve smol17b-1 --require-generation --timeout 1200
# shows the compose diff, asks to apply (or pass --yes), then:
#   open webui: http://127.0.0.1:13000
```

**Prefer to look before you leap?** `serve` does two things — *render* the
on-disk compose project, then *apply* it (`docker compose up`). They're separate
verbs: `serve … --no-apply` stages (declares the lease + writes the compose file
+ computes GPU placement, but starts nothing), and `infer-stack apply` is the
trigger. So you can read exactly what would run before committing:

```bash
infer-stack serve smol17b-1 --no-apply      # stage: writes the project, no `up`
#   smol17b-1: GPU 0  (grp-…)
#   compose: …/leasing/compose/docker-compose.yml
cat "$(infer-stack config get data_dir)"/leasing/compose/docker-compose.yml  # inspect
infer-stack apply                           # bring the staged set up (the trigger)
infer-stack wait smol17b-1 --require-generation   # then block until ready
# (change your mind instead? `infer-stack release --all` discards the staged lease.)
```

The compose file carries `name: infer-stack`, so if you'd rather drop the tool
entirely, `docker compose -f <that file> up -d` brings up the **same** project —
identical container names and network to what `infer-stack apply` would do.
`infer-stack render` (lease-free) just re-writes that file for whatever's
currently declared, without starting anything.

Watch it come up from another shell:

```bash
infer-stack leases          # leases + deployments: state (desired) vs running, and GPUs
infer-stack stack ps        # vllm-… + litellm + open-webui  (alias: infer-stack ps)
infer-stack stack logs -f   # follow startup (Ctrl-C to stop) (alias: infer-stack logs -f)
```

`leases` now shows the **deployments** table with both what the ledger *wants*
(`state`) and what's *actually up* (`running`), plus the GPUs each model is on
(or `→N` for one that's slated but not started yet) — so it's obvious when
something is still warming up vs truly live.

> **Prefer a live dashboard?** `infer-stack tui` opens a multi-pane Textual UI
> (run `infer-stack config init` once first):
> a **catalog** pane to pick an endpoint and serve it (`s`/Enter), live
> **leases** + **deployments** tables, and three collapsible panes below: **docker**
> (a `logs -f` tail + a **Containers** `docker ps` view: status/uptime · created
> · container id · ports), **system** (`nvidia-smi` GPUs + host CPU/mem), and
> **api** (send a prompt to a *ready* model — only running models are listed).
> Each pane carries its own description and buttons; expanding a pane (or its
> tab) is what triggers its polling, so hidden data is never fetched.
> New here? The catalog buttons (or `g`) **Suggest** a set sized to your GPUs,
> and `m` / `n` open wizards to add a model / endpoint by hand. **Clean up**
> (`x`) forgets released/stopped entries. **Ctrl+click** a served endpoint
> (or `o`) opens it in Open WebUI. Resize panes by dragging the splitter bars
> (or `[` `]` / `-` `+`). Opt-in extra: `pip install "infer-stack[tui]"`.

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

> **One port instead of two?** Turn on the reverse proxy and you hit a single
> origin — the UI at `/` and the API at `/v1` — so there's nothing to remember:
> ```bash
> infer-stack serve smol17b-1 --reverse-proxy        # or: config set reverse_proxy true
> # then:  http://localhost/        -> Open WebUI
> #        http://localhost/v1      -> the OpenAI API  (test still hits :14042 directly)
> ```
> It's plain HTTP — no TLS or auth — so it's for localhost / a trusted network,
> not public exposure. Set the port or mount your own `nginx.conf` via the
> `reverse_proxy` setting (`config edit`): `{enabled: true, port: 8080,
> config_path: /my/nginx.conf}`.

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

> **One model per GPU.** Placement is whole-GPU: each model lands on its own GPU,
> and `--gpu-mem` is only vLLM's reservation *within* that GPU, not a knob to pack
> two models onto one. So the first model takes GPU 0 and the second takes GPU 1
> — including yardrat's display-attached GPU 1, since placement uses **every** GPU
> by default (so single-GPU hosts work too). Want to keep the monitor's GPU free?
> `infer-stack config set skip_display_gpus true` (or `--skip-display-gpus` per
> command) — then a second model has nowhere to go and `serve` fails fast.
> `infer-stack leases` shows which GPU each model is on.

```bash
infer-stack serve smol135-1 --require-generation --timeout 600
infer-stack leases            # two live deployments now — GPU 0 and GPU 1
infer-stack test smol135-1    # confirm the new model serves
```

Both models now show in Open WebUI by name (`smol17b-1`, `smol135-1`); the UI
container itself is untouched by the change.

**Bring several up in parallel — across families.** `serve` blocks until ready
by default; pass `--no-wait` to kick a deployment off and return immediately,
then `wait` for them together (so models load at once instead of back-to-back).
Here we fan out a SmolLM2 and a Qwen endpoint so both lineages serve side by side:

```bash
# two models -> two GPUs (GPU 1 is the display GPU, used by default; see note above)
infer-stack serve smol17b-1 --no-wait --yes   # SmolLM2 -> GPU 0
infer-stack serve qwen15-1  --no-wait --yes   # Qwen -> GPU 1
infer-stack wait smol17b-1 qwen15-1 --require-generation --timeout 1200
# (bare `infer-stack wait` waits for every live model)
infer-stack test qwen15-1                       # confirm the Qwen endpoint serves
```

Open WebUI's picker now lists both `smol17b-1` and `qwen15-1`, so you can A/B the
two families from the same UI. `infer-stack leases` shows each on its own GPU
(GPU 0 and GPU 1); the 16 GiB display GPU holds a small model like `qwen15-1`
comfortably at `--gpu-mem 0.4`.

> `--require-generation` is the readiness *criterion* (ready == a real generated
> token, not just a listed model); `--no-wait` / `wait` are the *blocking*
> control. They're separate knobs: `serve` already waits unless you say
> `--no-wait`, and `wait` re-blocks later.

Drop a standing service when you're done. A `serve` lease has no env-file, so
release it by its lease id (copy it from `leases`):

```bash
infer-stack leases          # note the lease id of the lease to drop
infer-stack release <SESSION_ID>
# smol17b-1 is reclaim:keep-warm (stays resident); smol135-1 is reclaim:stop
# (its container is torn down once no lease protects it).
```

A `keep-warm` model stays resident after release (no cold-start next time) — but
it holds its GPU. To free that GPU now, **evict** it (overrides keep-warm):

```bash
infer-stack evict smol17b-1  # tear down the idle `smol17b-1` deployment now (by name)
infer-stack evict --all      # ...or every idle/released model at once
# or do it in one step at release time:
infer-stack release <SESSION_ID> --evict
```

`release`/`evict` only tear down **models** (freeing GPUs). The front door — the
gateway and Open WebUI — stays up even when zero models are served, so the UI
never blinks and reconnects as you serve again. To take the *whole* stack
(gateway + UI included) down, use `infer-stack stack down` (see §6).

> The explicitly-named `chat` endpoint from §1b is the **stable handle** case:
> re-point it at a different model in place and clients/Open WebUI never change
> what they ask for —
> `infer-stack catalog endpoint add chat --model smol135 --force`, then
> `infer-stack serve chat`. Same `chat` alias, new model behind it.

---

## 6. Teardown

```bash
# release frees the models/GPUs but leaves the front door (gateway + Open WebUI)
# up; `stack down` is what takes the whole project — gateway, UI, everything —
# down for a clean slate.
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
  *and* down to zero models — the gateway + UI are a standing front door that
  `release`/`evict` leave running (only `stack down` removes them), so it never
  blinks and reconnects as you serve again.
- ✅ **Concise smoke test** — `infer-stack test smol17b-1` instead of
  hand-rolled `curl`.

Still open (tracked in `dev/leasing-followups.md`):

- **No endpoint-addressed teardown for standing services.** Stopping a `serve`
  means copying a lease id out of `infer-stack leases`. Want
  `infer-stack release --endpoint smol17b-1` / `unserve smol17b-1`. (`evict
  smol17b-1` already tears the deployment down, but doesn't release the lease.)



### JON TEST

infer-stack serve qwen05-1

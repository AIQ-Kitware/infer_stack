# Tutorial: a self-managing Ollama + Open WebUI box

Goal: stand up a plain **Ollama** server with **Open WebUI** in front of it,
driven entirely by the `infer-stack` CLI — a drop-in replacement for a
hand-rolled `ollama` + `open-webui` docker-compose. Ollama manages its own
models: you pull/run/delete them from the Open WebUI model panel and the daemon
loads them on demand. No per-model catalog entries required beyond one anchor.

Worked example: a 2× GTX 1080 Ti box (Pascal, 11 GiB each). Ollama is the right
engine here — vLLM's bf16/FP8 paths are awkward on pre-Ampere cards, ollama
doesn't care.

---

## TL;DR

```bash
# 1. one-time setup
infer-stack config init                 # storage + pick backend = compose
infer-stack config set litellm false    # lean: Open WebUI talks straight to ollama

# 2. declare the daemon (pinned to both 1080 Tis) + one anchor model
infer-stack catalog init
infer-stack catalog host add local-ollama --engine ollama --gpu 0 1 --keep_alive 30m
infer-stack catalog endpoint add chat --engine ollama --host local-ollama --model llama3.2:3b

# 3. bring it up
infer-stack serve chat

# 4. open http://localhost:13000  -> pull any other model from the UI
```

That's the whole thing. The rest of this doc explains each piece and the knobs.

---

## What gets stood up

`infer-stack serve chat` renders a docker-compose project and brings it up. With
`litellm` off you get exactly two services:

| service        | what it is                                  | port            |
|----------------|---------------------------------------------|-----------------|
| `ollama-<id>`  | the Ollama daemon, pinned to your GPUs      | 11434 (host)    |
| `open-webui`   | Open WebUI, wired to the daemon's native API | **13000** (host) |

Open WebUI's **Ollama connection** points straight at the daemon, so its
*Settings → Models* panel can pull, run, and delete models, and the daemon loads
them on demand — just like talking to ollama directly. Chat history and pulled
models persist across restarts (docker volumes under your data dir).

If you'd rather also have a unified OpenAI `base_url` for scripts/other clients,
**keep LiteLLM on** (skip `config set litellm false`). Then you get a third
`litellm` service; Open WebUI uses it for chat *and* still gets the native Ollama
connection for model management. See [Variants](#variants).

---

## Step by step

### 1. One-time setup

```bash
infer-stack config init
```

Pick the **compose** backend and a data directory (where weights/state are
bind-mounted). Then make the stack lean:

```bash
infer-stack config set litellm false
```

This is the persistent default; you can also decide per-invocation with
`serve … --no-litellm` / `--litellm`.

### 2. Declare the Ollama daemon

A runtime host is the long-lived daemon (one daemon serves many tags):

```bash
infer-stack catalog host add local-ollama \
    --engine ollama \
    --gpu 0 1 \           # pin to GPU 0 and 1 (both 1080 Tis)
    --keep_alive 30m      # how long a model stays resident after last use
```

`--gpu` is exactly the GPU restriction you wanted: the daemon only ever sees the
indices you list (it becomes the container's `CUDA_VISIBLE_DEVICES`). Use
`--gpu 0` to keep it off your display card, `--gpu 0 1` to let ollama use both
(it will split a model across them when one card can't hold it).

Other optional knobs: `--num_parallel`, `--max_loaded_models`,
`--context_length`, `--image` (a custom ollama image).

### 3. Declare one anchor model

`serve` needs at least one endpoint to start the daemon. Declare a small default
— it gets pulled automatically so the box is useful the moment it's up:

```bash
infer-stack catalog endpoint add chat \
    --engine ollama --host local-ollama --model llama3.2:3b
```

For ollama, `--model` is just the **ollama tag** (`llama3.2:3b`, `qwen2.5:7b`,
…) — no separate `catalog model add` needed. The alias `chat` is what shows up
in Open WebUI. (Omit the name and it auto-names `llama3.2-3b-1`.)

You can review what you built any time:

```bash
infer-stack catalog show
```

### 4. Bring it up

```bash
infer-stack serve chat
```

This places the daemon on your GPUs, writes the compose project, runs
`docker compose up`, pulls the `llama3.2:3b` tag, and waits until it's serving.
On a terminal you'll see the compose diff and a confirm prompt first
(`--yes` to skip).

### 5. Use it — and let ollama manage the rest

Open **http://localhost:13000**. The `chat` model is already there. To add more,
go to **Settings → Admin → Models** (or the model picker's *Manage* /
*Pull a model* control) and pull any tag — `qwen2.5:14b`, `gemma2:9b`, whatever.
The daemon downloads it, keeps it on disk, and loads it on demand. You never
have to touch the catalog again for day-to-day model juggling.

---

## Managing the stack

```bash
infer-stack status        # holistic overview: backend, GPUs, leases, URLs
infer-stack ps            # docker compose ps for the stack
infer-stack logs          # tail service logs
infer-stack stack down    # stop everything
```

To change the daemon's GPUs or keep-alive later, re-run `catalog host add
local-ollama … --force` and `infer-stack serve chat` again — the daemon (and its
on-disk models) is recreated with the new pinning; Open WebUI stays up.

---

## Variants

### Keep LiteLLM too (OpenAI base_url for scripts)

Leave `litellm` on (the default). You then also get an OpenAI-compatible gateway:

```bash
infer-stack config set litellm true     # or just don't turn it off
infer-stack serve chat
infer-stack env                          # prints base_url + API key for clients
```

Open WebUI uses the gateway for chat **and** keeps the native Ollama connection
for pull/manage — best of both. The trade-off: only models declared as endpoints
are routable through the gateway; ad-hoc models you pull from the UI are
reachable in the UI but not through the OpenAI `base_url` until you also
`catalog endpoint add` them.

### Leave the display GPU free

If one 1080 Ti drives your monitor and you don't want ollama touching it, pin to
the other:

```bash
infer-stack catalog host add local-ollama --engine ollama --gpu 0 --keep_alive 30m --force
```

(Or globally: `infer-stack config set skip_display_gpus true`, which drops
display-attached GPUs from the placement pool.)

### Change the Open WebUI port

```bash
infer-stack config edit      # set ports.open_webui
```

---

## Notes for Pascal (1080 Ti)

* Ollama runs fine on Pascal; no special flags. (This is the main reason to pick
  ollama over vLLM on this hardware.)
* 11 GiB per card: a ~13B Q4 model needs both cards — `--gpu 0 1` lets ollama
  split it. Smaller models sit on one.
* `--keep_alive` trades VRAM for latency: longer keeps models resident (snappy,
  but holds VRAM); `0` unloads immediately after each request.

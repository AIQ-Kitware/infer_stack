# Leasing e2e harness (developer tests)

**This is a developer harness, not a unit suite.** It lives in `dev/` (not
`tests/`) on purpose: it stands up real docker/GPU/LiteLLM and is meant to be run
by hand on a serving host (yardrat), produce a report, and let us see what
breaks — efficiently and ergonomically. The polished, always-run, no-side-effect
checks stay in `tests/`.

It is the executable companion to `dev/leasing-test-plan.md` (the copy-paste
runbook). Same scenarios, but here the harness runs them, asserts the wiring, and
writes one self-contained report you can `rsync` back.

## Run it

```bash
# on yardrat, with the infer-stack venv active:
cd submodules/infer-stack/dev/e2e_tests

./run.sh                 # non-serving tiers (no GPU, no image pulls)
./run.sh --gpu           # + real serving on GPU 0 (vLLM + Ollama)
```

Then rsync the printed results dir back; I review `report.md`.

### Before `--gpu` (one-time, so timeouts reflect logic not 10GB pulls)

```bash
docker pull vllm/vllm-openai:v0.19.1
docker pull ghcr.io/berriai/litellm:v1.82.3-stable
docker pull ollama/ollama:latest
```

Model weights download lazily on first `acquire` (not pre-pulled): SmolLM2 135M
/ 360M, plus Qwen2.5-0.5B for the `noblip` swap tier (~1 GB, a 3rd distinct
model). The `noblip` group also needs **2 GPUs** (it skips otherwise).

GPU tiers assume **GPU 0 free**. Placement now uses **every** GPU by default —
including yardrat's display-attached **GPU 1** — so the both-GPUs / GPU-1-pin
tiers need no special flag (set `config set skip_display_gpus true` to exclude
the display GPU; `60_dedicated_f5` probes placement on a busy box).

## What runs

| tier | `--gpu`? | what it proves |
| --- | --- | --- |
| `01_environment` | always | infer-stack imports, docker compose v2, (gpu) nvidia runtime |
| `10_dryrun` | no | acquire→leases→env-file→release on the null backend; bundles |
| `20_ergonomics` | no | `paths`, `config paths leasing`, `status`, `secrets`, day-2 fallback |
| `30_negative` | no | unknown endpoint / kubeai / missing catalog fail *friendly* (no traceback) |
| `40_single_vllm` | yes | acquire a vLLM model, real chat completion, day-2 `ps`, release |
| `45_both_gpus` | yes | `--include-display-gpus` spreads two models across GPU 0 + GPU 1 |
| `50_coalescing` | yes | two leases on one model → one group (demand 2), one container |
| `60_dedicated_f5` | yes | a dedicated group needs its own GPU; placement on a busy box |
| `70_ttl_reclaim` | yes | short TTL expires; reclaim:stop group reclaimed |
| `80_run_wrapper` | yes | `run -- <cmd>` injects env, releases on exit, propagates exit code |
| `85_ollama` | yes | ollama daemon lazy pull/warmup + chat through the gateway; Open WebUI gets both the gateway (OpenAI) and the daemon (native Ollama) connections |
| `86_ollama_lean` | yes | `--no-litellm`: no gateway, Open WebUI wired straight at the daemon's native API, tag still pulled, chat hits the daemon directly, `status` shows the UI URL |
| `88_gpu_pinning` | yes | pin an ollama daemon to **GPU 1** (`--include-display-gpus`): device-reservation `device_ids=[1]`, no host-index `CUDA_VISIBLE_DEVICES`, and it runs ON the GPU (not a CPU fallback) |
| `90_concurrency` | yes | two racing acquires; file lock keeps the compose file valid |
| `91_queue` | yes | `acquire --queue`: a second dedicated group queues behind a busy GPU and lands when it frees; a no-free-GPU queue times out and rolls back (no phantom lease). Pins to GPU 0 via `--allowed-gpus 0` for deterministic contention |
| `92_gc` | yes | `infer-stack gc`: a leaked (TTL-expired, never-released) lease is a no-op before TTL, reclaimed + its stop-policy group torn down after; plain `gc` leaves an idle keep-warm model, `gc --evict` tears it down |
| `93_noblip_swap` | yes (2 GPUs) | LiteLLM no-blip across a model swap: two models on two GPUs, a process hammers one continuously while the other is released and a third model is brought up on the freed GPU — asserts **zero** failed requests and an unchanged litellm container id. Skips if `<2` GPUs |
| `99_cleanup` | always | release stragglers + `down` the project (never leak containers) |

### Running logical groups

Tiers are bucketed into groups so you can iterate on one area without running
everything (run-all is still the default — just `./run.sh --gpu`):

```bash
./run.sh --list-groups            # show the groups and their tiers
./run.sh --gpu --group vllm       # only the core vLLM serving group
./run.sh --gpu --group 'queue noblip'   # two groups at once
./run.sh --gpu --skip-group ollama      # everything EXCEPT ollama
./run.sh --gpu --only '40 50'     # ad-hoc by prefix (still supported)
```

| group | tiers | what |
| --- | --- | --- |
| `smoke` | 10 20 30 | fast, no-GPU: dry-run, ergonomics, negative cases |
| `vllm` | 40 45 50 60 70 80 90 | core vLLM serving, placement, concurrency |
| `ollama` | 85 86 88 | ollama daemon pull/warmup, lean, GPU pinning |
| `queue` | 91 92 | admission queue (`--queue`) + `gc` reclaim |
| `noblip` | 93 | LiteLLM no-blip across a model swap (needs 2 GPUs) |

`01_environment` and `99_cleanup` always run (the bookends). GPU groups need
`--gpu`; without it their tiers just record skips. Adding a tier? Put its prefix
in the right group in `expand_group()` in `run.sh`.

GPU tiers are isolated: before each serving tier the runner tears down the
`infer-stack` compose project **and wipes the ledger**, so a leftover group from
one tier can't hold the only usable GPU and starve the next. Long-running steps
(model cold-starts) print a `… still running (Ns)` heartbeat every 20s so they
don't look hung.

## Stuck run? `cleanup.sh`

If you Ctrl-C mid-run (or a box is left with leftover containers), reclaim it:

```bash
./cleanup.sh                    # down the 'infer-stack' compose project + stragglers
./cleanup.sh --wipe-ledger DIR  # + wipe the ledger under a run's results/<ts>/infer-stack-data
```

The compose project is always named `infer-stack`, so the teardown works without
needing the original run's data dir.

## The report

`run.sh` writes a timestamped dir under `results/` (gitignored):

```
results/<ts>/
  report.md            # READ THIS — summary, wiring axes, failures w/ log tails
  results.jsonl        # one machine-readable record per step
  environment.txt      # host / docker / nvidia-smi capture
  logs/<section>.<step>.log   # full combined output of every step
  infer-stack-data/    # the run's ledger + rendered compose/litellm/.env (gpu)
```

`report.md` groups results into the three axes the harness is checking:
**Correctness** (does it serve / coalesce / release), **Efficiency** (one
container per shared model, reclaim, concurrency), **Ergonomics** (`paths` /
`secrets` / `status` / env-file / day-2 wrappers). Failures are surfaced first
with their log tails so a single file tells you what broke and where.

## Knobs

| flag | effect |
| --- | --- |
| `--gpu` | enable the serving tiers |
| `--only '40 50'` | run only those numeric prefixes (+ cleanup) |
| `--group NAME` | run a logical group (e.g. `vllm`); space-separate for several |
| `--skip-group NAME` | run everything except a group (e.g. `ollama`) |
| `--list-groups` | print the groups and their tiers, then exit |
| `--keep-running` | don't `down` the compose stack at the end |
| `--keep-data` | (no-op placeholder; data dir is always kept inside results) |
| `--data-dir DIR` | use an explicit `INFER_STACK_DATA_DIR` |
| `--results DIR` | write the report somewhere other than `results/<ts>` |
| `--catalog PATH` | use a different catalog than `catalog.yaml` |

Each run uses a fresh `INFER_STACK_DATA_DIR` inside its results dir, so runs are
independent and the ledger/compose artifacts travel with the report.

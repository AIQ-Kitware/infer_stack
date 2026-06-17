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

GPU tiers assume **GPU 0 free** (GPU 1 is display-attached and skipped by the
placer — that's finding F5, which test `60_dedicated_f5` deliberately probes).

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
| `85_ollama` | yes | ollama daemon lazy pull/warmup + chat through the gateway |
| `90_concurrency` | yes | two racing acquires; file lock keeps the compose file valid |
| `99_cleanup` | always | release stragglers + `down` the project (never leak containers) |

Run a subset: `./run.sh --gpu --only '40 50'`. `99_cleanup` always runs.

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
| `--keep-running` | don't `down` the compose stack at the end |
| `--keep-data` | (no-op placeholder; data dir is always kept inside results) |
| `--data-dir DIR` | use an explicit `INFER_STACK_DATA_DIR` |
| `--results DIR` | write the report somewhere other than `results/<ts>` |
| `--catalog PATH` | use a different catalog than `catalog.yaml` |

Each run uses a fresh `INFER_STACK_DATA_DIR` inside its results dir, so runs are
independent and the ledger/compose artifacts travel with the report.

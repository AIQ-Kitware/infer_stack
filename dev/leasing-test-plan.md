# Leasing / Compose backend — manual test plan (yardrat)

Living runbook for exercising the `infer_stack.leasing` Compose path on real
hardware and recording what breaks. Validated against **yardrat**: 2 GPUs —
GPU 0 Quadro RTX 8000 (48 GiB, free), GPU 1 Quadro RTX 5000 (16 GiB, **display
attached**). Both are Turing (sm_75): **fp16 only, no bf16/FP8**.

Every test block is self-contained: it re-exports the env it needs, so blocks do
not depend on each other (only on the one-time **Setup** that writes
`catalog.yaml`). Copy-paste a whole block.

Two hardware facts shape the whole plan:
- GPU 1 has the display, and the placer skips display-active GPUs by default, so
  **only GPU 0 is usable** unless that's overridden. `--allowed-gpus` filters
  *after* the skip and can't re-include GPU 1.
- Turing needs `extra_args: ['--dtype=half']` on every vLLM endpoint or vLLM
  crashes trying bf16.

---

## Findings / fixes log

Newest first. `FIXED` = patched on the branch; `OPEN` = still to address;
`CONFIRM` = expected, verify on hardware.

- **F8 — a stale/invalid compose file bricked `acquire`. FIXED.**
  `reconcile` calls `observe()` (which runs `docker compose ps`, validating the
  on-disk file) *before* `converge()` rewrites it. A stale file from an earlier
  (pre-fix) run made `ps` raise and crashed the whole acquire before converge
  could overwrite it. `observe()` is now lenient (returns "nothing observed" on
  any docker/parse error), so converge self-heals the file. If you hit this on
  a box with an old compose file, the fix overwrites it on the next acquire; or
  clean it: `rm -rf "$INFER_STACK_DATA_DIR/leasing/compose"`.
- **F7 — unit-test isolation: ambient `INFER_STACK_*` leaked into the legacy CLI
  tests. FIXED.** The subprocess `run_cli` helpers used `setdefault`, so an
  exported `INFER_STACK_DATA_DIR` (from these test-plan blocks) made the tests
  read the real data dir. Forced to `tmp_path`. (Run unit tests in any shell
  now; the leasing acquire/run commands still use the export intentionally.)
- **F1 — `capabilities: [[gpu]]` rejected by Compose schema. FIXED.**
  `services.*.deploy.resources.reservations.devices.0.capabilities.0 must be a
  string`. `_gpu_reservation` emitted a list-of-lists; Compose wants a list of
  strings (`capabilities: [gpu]`). Fixed in `leasing/compose.py` + test.
- **F2 — LiteLLM auth: the probe must send the master key. CONFIRM/SETUP.**
  `probe_ready` sends `Authorization: Bearer $LITELLM_MASTER_KEY` from the host
  env; the container reads the same var via Compose interpolation. Export it
  before `acquire` or readiness 401s forever.
- **F3 — vLLM false-ready without `--require-generation`. CONFIRM.**
  Default readiness is "alias listed by LiteLLM", true before vLLM finishes
  loading. Use `--require-generation` for honest readiness.
- **F4 — Turing dtype. CONFIRM.** Needs `extra_args: ['--dtype=half']`.
- **F5 — Display GPU unusable, no override knob. OPEN.** Only GPU 0 is
  placeable; 2-distinct-model / dedicated tests fail placement. Needs a
  `--include-display-gpus` / `skip_display` CLI flag (not yet added).
- **F6 — No first-class dtype / protocol / image knobs. OPEN.** dtype only via
  `extra_args`; readiness probe is always `protocol=chat` (completions-only
  models fail it); images pinned in `config.py` with no leasing-CLI override.

---

## Setup (one time)

```bash
# infer-stack already installed editable on yardrat (uvpy3.11.2). Sanity:
infer-stack version          # expect 0.7.0

# Docker GPU path must work BEFORE infer-stack:
docker compose version       # need v2
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Pre-pull images so acquire timeouts reflect logic, not 10GB downloads:
docker pull vllm/vllm-openai:v0.19.1      # confirm this tag exists; else F6 blocks you
docker pull ghcr.io/berriai/litellm:v1.82.3-stable
docker pull ollama/ollama:latest

# Scratch + catalog (tailored to yardrat: Turing dtype, tiny ungated chat models)
mkdir -p ~/infer-stack-test
cat > ~/infer-stack-test/catalog.yaml <<'YAML'
models:
  qwen05b: {source: hf://Qwen/Qwen2.5-0.5B-Instruct}
  qwen15b: {source: hf://Qwen/Qwen2.5-1.5B-Instruct}
endpoints:
  qwen-small:
    engine: vllm
    model: qwen05b
    runtime: {max_model_len: 8192, gpu_memory_utilization: 0.3, extra_args: ['--dtype=half']}
    reclaim: {policy: stop}
  qwen-dup:                       # same model+runtime, exposed as qwen-small -> must coalesce
    engine: vllm
    model: qwen05b
    public_name: qwen-small
    runtime: {max_model_len: 8192, gpu_memory_utilization: 0.3, extra_args: ['--dtype=half']}
  qwen-15b:                       # 2nd distinct model -> exercises the display-GPU limit (F5)
    engine: vllm
    model: qwen15b
    runtime: {max_model_len: 8192, gpu_memory_utilization: 0.3, extra_args: ['--dtype=half']}
    reclaim: {policy: keep-warm}
  qwen-ollama:
    engine: ollama
    host: local-ollama
    model: qwen2.5:0.5b
runtime_hosts:
  local-ollama:
    engine: ollama
    placement: {gpu_indices: [0]}            # run only when no vLLM lease holds GPU 0 (or [] for CPU)
    settings: {keep_alive: 5m}
bundles:
  pair: [qwen-small, qwen-15b]
YAML
echo "catalog written to ~/infer-stack-test/catalog.yaml"
```

---

## 1. Dry-run sanity (null backend, no GPU)

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export CAT=~/infer-stack-test/catalog.yaml
infer-stack acquire qwen-small --catalog "$CAT" --env-file /tmp/is.env --no-wait --json
infer-stack leases --json
cat /tmp/is.env
infer-stack release --env-file /tmp/is.env
```
Expect: JSON with a session id, one active lease / one live group (demand 1),
env-file with `INFER_STACK_ENDPOINT_QWEN_SMALL=qwen-small`, clean release. No
containers (null backend).

---

## 2. Single vLLM model on GPU 0 — the core path

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test          # F2: needed by container AND probe
export CAT=~/infer-stack-test/catalog.yaml
infer-stack acquire qwen-small --backend compose --catalog "$CAT" \
  --require-generation --env-file /tmp/is.env --timeout 1200
```
While it blocks, in a second shell watch:
```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export C="$INFER_STACK_DATA_DIR/leasing/compose"
docker compose -p infer-stack -f "$C/docker-compose.yml" ps
docker compose -p infer-stack -f "$C/docker-compose.yml" logs --tail=80 | tail -80
```
On success, verify serving + descriptor, then release:
```bash
export LITELLM_MASTER_KEY=sk-local-test
set -a; . /tmp/is.env; set +a
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $LITELLM_MASTER_KEY" | python -m json.tool
curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"qwen-small","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
infer-stack release --env-file /tmp/is.env
```
Expect: `qwen-small` listed by `/models`; a chat completion returns text;
`qwen-small` is `reclaim: stop` so its container is gone after release. Watch
for F2 (401 → never ready), F3 (returns ready before vLLM loaded — shouldn't,
since `--require-generation`), F4 (dtype crash in vLLM logs).

---

## 3. Coalescing & demand

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
export C="$INFER_STACK_DATA_DIR/leasing/compose"
infer-stack acquire qwen-small --backend compose --catalog "$CAT" --owner alice --require-generation --timeout 1200
infer-stack acquire qwen-dup   --backend compose --catalog "$CAT" --owner bob   --require-generation --timeout 1200
infer-stack leases --json     # ONE group, demand 2; two leases
docker compose -p infer-stack -f "$C/docker-compose.yml" ps   # exactly one vllm-… service
```
Cleanup:
```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
infer-stack leases --json | python -c 'import json,sys; [print(l["id"]) for l in json.load(sys.stdin)["leases"]]' \
  | xargs -n1 infer-stack release
```

---

## 4. Dedicated (exercises the display-GPU limit, F5)

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
infer-stack acquire qwen-small --backend compose --catalog "$CAT" --owner a --require-generation --timeout 1200
infer-stack acquire qwen-small --backend compose --catalog "$CAT" --owner b --dedicated --require-generation --timeout 300
infer-stack leases --json     # two groups requested...
```
Expected on yardrat: the dedicated 2nd group **can't place** (only GPU 0 free →
placement error, no container) so its acquire times out. Confirm via `leases`
(group with demand 1 but not running) and the compose `ps`. This is F5.

---

## 5. TTL & reclaim

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
infer-stack acquire qwen-small --backend compose --catalog "$CAT" --ttl 90s --require-generation --env-file /tmp/is.env --timeout 1200
sleep 100
infer-stack leases            # lease EXPIRED; qwen-small is reclaim:stop -> group reclaimed
```
Repeat with `qwen-15b` (reclaim: keep-warm) to confirm it stays running after
TTL expiry.

---

## 6. `run` wrapper (the kwdagger pipeline-node seam)

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
infer-stack run --endpoint qwen-small --backend compose --catalog "$CAT" --require-generation -- \
  bash -c 'curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d "{\"model\":\"$INFER_STACK_ENDPOINT_QWEN_SMALL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}"'
echo "exit=$?"
infer-stack leases            # the run lease should be released
# exit-code propagation:
infer-stack run --endpoint qwen-small --backend compose --catalog "$CAT" -- bash -c 'exit 7'; echo "exit=$? (expect 7)"
```

---

## 7. Ollama daemon + pull/warmup (run with GPU 0 free)

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
export C="$INFER_STACK_DATA_DIR/leasing/compose"
infer-stack acquire qwen-ollama --backend compose --catalog "$CAT" --require-generation --timeout 900 --env-file /tmp/o.env
docker compose -p infer-stack -f "$C/docker-compose.yml" logs --tail=80 | grep -i pull   # expect "ollama pull qwen2.5:0.5b"
set -a; . /tmp/o.env; set +a
curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -d '{"model":"qwen-ollama","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
infer-stack release --env-file /tmp/o.env
```

---

## 8. Concurrency / "last render wins" fix (the headline)

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export LITELLM_MASTER_KEY=sk-local-test
export CAT=~/infer-stack-test/catalog.yaml
( infer-stack acquire qwen-small --backend compose --catalog "$CAT" --owner u1 --require-generation --timeout 1200 ) &
( infer-stack acquire qwen-dup   --backend compose --catalog "$CAT" --owner u2 --require-generation --timeout 1200 ) &
wait
infer-stack leases --json     # both leases present, demand correct, compose file intact
```
Watch for: the file lock serializing converge, no corrupt/half-written
`docker-compose.yml`, no dropped service.

---

## 9. Negative / edge cases

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export CAT=~/infer-stack-test/catalog.yaml
infer-stack acquire nope --backend compose --catalog "$CAT" --no-wait    # friendly "unknown endpoint"
infer-stack acquire qwen-small --backend kubeai --catalog "$CAT" --no-wait   # "not implemented"
```
Also: temporarily remove `--dtype=half` from `qwen-small` in the catalog and
acquire → confirm the vLLM bf16/Turing crash in the logs (F4).

---

## Debugging — capture these when something breaks

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export C="$INFER_STACK_DATA_DIR/leasing/compose"
docker compose -p infer-stack -f "$C/docker-compose.yml" ps
docker compose -p infer-stack -f "$C/docker-compose.yml" logs --tail=120
infer-stack leases --json
echo '--- rendered compose ---';   cat "$C/docker-compose.yml"
echo '--- litellm config ---';     cat "$C/litellm_config.yaml"
echo '--- backend sidecar ---';    cat "$C/leasing-compose-state.json"
```
Those pin down whether a failure is placement, render, docker, or the probe.

---

## Cleanup / reset

```bash
export INFER_STACK_DATA_DIR=~/infer-stack-test/data
export C="$INFER_STACK_DATA_DIR/leasing/compose"
docker compose -p infer-stack -f "$C/docker-compose.yml" down --remove-orphans 2>/dev/null
rm -rf ~/infer-stack-test/data        # ledger + compose state (catalog.yaml kept)
```

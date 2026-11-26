# Presenter runbook: Infer-stack live

This runbook accompanies the eight-slide live demo and the hands-on tutorial.
The commands are examples for a **trusted local Compose host**; do not run
initialization or catalog changes on a production or shared active stack
without approval. Keep one control plane per Docker host, use a private
config/data root, and protect published ports with a firewall. The default
Open WebUI is unauthenticated; the demos use the API and disable UI during
fresh setup. Never put lease environment files, API keys, or unsanitized
benchmark ledgers in this presentation directory under version control.

## Rehearsal and setup

Use an existing prepared demo control plane or initialize a fresh dedicated
one. Linux, Python 3.10+, Docker/Compose and a virtual environment are
required. Only the real model and performance demos require NVIDIA GPU
hardware, driver and container toolkit. Check free disk space for weights and
cache, and pre-pull images / pre-download weights if possible. Generation
readiness and benchmark runs may take longer than the talk.

From the repository root, use a private demo working directory outside the
repo so lease files cannot be accidentally committed:

```bash
CLIENT="$(realpath presentations/infer-stack/demo_client.py)"
REPO="$(pwd)"
umask 077
DEMO_DIR="$(mktemp -d)"
cd "$DEMO_DIR"
```

Install in an active virtual environment (`pip install -e '.[tui]'` from the
repo root for a source checkout, or `pip install 'infer-stack[tui]'`). For a
fresh demo installation only:

```bash
infer-stack config init --backend compose
infer-stack config set ui false
infer-stack paths
```

On an existing demo installation, inspect settings and leases rather than
resetting it. Ensure the ordinary LiteLLM gateway is enabled. Then add the
simulator and inspect real suggestions:

```bash
infer-stack catalog suggest --simulator --apply
infer-stack catalog endpoint list
ENDPOINT=mock-smol
```

For a GPU demo, optionally add matching real endpoints and inspect the list:

```bash
infer-stack catalog suggest --apply
infer-stack catalog endpoint list
REAL_ENDPOINT=smollm2-135m  # substitute an existing compatible chat endpoint
```

Verify `REAL_ENDPOINT` exists before using it, and select a single-GPU
endpoint for the placement demonstration. A prepared historical benchmark
HTML report is useful as an offline fallback; do not imply it is live data.
The slides themselves already include sanitized measured historical charts.

## 1. Discover and experiment

```bash
infer-stack catalog suggest
infer-stack catalog suggest --simulate-hardware 4x96@12.0
infer-stack tui
```

Show the real TUI, open Browse models (`i`), search for a known model, and
compare its Fit indicator and YAML preview. Inspect a hardware-mismatched
DeepSeek profile without acquiring it. Demonstrate Import & edit only on a
new endpoint on this demo host; a saved import changes the catalog. Do not
remove placement constraints just to make a demo work. Exit the TUI before
continuing. CLI fallback: `infer-stack catalog import
inkling-2-mini-maya-long --dry-run` (or choose a profile absent from your
catalog). Simulated hardware is for planning, not a placement override.

## 2. Shared leases (works without a GPU)

```bash
infer-stack acquire "$ENDPOINT" --ttl 30m --env-file a.env
infer-stack acquire "$ENDPOINT" --ttl 30m --env-file b.env
infer-stack leases
infer-stack ps
```

Use the default shared-compatible endpoint. Inspect both distinct lease IDs
and their deployment; the expected result is one compatible deployment, not
two copies of the model. Existing policy or incompatible configuration can
change the result. Read the actual state rather than asserting a fixed count.

```bash
infer-stack release --env-file a.env && rm -- a.env
(source b.env; python "$CLIENT" 'Still there?')
infer-stack leases
infer-stack release --env-file b.env && rm -- b.env
infer-stack ps
```

The second lease remains usable after the first is released. Reclaim policy
controls the last release (stop or keep warm). Retain an environment file if
release fails; use it or its lease ID for recovery, not `release --all`.

## 3. Scoped job and interchangeable endpoint

```bash
infer-stack run --endpoint mock-smol -- python "$CLIENT" 'What is recursion?'
infer-stack leases
# Optional GPU branch:
infer-stack run --endpoint "$REAL_ENDPOINT" -- python "$CLIENT" 'What is recursion?'
infer-stack leases
```

Show that the script is identical: it reads `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `INFER_STACK_MODELS`, then calls Chat Completions. The
simulator returns mock text; the real endpoint generates an answer. The point
is no client-side rewiring and lease cleanup, not identical model behavior.
Use `--ttl` to choose a suitable backstop for longer jobs. A SIGKILL or host
crash still requires TTL recovery or explicit release. If a model is slow to
start, continue with the CPU demonstration.

## 4. Hardware placement and engine-only logs (optional)

On a host with multiple GPUs, first verify indices and memory with
`nvidia-smi`. Set `GPU` to the index you want eligible for this call.

```bash
GPU=1  # example only: change to the verified eligible physical index
infer-stack acquire "$REAL_ENDPOINT" --allowed-gpus "$GPU" --ttl 30m --env-file gpu.env
infer-stack ps
```

Follow in a second terminal with `infer-stack logs --engine -f`. Show the
engine logs rather than gateway/UI logs; stop following with Ctrl-C. Inspect
actual placement in `ps` and `nvidia-smi`. `--allowed-gpus` is per-call
eligibility, not a general exclusion or an exclusive reservation. Existing
compatible deployments and catalog pins affect placement. Release only this
lease:

```bash
infer-stack release --env-file gpu.env && rm -- gpu.env
```

On a no-GPU host, use the hardware simulation and explain how the real
placement fields constrain an acquisition rather than attempting this demo.

## 5. Simulator-based testing (optional, source checkout)

Use the existing API-fidelity simulator fixture on a quiescent demo stack.
`--catalog` selects a different published configuration, so don't mix this
step with outstanding leases. The echo endpoint returns the prompt:

```bash
infer-stack run --catalog "$REPO/dev/e2e_tests/catalog-mock.yaml" \
    --backend compose --endpoint mock-echo -- \
    python "$CLIENT" 'Check this input.'
```

Explain `mock-flaky`: its fixture injects provider errors to test a real
client's error handling. The supplied minimal client intentionally fails
clearly on an HTTP error rather than adding retries. If you want to
rehearse an evaluation pipeline, the separate answer-aware oracle mock in
`docs/mock-endpoints.md` tests metric plumbing using a synthetic answer key
and configured ability. It requires its documented image build; do not
present its scores as real model evaluation. Use `docs/mock-endpoints.md`
for the fidelity gaps and the oracle setup. Return to the ordinary catalog
with the normal acquisition path after this scoped run; no separate
controller or new data root is needed.

## 6. Live benchmark (real GPU only)

After releasing the demo leases, choose a real compatible endpoint with
headroom for 8192 input tokens plus output. Pre-stage weights. From the
normal catalog:

```bash
infer-stack catalog endpoint list
infer-stack benchmark run "$REAL_ENDPOINT" --contexts 1024,8192 --generation-tokens 128
```

Open the `report.html` printed at startup. Show successive metrics and the
comparison dashboard; if two compatible endpoints are prepared, pass both
names for a serial comparison. The ledger is the durable record. The runner
uses normal leasing and preserves unrelated leases; it skips already-shared
deployments. The short sweep is for the demonstration, not a stable estimate
of performance. Save full results for a more deliberate experiment. Never
benchmark mock text as model performance. For the historical case study, use
the evidence deck: it includes the original six-profile sweep on one 3090.

## Finish and recover

Inspect `infer-stack leases` and release only any remaining demonstration
lease by its environment file or lease ID. Leave unrelated demand and the
shared front door intact. Delete local env files after successful release;
remove the private demo directory when no longer needed. Review
`infer-stack ps` and report any failures accurately. If a demo cannot run,
use the diagrams and recorded benchmark slides, explicitly identifying them
as explanatory or historical rather than a live result.

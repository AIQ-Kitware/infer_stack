# Inkling 2 Mini via Maya on a single RTX 3090

[Maya's Inkling Mini fork](https://github.com/alesha-pro/maya-inkling-mini)
implements its own CUDA engine and OpenAI-compatible HTTP server. It keeps the
compressed NVFP4 experts resident on Ampere, decoding their weights in the
kernels rather than requiring native Blackwell FP4 hardware. infer-stack
manages it with the existing generic custom launcher (`engine: vllm`, Compose
only); there is no vLLM inside this image.

The opt-in suggestion targets one 24 GiB sm_86 GPU (3090/A6000 class). It
reserves a whole GPU and keeps the other GPUs available. The profiles share
one persistent model directory and use the upstream FP8 KV configuration.

| Endpoint | Total context (prompt + generation) | DSpark |
| --- | ---: | --- |
| `inkling-2-mini-maya` | 196608 | on |
| `inkling-2-mini-maya-long` | 262144 | on |
| `inkling-2-mini-maya-huge` | 344064 | off |

The default leaves room for 163840 prompt tokens plus generation. The other
windows follow the upstream [guide](https://github.com/alesha-pro/maya-inkling-mini/blob/main/docs/INKLING2-MINI.md).
The 192 Ki-token setting is a derived profile, not a separately measured one.
At 262K and especially 344K there is less room for other CUDA allocations;
reserve the card for Maya. The drafter's own context remains at upstream's
`IK2_DRAFT_CTX=65536`; this is not the target model's context. Speculative
speedup at 128K+ occupied context needs a real benchmark.

## Add and acquire

The first acquisition automatically builds the bundled Maya image if it is
not already present. Docker and the NVIDIA container runtime must be working,
with a CUDA 12.8-compatible host driver. No host CUDA compiler, Maya checkout,
manual Docker build, or upstream revision selection is required.

```bash
# Use the installed infer-stack from this checkout (e.g. pip install -e .).
infer-stack catalog suggest --include inkling-2-mini-maya --apply
infer-stack acquire inkling-2-mini-maya
```

Image preparation occurs before infer-stack removes any departing containers.
It reports the build, caches the image in Docker, and reuses it for all three
variants. The source revision is pinned to
[`27daec5e608aa59dab45184e82a1f0611ea7c1aa`](https://github.com/alesha-pro/maya-inkling-mini/commit/27daec5e608aa59dab45184e82a1f0611ea7c1aa)
in the bundled Dockerfile; the image tag identifies that revision and recipe.
The build context ships in the Python distribution, including installed wheels.
The default build uses CUDA 12.8 and sm_86. This is a fixed upstream revision,
not a GPU-validated infer-stack build yet. Base image and Python dependency
versions are not fully digest/lockfile pinned. Future recipe changes need a new
image tag so that a cached image cannot silently mask the change.

The first launch downloads `Inkling-2-Mini-UD-Q4_K_XL.gguf` (about 17.6 GB)
from `unsloth/Inkling-2-Mini-GGUF` and `mtp.safetensors` from
`thinkingmachines/Inkling-2-Mini` (about 0.6 GB). Do not substitute the older
llama.cpp profile's Q4_K_M GGUF; Maya expects this particular weight layout.
Downloads go into `$(infer-stack paths runtime_dir)/maya/inkling-2-mini/models`.
Existing weights are reused across releases and across the three endpoint
variants. The launcher accepts infer-stack's managed `HF_TOKEN` if required.
The HF files are not revision-pinned by default; keep the existing files for
repeatable experiments, or download known revisions to this directory before
acquiring.

For a slow connection, use a longer readiness budget on the first acquisition
(e.g. `--timeout 1800`). No special image preparation command is required for ordinary use.

## Prepare the image separately (optional)

```bash
infer-stack image list
infer-stack image build maya-inkling-mini
```

These commands work from the installed package, even before a catalog or an
endpoint exists. The recipe name resolves to the exact versioned Docker tag
used by all three profiles. `image build` checks the Docker cache, builds
only if missing, and streams Docker's progress. It does not acquire a GPU,
download model weights, start containers, or change the catalog or ledger.
This is useful on a new host to avoid compiling during `acquire` while the
publication lock is held. It targets the Docker daemon selected by the usual
infer-stack Docker environment, so run it on the actual inference host.
The advanced `config publish --pull` path also prepares this built-in image
instead of attempting to pull a nonexistent registry image. The automatic
path and the explicit build share the same implementation.

Each managed startup extracts the tokenizer with the pinned upstream template and generates a text-only config from the upstream profile.
The launcher uses the catalog's context, port, and served name verbatim, with
Maya's own load/unload API disabled and automatic idle unload turned off so
infer-stack owns lifecycle. Maya's own sampling and loop-guard behavior are
otherwise unchanged. It defaults to `high` reasoning, overridable per request.
`minimal` is deliberately not advertised because Maya's HTTP frontend maps it
to thinking off; `max` is an alias for `xhigh`.

The normal Compose lease, placement, gateway, and release operations apply.
With LiteLLM enabled (recommended), the Maya server is on the internal Compose
network, not published directly. If you disable LiteLLM and publish the
upstream, restrict access with the host firewall; the internal Maya API is
unauthenticated like the ordinary custom launchers.

## GPU acceptance

These commands exercise the actual gateway, not merely the fact that a
container exists. Use the environment from `acquire` or the configured gateway:

```bash
infer-stack test inkling-2-mini-maya
infer-stack logs --tail 100 inkling-2-mini-maya
export OPENAI_API_KEY="$(infer-stack env LITELLM_MASTER_KEY)"
export OPENAI_BASE_URL="$(infer-stack env OPENAI_BASE_URL)"
curl -fsS "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini-maya","messages":[{"role":"user","content":"Reply with OK"}],"reasoning_effort":"none","max_tokens":128}'
```

Repeat with `low`, `medium`, `high`, and `xhigh` and a sufficient completion
budget. For agent use, verify streaming content/reasoning separation, valid
JSON tool calls and a subsequent tool-result turn. A useful long-context test
is a 128K and 164K occupied prompt followed by a fixed generation budget,
measuring time to first token, decode throughput, peak VRAM and task accuracy.
Compare the same prompts and sampling/effort settings against HyperQwen; do
not infer SWE-bench scores from a tokens/sec benchmark. Test the long/huge
variants separately if needed. The upstream project reports roughly 128 t/s
plain short-context decode and up to 212 t/s with DSpark on edits, not a
guarantee of that throughput for every coding prompt or context.

This integration has not yet been validated by building/running CUDA on a real
GPU. The included regression tests cover catalog/launcher contracts without
requiring CUDA; keep the suggestion opt-in until the host-side acceptance
above is complete. KubeAI remains unsupported for these custom launchers.

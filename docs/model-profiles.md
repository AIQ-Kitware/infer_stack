# Model discovery and serving profiles

[Documentation index](index.md) · [Quickstart](../README.md#quickstart)

A **model** is a weight source. An **endpoint** is a named serving configuration
with a particular engine, context length, image, and GPU requirement. The same
model can have multiple endpoints tuned for different hardware or workloads.

## Suggest versus Browse

```bash
infer-stack catalog suggest                 # preview fitting recommendations
infer-stack catalog suggest --all           # all fitting entries, including optional ones
infer-stack catalog suggest --include mistral-7b-instruct
infer-stack catalog suggest --simulate-hardware 4x96@12.0
infer-stack catalog suggest --apply          # merge without pruning existing entries
infer-stack catalog endpoint list
```

The normal suggestions are a curated frontier across model sizes and modalities,
plus GPT-2 (completions) and SmolLM2-135M-Instruct (chat) smoke-test endpoints.
Every fitting frontier model is included, even when a larger one also fits.
Gemma 4 E2B remains a small multimodal choice; E4B, 26B and 31B are optional.
Older Qwen, Llama, Mistral, GPT-OSS and comparison variants remain available.
This is editorial curation, not a measured ranking. Context sizes and VRAM
requirements are starting estimates. `--all` and `--include` still respect
hardware eligibility. `--simulate-hardware` is an offline planning tool.

In `infer-stack tui` (install `infer-stack[tui]`), **Browse models** (`i`)
searches the complete suggestion registry, including optional and hardware-
mismatched profiles. The Fit column and YAML preview show hardware/backend
constraints. **Import** saves without starting; **Import & edit** lets you
adjust GPU count, context, placement floors, image, launcher, flags, and
container environment; **Import & acquire** follows the managed acquisition
path. Backend-incompatible profiles may be imported but cannot be launched
from the browser on that backend. Standalone bundled presets are available
through CLI import.

```bash
infer-stack catalog import inkling-2-mini-maya-long --dry-run
infer-stack catalog import inkling-2-mini-maya-long
infer-stack catalog endpoint show inkling-2-mini-maya-long
```

Registry imports add exactly one endpoint and its model dependency. Fitting
profiles are sized to detected hardware; unmatched profiles retain their
proposed settings with an advisory warning. Imports are additive and idempotent;
conflicting existing names are rejected. Existing endpoints are not pruned or
silently retuned when the registry changes. Placement checks still apply when
acquiring. For schema and editing, see [catalog](catalog.md).

## Selected model families

The following profiles illustrate the supported configurations, not an exhaustive
list. Use the browser for the current registry.

- **Inkling 2 Mini through native vLLM:** discovery on SM120 GPUs with at least
  80 GiB offers `inkling-2-mini`, using the pinned official checkpoint and
  Inkling-enabled vLLM image, reasoning/tool parsers, TP1 and 128K context.
  Generation has been verified on a 96-GiB SM120 GPU.

- **[Inkling 2 Mini via Maya on one RTX 3090](maya-inkling-mini.md):**
  a text-only native CUDA serving profile with compressed NVFP4 experts,
  FP8 KV and DSpark. On matching GPUs, 192/256 Ki-token profiles appear in
  normal suggestions; the 336 Ki-token profile without DSpark is optional.
  Use the TUI **Browse models** button or `infer-stack catalog import
  inkling-2-mini-maya-long`, then `infer-stack acquire inkling-2-mini-maya-long`. The first acquire automatically builds and caches
  the pinned image; optionally prebuild it with `infer-stack image build
  maya-inkling-mini` (see `infer-stack image list`). No manual Docker build is
  needed. Compose manages the
  engine through its custom launcher interface (no vLLM in the image). The
  long profile has reached readiness on a 3090; performance and quality
  depend on the workload. See [benchmarking](benchmark.md).

- **Inkling 2 Mini via llama.cpp on a 24 GiB NVIDIA GPU:** an experimental text-only GGUF
  profile is available for Compose via `infer-stack catalog suggest --include
  inkling-2-mini-unsloth-q4 --apply`, then `infer-stack acquire
  inkling-2-mini-unsloth-q4`. It uses the Unsloth UD-Q4_K_M file and a CUDA
  llama.cpp server with an 8192-token context. The profile has not yet been
  validated on a 3090; available VRAM, context overhead, upstream image changes,
  and model support may affect startup. It downloads the weights on first use.
  It is a custom OpenAI-compatible launcher under the vLLM catalog engine and
  is intended for Compose, not KubeAI. Images are disabled in this profile.

- [Qwen3.8-Flash-Next on one RTX 3090](qwen38-flash-next-3090.md):
  `infer-stack catalog import qwen38-flash-next-3090` installs a bundled,
  non-destructive Compose preset; `infer-stack paths runtime_dir` supplies its
  configured weight/cache root directly for download commands. The patched
  vLLM backend offloads experts to host RAM and reserves only GPU 1, leaving
  GPU 0 available.

- [HyperQwen](https://github.com/syv-ai/HyperQwen) is a specialized model-preparation
  and serving stack for Qwen3.8-27B. infer-stack delegates requantization,
  patched-vLLM launchers, speculative decoding, and GPU-level tuning to its
  image, while hardware discovery and suggestion choose explicit catalog
  profiles by GPU *class* (compute capability + VRAM) rather than product-name
  strings. Memory-tight 24-32 GiB Ampere-or-newer cards get the reference
  fast/long/huge choices in normal CLI and TUI suggestions, without `--all`.
  High-VRAM Turing cards (sm75-sm79, >=48 GiB) get the
  measured full-262K FP16/Triton profiles, with the prepared `-fast` checkpoint
  preferred. Roomy Ampere-or-newer cards (>=48 GiB) get the fast baseline plus
  a conservative/provisional full-262K ordinary-KV prefab; use
  `dev/profile_qwen38_hyperqwen.sh` to refine that class on new hardware such as
  Blackwell. Exact GPU-name gates remain available for future exceptions backed
  by model-specific measurements. The model identity is
  `qwen3.8-27b-dbirks-hyperqwen` because it starts from the
  `dbirks/Qwen3.8-27B-W4A16-AutoRound` derivative; the unsuffixed
  `qwen3.8-27b` identity is left available for the official checkpoint.

- **[DeepSeek V4 Flash 0731](deepseek-v4-flash-sm120.md):** the TP4/128K
  configuration has passed generation readiness on four RTX PRO 6000 GPUs.
  A TP2/32K experimental configuration is also available. Neither targets a
  3090; both can be imported and edited from the registry on other hardware.

## Preparing runtime images and gated weights

Registered packaged image recipes (such as Maya) are prepared automatically
when acquisition needs them. To prepare one without acquiring GPUs:

```bash
infer-stack image list
infer-stack image build maya-inkling-mini
```

For gated weights, set the managed `HF_TOKEN` before the first acquisition
(see [Compose credentials](compose.md#custom-env-values-are-preserved)).
Most first starts also download weights and may compile kernels. Readiness does
not measure task quality or sustained throughput. Use [benchmark run](benchmark.md)
for comparative measurements on your own hardware.

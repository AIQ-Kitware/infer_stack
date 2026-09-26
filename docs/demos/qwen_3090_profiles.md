# Qwen profiles for a 3090 workstation

> **Pre-leasing.** This page uses the stack-profile CLI (`setup`, `up`, `deploy`, `switch`, `smoke-test`, `config.yaml` profiles), which was removed. Its commands no longer run. The model and vLLM settings may still inform a catalog entry; the current workflow is in the [README](../../README.md).

These built-in profiles are intended for quick GPU-1 testing on a workstation-class
RTX 3090 host.

## Qwen3.5 9B on GPU 1

```bash
infer-stack switch qwen3.5-9b-vllm-gpu1-3090 --apply --yes
infer-stack wait-ready --model qwen3.5-9b
infer-stack smoke-test --model qwen3.5-9b
```

This profile runs `Qwen/Qwen3.5-9B` through vLLM on physical GPU 1 and routes
Open WebUI through LiteLLM. It intentionally lowers context/runtime settings from
the catalog default to better fit a 24 GB 3090.

## Qwen3.6 35B-A3B FP8 on GPU 1, experimental

```bash
infer-stack switch qwen3.6-35b-a3b-vllm-gpu1-3090 --apply --yes
infer-stack wait-ready --model qwen3.6-35b-a3b
infer-stack smoke-test --model qwen3.6-35b-a3b
```

This profile uses `Qwen/Qwen3.6-35B-A3B-FP8` with a short context and low
concurrency. It is included as a best-effort experiment for the 3090 machine, but
Qwen's official vLLM recipe targets data-center GPUs for this checkpoint. Expect
OOM or unsupported-kernel failures on a 24 GB / non-FP8 GPU.

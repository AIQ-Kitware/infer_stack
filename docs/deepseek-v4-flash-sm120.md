# DeepSeek V4 Flash on SM120 workstations

`deepseek-v4-flash-0731` appears in **Browse models** on every host, with a
warning on hardware outside the suggested configuration. Automatic `catalog
suggest` remains restricted to hosts with at least four eligible SM120 GPUs
with 80 GiB each. The official
[0731 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
is pinned by revision. Its mixed FP4/FP8 weight files total about 155.4 GiB.

```bash
infer-stack catalog suggest --apply
infer-stack run --endpoint deepseek-v4-flash-0731 --timeout 7200 --ttl 8h -- \
  infer-stack test deepseek-v4-flash-0731 --max-tokens 512 --timeout 180
```

Run inside tmux for the first image/weight download. `run` releases its lease
when the test exits; the suggested endpoint stops when its last lease ends.
Existing leases retain their GPUs. Run this after those GPUs become available.

The profile uses TP4, vLLM v0.28.0, 128K context, two sequences,
FP8 KV and indexer caches, and eager execution. On November 20, 2026 it loaded
on four RTX PRO 6000 GPUs and passed the gateway generation-readiness probe.
The upstream [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)
also reports an eight-card configuration. Workload throughput on four cards
has not been measured. The model supports 1M context; this profile reserves
memory with a shorter limit.

The first reported host start spent about 38 minutes between creating the
model container and verifying generation. The engine's profile/cache/warmup
phase reported 215 seconds; later kernel JITs still ran on the first request
despite eager mode. An additional child test hit an unrelated CLI argv bug
after readiness and triggered normal lease cleanup. The public entrypoint
now preserves literal child arguments before modal parsing.

SM120 uses FP8 indexer cache and automatic MoE selection. SM100-specific
FP4-indexer and mega-MoE flags do not belong on this hardware. Speculative
decoding is disabled in this profile. The endpoint advertises the checkpoint's
`low`, `high` and `max` reasoning levels; clients should record their selected
level and output budget. A 128K serving limit is insufficient for the model's
largest recommended reasoning budget.

An offline preview exercises discovery without launching containers:

```bash
infer-stack catalog suggest --simulate-hardware 4x96@12.0
```

Unknown compute capability cannot select this profile. GPU discovery must
report SM120, and display-GPU reservation can reduce the eligible count.


## Explicit import and experimentation

On other hardware, use **Browse models**, select the profile, preview the
configuration, and choose **Import & edit**. Alternatively:

```bash
infer-stack catalog import deepseek-v4-flash-0731 --dry-run
infer-stack catalog import deepseek-v4-flash-0731
```

Importing an unmatched recipe does not reserve GPUs, download weights, or
lower its VRAM floor. The ordinary controller still validates actual placement
at acquisition. The endpoint editor permits deliberate changes to tensor
parallelism, context, GPU memory utilization, per-GPU VRAM requirements, and
vLLM arguments. Only disable the automatic cached-weight floor if the chosen
runtime actually supports offloading or another arrangement where it would
be inappropriate.

A second registry profile, `deepseek-v4-flash-0731-tp2-experimental`, is an
**unverified** TP2 / 32768-token starting configuration for two SM120 cards
with at least 90 GiB each. It uses the same pinned checkpoint and vLLM image,
one concurrent sequence, and a 0.95 memory utilization target. The checkpoint
weight bytes alone amount to roughly 77.7 GiB per GPU before runtime overhead,
so two 96-GiB cards may still be insufficient. It is deliberately excluded from
normal suggestions even when the hardware matches. Use `catalog suggest --all`
on matching hardware, or explicitly browse/import it on any host. No single
RTX 3090 recipe is claimed by either profile; using this model on much smaller
GPUs would need a different quantization or offload backend.

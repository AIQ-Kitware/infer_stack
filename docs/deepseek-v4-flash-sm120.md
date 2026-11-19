# DeepSeek V4 Flash on SM120 workstations

`deepseek-v4-flash-0731` appears in Discover models and `catalog suggest` on
hosts with at least four eligible SM120 GPUs with 80 GiB each. The official
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

The starting profile uses TP4, vLLM v0.28.0, 128K context, two sequences,
FP8 KV and indexer caches, and eager execution. This is a candidate for
four-card hosts: the upstream [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)
reports an eight-card RTX PRO 6000 configuration. Local loading, generation
and throughput on four cards remain to be measured. The model supports 1M
context; this initial profile reserves memory with a shorter limit.

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

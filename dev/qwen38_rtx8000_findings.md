# Qwen3.8-27B on Quadro RTX 8000: infer-stack profiling notes

This document records the measured Qwen3.8-27B / HyperQwen behavior on a
48-GiB Quadro RTX 8000 so future work does not have to rediscover the same
Turing-specific constraints. It is intentionally empirical: measured results
are separated from hypotheses and from follow-up ideas.

## Scope and pinned setup

The experiments use infer-stack custom endpoints around the HyperQwen image
`ghcr.io/syv-ai/hyperqwen:sha-684e927`, based on HyperQwen commit
`684e9277f163d1701d6179194c7f6bc1b9175d44`. The target model identity in
infer-stack is `qwen3.8-27b-dbirks-hyperqwen`, sourced from
`dbirks/Qwen3.8-27B-W4A16-AutoRound`.

Measured host GPU:

- NVIDIA Quadro RTX 8000
- 49,152 MiB physical VRAM (about 47.27 GiB visible to vLLM)
- compute capability 7.5 (Turing / sm75)
- 260 W power limit in the recorded runs

The performance goal is a single-user coding-agent endpoint with the full
Qwen3.8 model context (`262144` tokens), prioritizing usable long context over
maximum short-prompt throughput.

The benchmark scripts deliberately create explicit infer-stack endpoints. The
hardware is consulted while constructing the endpoints, but acquire/apply does
not silently retune them. Benchmark requests that may exceed LiteLLM's gateway
timeout go directly to the vLLM API inside the managed container; infer-stack
still owns endpoint catalog data, placement, lease lifecycle, persistent model
mounts, and routing.

## Executive summary

The card has enough memory for the full 262,144-token model context with the
prepared W4A16 target and ordinary FP16/`auto` KV. Low-bit KV is not required
for capacity. The working path is `dtype=float16`, `TRITON_ATTN`, prefix
caching on, one sequence, and no speculative decoding.

The principal limitation is cold long-prompt prefill performance, not memory.
A measured ~32K prompt takes about 230 seconds on the working sm75 Triton path.
Short decode is about 33-35 tokens/s. HyperQwen's prepared `-fast` target gives
about a 3% short-decode improvement without materially changing 8K/32K prefill.

Round 3 therefore measures the metric that matters for a persistent coding
agent: whether prefix caching makes subsequent turns cheap as the conversation
history grows. It compares the normal and `-fast` target at increasing context
depths and records cold prefill, exact cached reuse, an appended-turn reuse,
and 256-token decode from a fully cached long prompt.

## Round 1: capacity and compatibility matrix

Script: `dev/profile_qwen38_rtx8000.sh`

### Full-context W4A16 + FP16/auto KV works

The conservative endpoint booted with:

- `dtype=torch.float16`
- `max_seq_len=262144`
- `speculative_config=None`
- `kv_cache_dtype=auto`
- prefix caching enabled
- `max_num_seqs=1`
- `TRITON_ATTN`
- Marlin W4A16 linear kernels
- Triton/FLA GDN prefill and Triton GDN decode

A representative boot reported a 443,110-token GPU KV cache and maximum
concurrency 1.69x at 262,144 tokens. At `gpu_memory_utilization=0.90`, vLLM
reported about 27.34 GiB of KV cache while the model/other consumed memory was
about 14.71 GiB. There is therefore ample capacity for one full-context request
without low-bit KV compression.

Measured short decode for the plain W4A16 endpoint was 33.790 completion
tokens/s. A 32,048-token request completed in 228.47 seconds.

### Turing backend behavior

FlashAttention 2 is not available on sm75. vLLM automatically selected
`TRITON_ATTN` from the compatible candidates. Qwen's fused GDN decode kernel
also requires compute capability 8.0+, so the model fell back to the Triton GDN
decode path. These fallbacks are expected for this card and are not boot
failures.

### Activation int8 did not help

Two activation-int8 variants were tested against the same FP16/auto-KV base:

| Variant | short decode | 32K wall | KV tokens |
| --- | ---: | ---: | ---: |
| W4A16 baseline | 33.790 tok/s | 228.47 s | 443,110 |
| int8 gate-up | 33.240 tok/s | 230.92 s | 416,183 |
| int8 MLP | 32.709 tok/s | 225.45 s | 413,807 |

The int8 variants were not faster in the measured interactive decode workload
and reduced available KV capacity. They are not current candidates for the
RTX 8000 endpoint.

### Low-bit KV paths were dead ends for this pinned stack

FP8 KV failed explicitly because the Triton attention backend on this Quadro
RTX 8000 does not support that FP8 KV path; the reported native FP8 requirement
was SM89+.

`int4_per_token_head` produced very large nominal KV pools (~1.66-1.68M tokens)
but the engine later died and never became a usable endpoint. That path is not
needed for capacity and is not worth pursuing until there is a separate reason
to debug its sm75 failure.

KVarN similarly produced ~1.98M nominal KV tokens, but the tested path ran into
HyperQwen/vLLM assumptions that select an FA2-dependent route requiring compute
capability 8.0+. MTP/DFlash2 + KVarN combinations also failed. Since ordinary
FP16 KV already exceeds the model's maximum context by a wide margin, these
compression paths solve no current capacity problem on the 48-GiB card.

### The first near-full test did not prove a model failure

The first ~260K probe went through LiteLLM. LiteLLM returned HTTP 408 after its
600-second timeout and the harness spent about 2409 seconds because gateway
retry behavior repeated the long request. During that period the vLLM
container remained healthy and continued returning HTTP 200 to `/health`.

Therefore the round-1 `near-full context probe failed` result must not be read
as evidence that 262K is unsupported. It is a gateway-timeout artifact. Later
long-context benchmark traffic should go directly to vLLM.

## Round 2: attention backend and prefill chunk size

Script: `dev/profile_qwen38_rtx8000_round2.sh`

Round 2 held W4A16 + FP16/auto KV + 262,144 context fixed and swept attention
backend and `max_num_batched_tokens`. Benchmark traffic went directly to the
vLLM container.

### Triton is the working attention backend

The successful base measurements were:

| Endpoint shape | short decode | ~8K wall | ~32K wall | KV tokens |
| --- | ---: | ---: | ---: | ---: |
| Triton / 2048 | 26.151 tok/s* | 23.533 s | 231.348 s | 417,371 |
| Triton / 4096 | 33.418 tok/s | 23.483 s | 230.728 s | 416,579 |
| Triton / 8192 | 33.443 tok/s | **23.296 s** | **230.547 s** | 414,995 |
| Triton / 16384 | 33.386 tok/s | 23.770 s | 233.624 s | 392,424 |

`*` The 2048 short-decode result conflicts with the earlier ~33.8 tok/s
measurement at essentially the same configuration and is likely contaminated
by first-arm JIT/cache effects. Do not infer that 2048 intrinsically harms
decode from this one number.

4096 and 8192 are effectively tied. 8192 was marginally fastest at 8K and 32K,
while 16384 was slower and reduced KV capacity. Round 3 uses 8192 as the
provisional operating point.

### FLEX_ATTENTION was blocked by hybrid page geometry

All tested Flex endpoints failed before serving. The actionable error was:

```text
flex_attn_kv_block_size must be a power of 2 and divisible by flex_attn_block_n,
got 400, None
```

Qwen's hybrid attention/Mamba geometry caused vLLM to select a 400-token page,
which is incompatible with this Flex requirement. This is not a benchmark
showing Flex is slower on Turing; Flex was never successfully measured. Fixing
that geometry may be interesting later, but it is not required to reach full
context.

### The HyperQwen `-fast` target improves decode modestly

On Triton / 8192, the prepared fast target measured:

| Target | short decode | ~8K wall | ~32K wall | KV tokens |
| --- | ---: | ---: | ---: | ---: |
| standard | 33.443 tok/s | 23.296 s | 230.547 s | 414,995 |
| `-fast` | **34.554 tok/s** | 23.443 s | 230.868 s | **425,291** |

The ~3.3% short-decode improvement is real enough to keep the fast target as a
performance candidate; no meaningful prefill improvement was measured. Quality
parity was not established by these performance experiments, so retain the
standard target as the conservative comparison rather than silently replacing
it.

### Round-2 speculation failures were configuration refusals, not a Turing verdict

The round-2 MTP/DFlash2 endpoints failed in 13-21 seconds, before normal model
loading. The script used `--dtype=half`, while the pinned HyperQwen
`single-user/start_qwen.sh` explicitly refuses FP16 with speculation:

```text
--dtype float16 needs SPEC=off: this repo's speculative path is bf16-only.
```

HyperQwen documents a BF16 assumption in its speculative verify attention
kernel. Therefore those failures do **not** establish that MTP or DFlash2 is
fundamentally impossible on sm75. Efficient speculation on this Turing card
would likely require changing/validating HyperQwen's speculative kernels for
FP16. Merely switching this card to BF16 is not obviously desirable because
Turing lacks Ampere-class native BF16 tensor-core support. Treat speculation as
a separate future kernel-porting project, not part of the current endpoint
configuration search.

## What is established versus still unknown

Established after the completed round-3 run
`qwen38-rtx8000-round3-20260924T150901`:

- The RTX 8000 has enough VRAM for Qwen3.8-27B W4A16 at
  `max_model_len=262144` with ordinary FP16/auto KV. Low-bit KV is unnecessary
  for one full-context request.
- `TRITON_ATTN` is the working attention backend in the pinned image on sm75.
- `max_num_batched_tokens=4096` and `8192` are effectively tied at short/32K
  contexts; 8192 is the measured operating point and 16384 is worse.
- HyperQwen's prepared `-fast` target remains the best measured performance
  candidate. It improves short decode by about 3% in round 2 and is a small,
  consistent decode-at-depth win in round 3 without changing cold prefill.
- Prefix caching materially changes the persistent-agent workload. Cold 64K
  prefill took about 875 seconds and cold 128K about 3340 seconds, while exact
  cached repeats took about 10 and 18 seconds respectively. Appending a small
  new turn over those cached prefixes took about 13 and 20 seconds.
- Decode throughput falls with context depth even when the long prefix is
  cached: the `-fast` target measured about 24.8 tok/s at 8K, 16.4 tok/s at
  32K, 11.5 tok/s at 64K, and 4.84 tok/s at 128K.
- FP8 KV, the tested int4 KV path, activation-int8 variants, and the tested
  KVarN path are not useful candidates on this card with the pinned stack.
- The first near-full LiteLLM result was a gateway timeout, not proof of a vLLM
  context-capacity failure.

Still not established:

- cold prefill time at 196K or 240K; the completed round-3 run used
  `RUN_DEEP=0`;
- decode-at-depth beyond 128K;
- quality parity of the standard and prepared `-fast` checkpoints;
- whether a modified Flex page geometry would improve sm75 prefill;
- whether HyperQwen's speculative kernels can be ported profitably to FP16/sm75.

Do not extrapolate the 128K cold-prefill time to 262K as a claimed measurement.
The model has sufficient KV capacity for the full configured window, but the
completed benchmark did not ingest a 196K/240K prompt.

## Round 3: prefix-cache and context-depth experiment

Script: `dev/profile_qwen38_rtx8000_round3.sh`

Round 3 narrowed the matrix to two endpoints:

- standard W4A16 target;
- HyperQwen `-fast` target.

Both used:

- `TRITON_ATTN`;
- `max_num_batched_tokens=8192`;
- `dtype=float16`;
- FP16/`auto` KV;
- `max_model_len=262144`;
- prefix caching enabled;
- one sequence;
- no speculative decoding.

At each context depth the script performed four direct-vLLM requests: a cold
base prompt, an exact cached repeat, a small appended turn over the cached
prefix, and a 256-token decode from the fully cached appended prompt.

The completed ordinary run produced:

| Target | standard cold | fast cold | standard append | fast append | standard decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 22.12 s | 22.49 s | 1.37 s | 1.40 s | 24.56 tok/s | **24.84 tok/s** |
| 32K | 229.00 s | 229.97 s | 2.01 s | 2.03 s | 16.09 tok/s | **16.35 tok/s** |
| 64K | 874.46 s | 876.15 s | 12.93 s | **12.88 s** | 11.40 tok/s | **11.54 tok/s** |
| 128K | 3343.78 s | **3340.80 s** | **19.90 s** | 20.18 s | 4.836 tok/s | **4.837 tok/s** |

The standard target exposed a 426,478-token KV cache (1.63x nominal concurrency
at 262K); the prepared `-fast` target exposed 436,774 tokens (1.67x). Exact
cached repeats reused essentially the entire reusable prefix: for example the
128K probe reported 130,800 cached tokens of a 131,131-token cold prompt.

The fast target therefore wins only modestly, but it wins without sacrificing
capacity or cold-prefill behavior. That is enough to make it the preferred
performance prefab while retaining the standard checkpoint as the conservative
comparison until quality parity is established separately.

The historical script can reproduce the exact experiment. The newer
`dev/profile_qwen38_hyperqwen.sh` generalizes the same workload by GPU compute
capability and VRAM so the Blackwell/high-VRAM path can be tuned without baking
product names into the benchmark.

## Current endpoint guidance

For a high-VRAM Turing card (sm75-sm79, at least 48 GiB), the best measured
performance configuration is:

- prepared `Qwen3.8-27B-W4A16-AutoRound-fast` target;
- full `max_model_len=262144`;
- FP16/auto KV;
- `TRITON_ATTN`;
- `max_num_batched_tokens=8192`;
- prefix caching enabled;
- one sequence;
- no speculative decoding.

Keep an otherwise identical standard-checkpoint endpoint available as the
conservative comparison. Do not choose low-bit KV merely to obtain full context
on this class: the working FP16/auto-KV path already has substantially more KV
capacity than one 262K request needs.

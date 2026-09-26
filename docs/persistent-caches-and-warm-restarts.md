# Persistent caches and warm restarts

Provider-specific on-disk state keeps restart time short without changing model
semantics or runtime performance. vLLM runtimes use Hugging Face and vLLM
caches. Ollama uses its own model store. None of these replaces the cost of
moving model weights into GPU memory — that work happens whenever the provider
process loads a model.

## What gets cached

### Ollama model store — `state.ollama -> /root/.ollama`

Ollama downloads GGUF/model blobs into `/root/.ollama`. The Compose template
mounts `<data dir>/ollama` there for every Ollama daemon, so pulled tags
survive container replacement, `release`, and `infer-stack stack down`.

Ollama model residency is controlled by daemon/request settings such as
`keep_alive` / `OLLAMA_KEEP_ALIVE`. A short keep-alive can let a mostly idle
home-assistant-style backend unload models after use; the next request pays the
load cost again.

### Hugging Face cache — `state.hf_cache -> /root/.cache/huggingface`

The Hugging Face Hub client downloads model weights, tokenizers, and config
files into this directory the first time a model is requested. Persisting the
mount across container restarts means the second start of the same model
reuses the on-disk weights instead of re-downloading them. For 70B+ models
this can be the difference between minutes and hours of cold start.

This mount existed before the vLLM cache was added; it is unchanged.

### vLLM cache — `state.vllm_cache -> /root/.cache/vllm`

vLLM stores compiled artifacts under `VLLM_CACHE_ROOT` (defaulted to
`/root/.cache/vllm` inside the container). The most expensive entries are
`torch.compile` graphs and CUDA-graph captures keyed by the engine
configuration (model architecture, tensor-parallel size, max context length,
dtype, etc.). On a cold container with an empty cache, vLLM has to
re-compile; on a warm restart against the same configuration, those artifacts
are reused.

Host path: `<data dir>/vllm-cache/cfg-<hash>`, one subdirectory per serve
configuration (`infer-stack paths` shows the data dir). It moves with the data
dir (`infer-stack config set data_dir <path>`); there is no separate setting.
The path is created on first volume mount; no manual `mkdir` is required.

The cache key is keyed on the engine configuration. Changing `max_model_len`,
`tensor_parallel_size`, `gpu_memory_utilization`, the optimization level,
eager/compile behaviour, or the model itself will (correctly) miss the cache
and trigger a recompile. **Do not** add `--enforce-eager` or `-O0` to shave
seconds off the restart — that sacrifices steady-state throughput to skip a
one-time cost the cache already amortises.

### Secondary compiler caches

The Compose template also persists the other obvious startup caches that sit
outside `VLLM_CACHE_ROOT`:

- `state.torch_cache -> /root/.cache/torch` for PyTorch / TorchInductor
  artifacts.
- `state.triton_cache -> /root/.triton` for Triton kernels.
- `state.cuda_cache -> /root/.nv` for NVIDIA driver JIT artifacts.

Each is mounted at the tool's default location; no cache environment
variables are set. Unlike the vLLM cache, these are shared across serve
configurations, because their entries are content-addressed.

These caches reduce repeated compile/JIT work, but they do not make model
switching instantaneous. A vLLM process still has to import Python modules,
construct the engine, deserialize the selected model, allocate KV cache, move
weights to the GPU, and run any cache-missed graph/cuda-graph setup.

## What is not cached

Model weights still have to be loaded from `/root/.cache/huggingface` (CPU
RAM / page cache) into GPU HBM after every container restart. There is no
way around this without keeping the engine process alive — i.e. avoiding
the restart in the first place. See the "minimal restart" guidance below.

Switching a single vLLM runtime from model A to model B still means replacing
the vLLM engine process, because vLLM serves one model configuration per
process in this stack. That is why even tiny models can take tens of seconds
to come back healthy: the expensive work is not just downloading weights.

## Shared memory

vLLM's workers share memory when a model spans GPUs (tensor or pipeline
parallel), and Docker gives a container a 64 MiB `/dev/shm`. Upstream vLLM's
Docker guidance is `ipc: host` or a `shm_size` of a few GiB. On Compose, set
it per endpoint:

```yaml
runtime: {tensor_parallel_size: 4, shm_size: 16g}
```

It is opt-in: an endpoint without it renders exactly as before, so the
upgrade recreates no running engine. A parallel engine without it logs one
warning naming the key, and `catalog suggest` sets it on the multi-GPU
entries it adds. KubeAI needs nothing: its vLLM pods mount a memory-backed
`/dev/shm` (checked on KubeAI v0.23.4).

## Minimal-restart workflow

LiteLLM deliberately does **not** depend on engine health in the rendered
Compose file, and its route table already names every catalog endpoint. So
replacing a vLLM container during a model swap neither restarts LiteLLM nor
rewrites its config.

To refresh one engine without bouncing the gateway or Open WebUI:

```bash
# Pull a refreshed image for one service (names from `infer-stack ps`).
infer-stack stack pull vllm-<alias>

# Restart just that service.
infer-stack stack restart vllm-<alias>
```

`infer-stack apply` re-renders from the ledger and recreates only services
whose definition changed. `infer-stack stack down` followed by `infer-stack
apply` is a full restart; the bind-mounted state (Open WebUI data, Ollama
model store, vLLM caches) is not touched by `stack down`.

Ollama tags are pulled into the running daemon when an endpoint that serves
them is acquired, so adding a tag never replaces the container.

## Verifying the cache is being reused

After a warm restart, `<data dir>/vllm-cache` contains one `cfg-<hash>`
subdirectory per serve configuration. The vLLM startup logs print
"Using cached compiled graph" (or similar; exact wording varies by version)
when the cache is hit. If you see a long compile pass on every restart,
check that the volume mount is actually pointing at the persistent path and
not at an anonymous Docker volume.

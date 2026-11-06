# Single-RTX-3090 Qwen3.8-Flash-Next in infer-stack (Compose)

This example integrates the upstream [single-GPU runtime](https://github.com/DominikBucko/qwen38-flash-next-3090),
not the two-GPU launcher. The single-GPU image is a patched vLLM backend built
on the dual-GPU project's image, and serves an OpenAI-compatible API. infer-stack
manages the selected GPU, container lifetime, readiness probe, and LiteLLM route.
The other physical GPU is never reserved for this endpoint.

## Prepare model weights

Install `infer-stack` with this overlay. Use the Compose backend. Before
acquiring the endpoint, locate its persistent **runtime state directory**:

```bash
infer-stack config init
infer-stack paths
```

Look for the `runtime_dir` entry. The catalog's `/model` mount refers to
`<runtime_dir>/qwen38-flash-next-3090/model` on the host. Download the
pinned checkpoint there (about 129 GB; keep it on an NVMe-backed data dir):

```bash
python3 -m pip install -U huggingface_hub
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir /ABSOLUTE/RUNTIME_DIR/qwen38-flash-next-3090/model
```

Replace `/ABSOLUTE/RUNTIME_DIR` with the path printed by `infer-stack paths`.
The model must contain `model.safetensors.index.json` and
`runtime/mtp-int4-g32/model.safetensors.index.json`.
Do not copy the checkpoint into tmpfs: it already keeps the actively used cold
experts in RAM with 128 GB installed, and still uses mmap for its N-gram table.

The `runtime.mounts` paths are under infer-stack's runtime directory, not
arbitrary host paths. If the checkpoint is already on your NVMe elsewhere,
move it to the runtime data dir or use a carefully configured host-side
symlink at the expected path (the Docker daemon must be able to resolve it).
The other mounts persist Humming and Triton compiled kernels across leases.

## Add and serve

Merge [the example catalog](../examples/qwen38-flash-next-3090.yaml) into your
existing `catalog.yaml`; do not overwrite other agents' or users' entries.
`infer-stack paths` identifies the catalog path. On a machine with two 3090s,
the example pins physical GPU **1**, leaving GPU **0** for other uses. Swap
`placement.gpu_indices: [1]` to `[0]` if preferred. The upstream image's
`CUDA_VISIBLE_DEVICES` fallback uses the remapped device 0 *inside the
container*; do not set host GPU indices in `runtime.env`.

```bash
infer-stack catalog validate
infer-stack acquire qwen38-flash-next-3090 --no-apply
infer-stack paths leasing   # inspect the rendered docker-compose.yml
infer-stack apply
infer-stack test qwen38-flash-next-3090
infer-stack release qwen38-flash-next-3090
```

Alternatively acquire without `--no-apply` for a normal managed lease. The
LiteLLM API base is `http://127.0.0.1:14042/v1` by default. Model requests
use the catalog alias `qwen38-flash-next-3090`, while the upstream image's
served name is `Qwen3.8-Flash-Next`.

## What infer-stack changes and what it does not

- `runtime.command: [/model]` passes the model path to the upstream image's
  `qwen38-serve` entrypoint. The upstream launcher supplies its patched vLLM
  arguments, INT8 KV, MoE CPU offload, and speculative decoding.
- `runtime.compose` adds host IPC, `SYS_PTRACE`, and memlock/stack limits as
  requested by its Docker launcher. These Compose-only options are deliberately
  not supported on KubeAI; no `privileged` container setting is introduced.
- `placement.weight_floor: false` disables infer-stack's generic checkpoint-size
  VRAM floor for this RAM-offloaded model. The explicit `min_vram_gib: 23`
  remains in effect. This does not affect ordinary non-offloaded models.
- The example sets `QWEN38_HOST_RESERVE_GIB=16`: with no Docker memory cap,
  the upstream runtime budgets roughly 128 - 16 GiB for its RAM pool and
  keeps the remaining system memory available to the host. This is not a
  hard cgroup RAM cap. The upstream `MEMORY_LIMIT` launcher variable is not
  interpreted by Compose; do not assume it is enforced here.
- The checkpoint stays on NVMe; serving uses the CPU/RAM/GPU hierarchy of
  the single-GPU implementation. The other GPU is not borrowed or split.
- The published decode numbers come from a high-memory-bandwidth Threadripper
  benchmark host, not the i9-11900KF. Benchmark prefill and decode independently.

No GPU, Docker daemon, or downloaded checkpoint was available in the overlay
build environment, so the end-to-end generation path must be validated on the
actual workstation. Static tests validate catalog/placement/Compose rendering.

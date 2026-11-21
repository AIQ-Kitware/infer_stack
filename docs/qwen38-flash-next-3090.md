# Single-RTX-3090 Qwen3.8-Flash-Next in infer-stack (Compose)

This preset uses the upstream [single-3090 patched vLLM runtime](https://github.com/DominikBucko/qwen38-flash-next-3090), **not** the dual-GPU launcher. infer-stack owns the container lifecycle, physical-GPU placement, readiness probes, and LiteLLM routing. It reserves GPU **1**, leaving GPU **0** available for other work.

This guide deliberately uses your existing infer-stack configuration. **No absolute host path, manual YAML merge, `cd` into the config directory, or copy/paste of a path is required.** Commands work from any working directory. If you haven't configured infer-stack yet, run `infer-stack config init` once; otherwise, leave your existing configuration alone.

## Setup (copy/paste)

```bash
# Download the ~129 GB checkpoint into the already configured runtime directory.
python3 -m pip install -U huggingface_hub
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir "$(infer-stack paths runtime_dir)/qwen38-flash-next-3090/model"

# Add the bundled preset without modifying unrelated catalog entries.
infer-stack catalog import qwen38-flash-next-3090
infer-stack catalog validate

# Acquire on physical GPU 1, then exercise the OpenAI-compatible API.
infer-stack acquire qwen38-flash-next-3090
infer-stack test qwen38-flash-next-3090
```

`infer-stack catalog import` is idempotent. Re-running an identical import leaves the catalog untouched; if an existing model or endpoint with the same name *differs*, it reports the conflicting fields and refuses to overwrite it. There is no `--force` shortcut for preset imports. To inspect the preset before merging, use `infer-stack catalog import qwen38-flash-next-3090 --dry-run`.

The checkpoint must contain `model.safetensors.index.json` and `runtime/mtp-int4-g32/model.safetensors.index.json`. `hf download --local-dir` creates the destination directory and reuses matching files on subsequent runs. The runtime's persistent Humming/Triton caches are also kept under your configured `runtime_dir`; infer-stack creates those bind mounts at launch.

**Hardware prerequisites:** Linux, Docker, NVIDIA Container Toolkit, a suitable NVIDIA driver, at least 23 GiB free on physical GPU 1, and enough host RAM. The single-GPU runtime is tuned for 24 GiB VRAM and can hold its full cold-expert pool in RAM when 96+ GiB RAM is available. Its published throughput came from a Threadripper PRO benchmark host; performance on an i9-11900KF should be measured separately. This setup does not reserve GPU 0. The image and checkpoint downloads require network access.

## Inspect paths without copying them by hand

`infer-stack paths` still shows the full human-readable inventory. **A single-path query prints only the pathname**, making it suitable for command substitution:

```bash
infer-stack paths runtime_dir        # absolute directory for custom-runtime weights/caches
infer-stack paths catalog            # absolute catalog.yaml path
infer-stack paths compose_file       # rendered docker-compose.yml path
infer-stack paths runtime_dir --json # one object with path, kind, and status
infer-stack paths                   # full grouped inventory (existing behavior)
```

`infer-stack config paths runtime_dir` is equivalent to `infer-stack paths runtime_dir`. Queries respect `INFER_STACK_CONFIG_DIR`, `INFER_STACK_DATA_DIR`, the persisted `config set data_dir`, and the corresponding CLI overrides. A missing directory is **not** an error: querying paths does not create directories. The command prints one absolute line without terminal decorations, status text, or diagnostic banners.

The preset's `runtime.mounts` paths are *relative to the configured* `runtime_dir`, so `/model` inside the container resolves to `qwen38-flash-next-3090/model` under that directory. No `MODEL_DIR` setting or host-side path edit is needed. The bundled preset is also available for reference as [`examples/qwen38-flash-next-3090.yaml`](../examples/qwen38-flash-next-3090.yaml).

## Day-2 use

```bash
infer-stack leases
infer-stack logs -f qwen38-flash-next-3090
infer-stack release qwen38-flash-next-3090
```

For an explicit preview that does not launch the container:

```bash
infer-stack acquire qwen38-flash-next-3090 --no-apply
infer-stack paths compose_file
infer-stack apply
infer-stack test qwen38-flash-next-3090
```

The managed gateway normally listens at `http://127.0.0.1:14042/v1`. Ask for the alias `qwen38-flash-next-3090`; the upstream vLLM container's served name is `Qwen3.8-Flash-Next`. One request is served at a time, up to 135,168 tokens per request, using an INT8 KV cache and MTP3 speculative decoding.

## Implementation notes

- `runtime.command: [/model]` delegates argument construction to the upstream `qwen38-serve` entrypoint, which owns its patched vLLM flags and CPU/GPU MoE configuration.
- `runtime.compose` supplies host IPC, `SYS_PTRACE`, and memlock/stack limits. This is Compose-specific; KubeAI deliberately rejects it.
- `placement.weight_floor: false` disables the generic on-disk-checkpoint VRAM floor for this RAM-offloaded model, while `placement.min_vram_gib: 23` remains mandatory.
- `QWEN38_HOST_RESERVE_GIB=16` asks the upstream runtime to leave host RAM for other work. This is an internal runtime budget, **not** a hard Docker memory cgroup limit. The upstream single-GPU launcher in its own repository has separate `MEMORY_LIMIT` handling.
- The checkpoint remains on NVMe, with hot experts in GPU VRAM, other experts in system RAM, and the FP8 N-gram table memory-mapped. A `tmpfs` checkpoint is not needed for this profile.
- If GPU 1 is already occupied, acquire will be unable to place the endpoint there. To prefer GPU 0 instead, use `infer-stack catalog edit` and change `placement.gpu_indices` to `[0]`. The import command never overwrites that local edit.

Offline contract tests validate the bundled preset, scalar paths, additive catalog import, and Compose/GPU placement. Actual CUDA startup and tokens-per-second require running the copied setup commands on the workstation.

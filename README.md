# Infer Stack

[![PyPI version](https://img.shields.io/pypi/v/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![Python versions](https://img.shields.io/pypi/pyversions/infer-stack.svg)](https://pypi.org/project/infer-stack/)
[![License](https://img.shields.io/pypi/l/infer-stack.svg)](https://github.com/AIQ-Kitware/infer_stack/blob/main/LICENSE)

**Run local language models on demand, without managing a separate Docker
stack for each model.**

Infer-stack manages model configurations, GPU placement, and the lifetime of
inference servers. Acquire a model when you need it, use a consistent API URL,
and release it when you're done. Use it from the command line, a terminal UI,
or an automated evaluation pipeline.

- **Discover** model configurations for your hardware, or import and tune known
  configurations yourself.
- **Serve** vLLM, Ollama, and compatible custom engines behind a shared LiteLLM
  gateway, optionally with Open WebUI.
- **Lease** endpoints for a standing service or a single command; reuse compatible
  deployments and reclaim idle ones.
- **Compare** inference performance across contexts, configurations, and machines
  with a live benchmark report and a portable results ledger.

## Quickstart

### 1. Install and configure

Start with a Linux workstation, Python 3.10+, Docker and the Docker Compose
plugin. For NVIDIA GPU inference, install a working NVIDIA driver and NVIDIA
Container Toolkit first. The no-GPU simulator below does not require the NVIDIA software or hardware.

In a Python virtual environment:

```bash
pip install 'infer-stack[tui]'
```

The `tui` extra is optional. From a source checkout, use `pip install -e '.[tui]'`
instead. Initialize the Compose backend and select a data directory with room
for model weights and caches:

```bash
umask 077
infer-stack config init --backend compose
```

**Security:** The default Open WebUI has no login screen, and Compose publishes
its ports on the host. Use a trusted, firewalled machine; do not expose it to
untrusted networks. For API-only use, run `infer-stack config set ui false`
before acquisition. Keep config/data directories private and generated
environment files secret (do not commit `lease.env`). See [known limitations](docs/planning/known-limitations.md).

### 2. Discover and acquire a model

```bash
infer-stack catalog suggest --apply
infer-stack catalog endpoint list
```

Suggestions are additive, hardware-aware starting configurations, not guarantees
that every model will fit. Choose an endpoint from the list. For a small chat
smoke test on a compatible GPU, use:

```bash
ENDPOINT=smollm2-135m
infer-stack acquire "$ENDPOINT" --env-file lease.env
infer-stack test "$ENDPOINT"
```

Review and confirm deployment changes when prompted. The first acquisition can
pull or build an image and download weights. It waits for a real generation,
not just a running container; allow a longer `--timeout` for large models.

**No GPU?** Substitute these commands for step 2 to exercise the same workflow:

```bash
infer-stack catalog suggest --simulator --apply
ENDPOINT=mock-smol
infer-stack acquire "$ENDPOINT" --env-file lease.env
infer-stack test "$ENDPOINT"
```

The [simulator](docs/mock-endpoints.md) generates mock text; it is not a real
model or a performance benchmark.

### 3. Use the endpoint

- **Browser:** open <http://127.0.0.1:13000> if Open WebUI is enabled.
- **API:** use the endpoint name as the model ID and the default LiteLLM base
  URL `http://127.0.0.1:14042/v1`. Load the credentials from your trusted local
  `lease.env` file (it exports `OPENAI_BASE_URL` and `OPENAI_API_KEY`).
- **Terminal UI:** run `infer-stack tui` to browse/import models, monitor GPUs,
  edit endpoints, acquire/release leases, and view logs.

For example, send a chat request through the OpenAI-compatible API (with the
normal LiteLLM gateway enabled and `curl` installed):

```bash
(
    source lease.env
    curl -fsS "$OPENAI_BASE_URL/chat/completions" \
        -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$INFER_STACK_MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}"
)
```

### 4. Release it

```bash
infer-stack leases
infer-stack release --env-file lease.env
```

This releases only the lease recorded in `lease.env`. Suggested endpoints stop when
their last lease ends; model weights and caches remain on disk. The gateway and
UI remain available for the next model. Remove the local `lease.env` when you
no longer need its credentials. For a full stack shutdown, see [operations](docs/operations.md#shutdown-and-reclaim).

## Everyday workflows

| I want to… | Command |
| --- | --- |
| See available endpoint names | `infer-stack catalog endpoint list` |
| Discover or edit known configurations | `infer-stack tui` → Browse models |
| Import a known profile | `infer-stack catalog import NAME` |
| Run a job with an automatic lease | `infer-stack run --endpoint NAME -- COMMAND` |
| See running deployments | `infer-stack leases` / `infer-stack ps` |
| Follow only model engine logs | `infer-stack logs --engine -f` |
| Benchmark several endpoints serially | `infer-stack benchmark run NAME1 NAME2` |
| Compare historical measurements | `infer-stack benchmark report` |
| Find configuration and cache directories | `infer-stack paths` |
| Explore all CLI commands | `infer-stack help tree` |

In the TUI, **Suggest** recommends configurations for the detected hardware;
**Browse models** shows the full registered suggestion pool, including optional
and experimental configurations with hardware-fit warnings. Import & edit is
the path for trying a model on different hardware. Importing does not override
placement checks. See [model discovery and profiles](docs/model-profiles.md).

## Choose a backend

**Compose** is the simplest way to serve models on one machine. For a Kubernetes
cluster, **KubeAI** uses the same catalog/lease interface with cluster scheduling
and vLLM; Ollama and custom launcher commands/mounts are Compose-only. See the
[backend comparison](docs/backend-parity.md) and [cluster setup](docs/cluster-setup.md).

## Documentation

The **[documentation index](docs/index.md)** organizes all the guides by task.
For common next steps: [custom endpoints](docs/catalog.md),
[operations and troubleshooting](docs/operations.md),
[Compose configuration](docs/compose.md), [benchmarking](docs/benchmark.md),
and [KubeAI and remote control](docs/kubeai-backend.md).

Infer-stack is a Linux-oriented, planning-stage project. See
[known limitations](docs/planning/known-limitations.md) for operational and
security boundaries. For contributors, see [AGENTS.md](AGENTS.md) and
[dev/README.md](dev/README.md).

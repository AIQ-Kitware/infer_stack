# Infer-stack documentation

[Project overview and quickstart](../README.md)

Start with the quickstart on a single Linux host. The remaining guides are
organized by the task you want to do; the [CLI help](../README.md#everyday-workflows)
remains the complete flag reference.

## Get started

| Guide | Use it when… |
| --- | --- |
| [Quickstart](../README.md#quickstart) | Installing, discovering and acquiring your first endpoint |
| [Model discovery and profiles](model-profiles.md) | Choosing, importing and tuning serving configurations |
| [Leasing walkthrough](source/manual/leasing-demo.md) | Running several models and using the browser or an API client |
| [Ollama + Open WebUI](source/manual/ollama-openwebui-tutorial.md) | Running a daemon that manages its own model downloads |
| [Mock endpoints](mock-endpoints.md) | Testing the lifecycle without a GPU or real model |

## Configure and operate

| Guide | Use it when… |
| --- | --- |
| [Catalog and custom endpoints](catalog.md) | Declaring models, endpoints, hosts, external services, placement and launchers |
| [Operations](operations.md) | Managing leases, running jobs, viewing logs, locating state and debugging |
| [Compose](compose.md) | Setting up local inference, UI, API, proxy and persistent storage |
| [Benchmarking](benchmark.md) | Measuring latency, throughput and cache reuse; combining ledgers across machines |
| [LiteLLM gateway routing](litellm-gateway-routing.md) | Configuring static/dynamic routes and upstream connectivity |
| [Persistent caches and warm restarts](persistent-caches-and-warm-restarts.md) | Understanding disk cache versus GPU residency |
| [Reasoning effort](reasoning-effort.md) | Configuring reasoning controls for HyperQwen and pi |
| [Slurm compatibility](slurm-compatibility.md) | Using infer-stack within job-managed GPU allocations |

## Model-specific recipes

- [Inkling 2 Mini via Maya on RTX 3090](maya-inkling-mini.md)
- [Qwen3.8 Flash Next on RTX 3090](qwen38-flash-next-3090.md)
- [DeepSeek V4 Flash on SM120](deepseek-v4-flash-sm120.md)
- [HyperQwen and other registered profiles](model-profiles.md#selected-model-families)

These are starting configurations with documented validation status. Fit
warnings are advisory when importing; actual acquisition enforces resource
requirements. Benchmarks are not model-quality evaluations.

## Kubernetes and multiple machines

- [Cluster setup](cluster-setup.md): K3s bootstrap, worker join/acceptance,
  and handing physical GPUs between Compose and Kubernetes.
- [KubeAI backend](kubeai-backend.md): install, configure, migrate an existing
  recovery ledger, monitor cluster state, and use the remote control service.
- [Backend parity](backend-parity.md): current similarities and differences.

## Architecture, limitations, and development

- [Known limitations](planning/known-limitations.md): current operational and
  security boundaries. Read before sharing an installation.
- [ADR 0001](adr/0001-user-config-is-authoritative.md): configuration authority
  and recovery snapshots.
- [Stack graph profiles (removed)](stack-graph-profiles.md): migration context
  for older releases.
- [Planning documents](planning/): proposals and design notes; not guarantees of
  current behavior.
- [Backend parity work queue](queue.md): engineering history and acceptance
  work, not a user guide.
- [Developer guide](../dev/README.md) and [agent instructions](../AGENTS.md).

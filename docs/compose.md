# Compose backend: single-host inference

[Documentation index](index.md) · [Quickstart](../README.md#quickstart)

Use Compose for one Linux host with Docker and its Compose plugin. It can run
vLLM, Ollama and custom OpenAI-compatible launchers, mixed freely, behind the
optional gateway and UI. NVIDIA GPU workloads also require a working NVIDIA
driver and container toolkit. To try the control plane without a GPU, use the
[simulator](mock-endpoints.md).

## Getting started

The recommended path is the [README quickstart](../README.md#quickstart).
For a deliberately hand-built catalog instead of automatic suggestions:

```bash
infer-stack config init --backend compose
infer-stack doctor --gpu
infer-stack catalog init

# A vLLM endpoint.
infer-stack catalog model add smol135 --source hf://HuggingFaceTB/SmolLM2-135M-Instruct
infer-stack catalog endpoint add --model smol135    # -> smol135-1
infer-stack acquire smol135-1 --env-file lease.env
# Release before trying another endpoint on the same GPU.
infer-stack release --env-file lease.env

# Alternatively: an Ollama endpoint, on a daemon pinned to GPU 0.
infer-stack catalog host add local-ollama --engine ollama --gpu 0
infer-stack catalog endpoint add chat --engine ollama --host local-ollama --model qwen3.5:4b
infer-stack acquire chat --env-file lease.env
```

For more complete examples, see the [leasing walkthrough](source/manual/leasing-demo.md)
and the [Ollama tutorial](source/manual/ollama-openwebui-tutorial.md).
For suggestions and imports, see [model discovery](model-profiles.md).

## Connect clients and check readiness

With LiteLLM enabled, every endpoint is reachable by its alias at:

```text
http://127.0.0.1:14042/v1
```

`acquire` already blocks until the endpoint returns a real generation through
that front door, which is stronger than Docker's container health. After
`acquire --no-wait`, block separately; to check again later, send one request:

```bash
# Use the name of the endpoint you actually acquired.
infer-stack wait chat
infer-stack test chat --prompt "Name three colors." --max-tokens 32
```

For clients without a lease env-file, query the front door and managed key:

```bash
export OPENAI_BASE_URL="$(infer-stack env OPENAI_BASE_URL)"
export OPENAI_API_KEY="$(infer-stack env LITELLM_MASTER_KEY)"
```

or get both, plus per-endpoint names, from
`infer-stack acquire <endpoint> --env-file lease.env`.

To replace the gateway's master key (refused while leases are active; the
gateway restarts, and clients must fetch the key again):

```bash
infer-stack secrets rotate
```

## Open WebUI authentication

Open WebUI runs with `WEBUI_AUTH=False`: no login screen, and anyone who can
reach port 13000 gets the UI. By default Compose publishes the UI and gateway
ports on the host, not only loopback; the localhost URLs in the examples are
client addresses, not network-access restrictions. Keep the host on a trusted,
firewalled network, or run without the UI (`config set ui false`). Direct
engine ports (when the gateway is disabled) also need protection. The gateway
key is not a substitute for restricting access to the UI.

Use a private data/config directory and `umask 077` before creating or
rendering configuration. Generated environment files contain secrets and should
only be sourced from trusted local inputs. See [known limitations](planning/known-limitations.md).

## Reverse proxy

An optional nginx service publishes one HTTP port with the UI at `/` and the
API at `/v1`. It needs the LiteLLM gateway.

```bash
infer-stack config set reverse_proxy true                        # port 80
infer-stack config set reverse_proxy '{enabled: true, port: 8080}'
```

`acquire --reverse-proxy` turns it on for one call. Add `config_path:
/path/to/nginx.conf` to the block to mount your own config at
`/etc/nginx/conf.d/default.conf` instead of the generated one.

It does no TLS and no authentication. The proxy publishes its own port in
addition to the gateway/UI ports; restrict direct access to those services
as well. No certificate is mounted, so terminate TLS in an external proxy. The TLS
and LDAP settings of the pre-leasing profiles no longer exist.

## Persistent state and database layout

Everything lives under the data dir (`infer-stack paths`):

* `open-webui/`: Open WebUI's data directory (accounts, chats, settings),
  mounted at `/app/backend/data`;
* `postgres-litellm/`: LiteLLM's route store, rendered only with
  `config set dynamic_routing true`;
* `ollama/`: the Ollama model store, mounted at `/root/.ollama`;
* `hf-cache/`, `vllm-cache/`, `torch-cache/`, `triton-cache/`, `cuda-cache/`:
  vLLM weights and compile caches (see
  [docs/persistent-caches-and-warm-restarts.md](persistent-caches-and-warm-restarts.md));
* `runtime/`: directories an endpoint's `runtime.mounts` asks for;
* `leasing/`: the ledger and the rendered compose project.

Open WebUI chat history is not tied to the models currently served, so old
chats may name aliases the gateway no longer advertises. That is expected.

## Custom .env values are preserved

The compose project's `.env` holds the managed secrets (`LITELLM_MASTER_KEY`,
`HF_TOKEN`, …). `infer-stack env KEY=VALUE` merges a value into it, and keys
you add are kept across renders. Compose uses the file for interpolation, so a
key reaches a container only when the rendered service references it (as
`HF_TOKEN` does for vLLM). Set `HF_TOKEN` before the first `acquire` of a
gated model.

## vLLM startup caches

Generated Compose mounts persist Hugging Face, vLLM, PyTorch/TorchInductor,
Triton, and CUDA JIT caches. Warm starts avoid redownloading and redoing many
compile/JIT steps, but a vLLM model swap still creates a new engine process and
must reload weights into GPU memory.

For logs, GPU placement, lease cleanup and troubleshooting, see
[operations](operations.md) and [catalog placement](catalog.md#constraining-placement-to-specific-gpus).
For static/dynamic gateway routes, see [LiteLLM routing](litellm-gateway-routing.md).

# Operations: leases, configuration, and troubleshooting

[Documentation index](index.md) · [Quickstart](../README.md#quickstart)

## The lease lifecycle

The catalog is your set of available endpoints. A lease is a claim that a
particular endpoint should be serving. The controller uses the active leases
to reconcile deployments, reusing compatible serving processes where possible.
A lease is not necessarily one process or an exclusive GPU reservation.

```bash
infer-stack acquire NAME --ttl 2h --env-file lease.env
infer-stack leases
infer-stack release --env-file lease.env
```

`acquire` with no TTL creates a standing lease. With `--ttl`, the lease can
expire; use `infer-stack gc` to reclaim expired leases and `infer-stack renew`
to extend a lease. On a local Compose host, `--queue` waits for GPUs rather than
failing immediately. Kubernetes scheduling is handled by the cluster.

`settings.yaml` (`infer-stack config …`) and `catalog.yaml` are the user
configuration. Leasing keeps an
internal frozen recovery snapshot so a crash cannot re-render committed state
with different settings, but ordinary `acquire` advances that snapshot
automatically. Compatible catalog additions can be acquired while other models
are live; conflicting redefinitions or global setting changes take effect once
the affected stack is quiescent. `infer-stack config publish` is an advanced
pre-seeding/preview tool for multi-catalog operators, not a required fourth
step. See [ADR 0001](adr/0001-user-config-is-authoritative.md).

### Run a command with a lease

```bash
infer-stack run --endpoint NAME -- your-command --your-option
```

Arguments after `--` pass to the child unchanged, including numeric
strings and child options. Completion, Ctrl-C and SIGTERM release the command's own lease;
cancellation during endpoint readiness also rolls back acquisition. SIGTERM
returns status 143. The wrapper forwards cancellation to its child and waits
up to ten seconds before killing an unresponsive child. Docker workers should
use `docker run --init` so termination reaches the container process. Release
runs inside the wrapper, without a separate scheduler cleanup job. SIGKILL or
a host crash cannot run cleanup: TTL expiry and explicit `release <lease-id>`
remain recovery paths.

[Serving provenance](catalog.md#inspect-an-endpoints-serving-identity) describes
configured model, image and runtime identity. `run` exports
`INFER_STACK_ENDPOINT_DESCRIPTOR`, the path to a public descriptor containing
the assigned deployment's serving facts. The file lasts for the child command;
credentials remain in its environment. Unpinned artifacts are reported as
uncertain. These facts describe the launch specification, not loaded-weight
attestation.

The child receives the endpoint's OpenAI base URL, API key and model metadata.
For `run`, a two-hour soft TTL is the default backstop; choose `--ttl` for the
expected job length. To reach an endpoint already served elsewhere, use
`access` or `run` with an [external catalog entry](catalog.md#catalog-schema).

## Where config and rendered artifacts live

`infer-stack` follows XDG basedir conventions, so the directory you invoke it
from never changes which config it reads or where it writes. There are two
roots:

| What | Default location | How to relocate |
| --- | --- | --- |
| `settings.yaml`, `catalog.yaml` | `~/.config/infer_stack/` (resp. `$XDG_CONFIG_HOME`) | `--config-dir` or `INFER_STACK_CONFIG_DIR` |
| **Everything generated**: `leasing/` (the ledger, the compose project, its `.env`) and the bind-mounted state (`hf-cache/`, `vllm-cache/`, `open-webui/`, `ollama/`, …) | `~/.local/share/infer_stack/` (resp. `$XDG_DATA_HOME`) | `config set data_dir <path>`, `--data-dir` or `INFER_STACK_DATA_DIR` |

`infer-stack paths` prints every resolved path and whether it exists. A single
path is available as plain absolute stdout, e.g.
`infer-stack paths runtime_dir`, `infer-stack paths catalog`, or
`infer-stack paths compose_file`. Add `--json` for metadata.

The data dir **relocates the one infer-stack installation controlling a
host/backend; it does not create an isolated second installation**. Do not run
controllers from multiple config/data roots against the same Docker host or
Kubernetes namespace. See the
[single-owner limitation](planning/known-limitations.md#one-control-plane-per-host-or-backend-namespace).

```bash
# Persist it once; later commands read it from settings.yaml.
infer-stack config set data_dir /data/service/docker/infer-stack

# Or relocate both roots before a fresh installation.
export INFER_STACK_CONFIG_DIR="$PWD/cfg" INFER_STACK_DATA_DIR="$PWD/stack"
infer-stack paths
```

`--config-dir` / `--data-dir` are accepted by every subcommand, after the
subcommand name.

## Inspect deployments and logs

`ps` and `logs` read the backend directly; `stack` wraps `docker compose` on
the rendered project, so you never `cd` into it or repeat `-f`/`--env-file`.

```bash
infer-stack ps                            # engines, gateway, UI
infer-stack ps -a                         # include exited instances
infer-stack logs -f NAME                  # follow whatever serves an endpoint
infer-stack logs --engine -f               # only engines, including newly acquired ones
infer-stack logs --tail 200 litellm       # the gateway's backlog
infer-stack logs -f --raw litellm         # full LiteLLM tracebacks
infer-stack stack restart open-webui      # docker compose restart
infer-stack stack stop                    # stop everything (no remove)
infer-stack stack start                   # start it back up
infer-stack stack pull                    # refresh images
infer-stack stack compose -- ps --format json   # any other docker compose command
```

`--engine` filters to model engine instances using the same classification as
the TUI, excluding gateway and WebUI logs. It applies to history and follow
mode. `logs` accepts a service or pod name, a container id prefix, a deployment id
or an endpoint alias. Interactive `infer-stack logs -f` compacts only
explicitly registered, known-noisy LiteLLM traceback shapes; unknown
tracebacks pass through unchanged. Redirected or piped output stays raw, and
`--raw` disables compaction in an interactive follow. `--no-color` drops the
name-prefix colors. The TUI uses the same compactor.

Ollama tags are pulled into the daemon on the first `acquire` of an endpoint
that serves them. `stack compose` reaches the daemon for anything else, e.g.
`infer-stack stack compose -- exec ollama-local-ollama ollama list`.

On the KubeAI backend `ps` and `logs` read pods, and the `stack` Compose verbs
act on the gateway's Compose project on this host.

## Inspect an endpoint before running it

```bash
infer-stack catalog endpoint show NAME
infer-stack acquire NAME --no-apply   # declare + write the compose project; start nothing
infer-stack paths leasing                   # where docker-compose.yml landed
infer-stack apply                           # start it (or release its lease to discard)
```

## Shutdown and reclaim

The following commands affect the entire managed stack, not just your lease.
Prefer `release --env-file lease.env` or `release LEASE_ID` for routine use.

```bash
infer-stack release --all            # drop every lease
infer-stack release --all --evict    # ...and stop the engines now
infer-stack clean -f                 # no leases, nothing on a GPU; the gateway stays
infer-stack stack down               # docker compose down, bypassing the ledger
```

After the final `release` on an endpoint, the default `reclaim: stop` tears the
model down. Endpoints explicitly set to `reclaim: keep-warm` remain resident
until another lease needs their GPUs or `infer-stack evict` stops them.
`stack down` releases no lease, so the next `apply` or `acquire` brings leased
models back.

Weights, compile caches, and chat history are bind-mounted from the data dir;
these Compose commands do not remove them. Do not delete the data root unless
you intend to lose its persistent state, including the recovery ledger.

## Switching models

There is no single active model to switch. `acquire` another endpoint and it
runs beside the first; release the first when you are done:

```bash
infer-stack acquire smol135-1
infer-stack acquire chat          # both serve if their placement fits
infer-stack leases                # find smol135-1's lease id
infer-stack release LEASE_ID
```

With the default static route table, the gateway carries routes for catalog
endpoints; starting or stopping an already-declared model does not require
recreating the gateway or Open WebUI. When GPUs are
short, `acquire` fails fast; `--queue` waits for a GPU instead, and idle
`keep-warm` deployments are evicted to make room. vLLM containers are named
after the served alias (`vllm-<alias>`), so `docker ps` and
`infer-stack logs vllm-<alias>` identify them.

## Diagnosing readiness

`docker compose` health only means a container-level healthcheck passed. It
is not "the routed model answers a request through the front door", which is
what `acquire` and `wait` check: a model swap starts a new engine process,
and LiteLLM stays up while returning upstream connection errors until vLLM
has loaded the weights.

```bash
infer-stack acquire NAME --yes     # waits for a real generation
infer-stack wait NAME              # after acquire --no-wait
infer-stack test NAME              # one generation through the gateway
infer-stack status                       # desired vs running, per deployment
infer-stack logs NAME --tail 80          # the engine or gateway log
infer-stack logs --engine -f              # follow only model engines
```

A detected crash-looping engine can fail acquisition early with its error quoted.
The readiness timeout also bounds waiting for an unavailable server. Reading Docker's own
signals: `litellm exited with code 137` is a SIGKILL (an OOM kill or a forced
replacement), whereas LiteLLM returning HTTP 500 with `Cannot connect to host
vllm-*` means LiteLLM is running but cannot reach its upstream. This is normal
while the engine loads; if it persists, inspect engine logs and routing.

For KubeAI-specific problems, see [the cluster guide](kubeai-backend.md#diagnosing-serving-failures).

## Python and scripting interface

The CLI is built on [kwconf](https://github.com/Erotemic/kwconf), so commands
are also importable Python classes. For example, if `smollm2-135m` is in the
catalog:

```python
from infer_stack.cli import AcquireCLI, TestCLI

AcquireCLI.main(argv=False, names=['smollm2-135m'], yes=True)
TestCLI.main(argv=False, name='smollm2-135m')
```

Use `infer-stack leases` to find and release the lease when done. `manage.py` in a checkout and the installed
`infer-stack` command share the same entry point. `infer-stack help tree` is the
command inventory; individual subcommand help is the flag reference.

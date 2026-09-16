# Known limitations

`infer-stack` is still at planning-stage maturity. The limitations below are
accepted for now and define the environment in which the current implementation
is intended to be operated.

## Generated shell environment files require trusted inputs

Lease helper commands can emit shell files containing `export NAME=value`
assignments. Values are not currently shell-escaped. Sourcing one of these files
can therefore execute shell syntax embedded in a configured value.

Treat the catalog, config, model names, endpoint URLs, API keys, and generated
shell files as fully trusted local input. Do not source generated environment
files derived from untrusted or multi-tenant configuration. This is a known
security limitation, not a supported sanitization boundary.

A future hardening pass should quote values for the target shell, validate
variable names, and reject values containing unsupported control characters.

## Generated Compose environment files may be readable by other local users

Generated Compose `.env` files can contain deployment credentials such as the
LiteLLM master key or database password. They are currently created using the
process umask rather than an enforced private file mode. On a multi-user host,
a permissive umask can make those files readable by other local accounts.

Operate infer-stack under a dedicated trusted account and keep its config/data
roots private. Until file modes are enforced by the application, use a
restrictive umask such as `umask 077` before setup, render, and controller
operations. Do not treat the generated directory as safe for mutually
untrusted local users.

## One control plane per host or backend namespace

`--data-dir` and `INFER_STACK_DATA_DIR` relocate infer-stack state; they do not
namespace independent controllers. Compose resources currently share a fixed
project identity, and KubeAI reconciliation identifies resources with a shared
managed label. Two controllers using different roots against the same Docker
host or Kubernetes namespace can disagree about desired state and remove or
replace one another's resources.

The supported operating model is one infer-stack control plane per Docker host,
or one per Kubernetes namespace:

1. Pick one config root and one data root.
2. Ensure every controller and administrative command uses those same roots.
3. Stop the existing controller before relocating either root.
4. Move or re-create the state, update the configured paths, and then start the
   single controller again.

This limitation is deliberate for now. Adding an instance identifier to every
Compose and Kubernetes resource could isolate cleanup operations, but it would
not make two independent GPU schedulers safe or cooperative on one machine.
True multi-instance support therefore needs both resource namespacing and a
shared hardware-allocation model; partial namespacing would give a misleading
sense of safety.

## Linux-only execution

Linux is the only supported execution platform. The implementation relies on
POSIX process locking and Linux-oriented container, GPU, and service-management
workflows. Windows is not supported or tested. Other POSIX systems are not a
supported deployment target even when individual pure-Python modules happen to
work there.

## Leasing: scope boundaries and known faults

These are decisions about what leasing deliberately does **not** do. They are
recorded so a missing feature is recognised as a choice, and not added
piecemeal. Anything here needs its own design before it is implemented; do not
extend an existing code path to approximate it. The reasoning behind them is in
`dev/tmp/plan-keep-warm-admission-2026-09-16.md` and the investigations beside
it.

Items marked **(current)** describe today's code. Items marked **(design
boundary)** constrain the leasing redesign in that plan and apply to future work
too.

### Catalog and configuration are fixed during a leasing epoch (design boundary)

Publish catalog endpoints and global settings (gateway, UI, dynamic routing,
reverse proxy, image pins) **before** a workload starts acquiring and releasing
leases. Editing them while leases churn is not a supported operation.

- Today every command reloads the catalog and settings (current), so a change
  can take effect in the middle of a workload, including through an unrelated
  `release`.
- The redesign freezes them into a published profile that changes only through
  an explicit publication.

Hot catalog mutation, and merging a live catalog with a published one, are out
of scope.

### Admission is first-come, not fair (design boundary)

There is no FIFO order or reservation for a waiting request. A request that
needs several GPUs can be starved indefinitely by a stream of smaller requests
that each fit. Queue fairness would need a durable pending state and is out of
scope for the current leasing work.

### Admitted leases are never preempted (design boundary)

A deployment serving an active lease is never stopped, moved or displaced to
admit another request. Only idle keep-warm residency yields to demand.

### LIVE deployments are never moved between GPUs automatically (design boundary)

A deployment keeps the GPUs it was placed on for as long as it is live. If they
become unusable, it is reported as degraded, not re-placed. Any future migration
must be explicit and go through a GPU handoff barrier.

### Displaced keep-warm models are not re-warmed (design boundary)

A keep-warm deployment that loses its GPU to live demand, or whose container is
gone, is not restarted when capacity frees. It comes back only when a request
asks for it.

### Multi-node placement (current)

Placement is per host. Spanning one deployment across machines, or scheduling
across several hosts, is not supported.

### Forged ownership labels are outside the threat model (design boundary)

infer-stack identifies its containers by the Compose project and its own labels.
A container deliberately created with those labels is indistinguishable from a
managed one. See also *One control plane per host*.

### `observe()` is best-effort by contract (current)

`ComposeBackend.observe()` returns an empty set when Docker cannot be read, so
that `acquire` survives a stale compose file. It must not be used for a decision
that stops, removes or hands over a GPU. Use `ComposeBackend.residency()`, which
raises `ResidencyUnknown` instead. Changing `observe()` to be strict is out of
scope.

### A staged lease starts on the next ordinary apply (current)

`infer-stack acquire <alias> --no-apply` stages a lease: it enters the desired
state without starting anything. It is still part of that desired state, so the
**next ordinary apply by anyone** starts it as well. That includes a later
`acquire`, `release` or `infer-stack apply`. There is no "declared, but startable
only by an explicit apply" state, and adding one is out of scope.

### Known fault: idle keep-warm deployments can starve new leases (current)

An idle keep-warm deployment still claims GPUs during placement, ordered by
creation time rather than by demand. A new request can therefore wait out its
whole admission timeout while every GPU is physically empty. This is being fixed
in the leasing redesign.

**Workaround:** before a batch run, evict idle deployments it will not reuse:

```bash
infer-stack evict <alias> [<alias> ...]
```

### Known fault: the gateway can route a model's traffic to another container (current)

When a model's container is removed and another container receives its IP
address, the LiteLLM gateway can keep a pooled connection to that address and
send the removed model's requests to the other model indefinitely, while traffic
continues. This was reproduced: every request over 11 minutes was misrouted,
although Docker DNS was correct. The fix, stable per-service addresses, is in
the leasing redesign.

**Mitigations until then:**

- avoid recreating a `stop`-policy model seconds after another container was
  removed;
- after a misroute, stop traffic for about five minutes before retrying;
- setting `AIOHTTP_TTL_DNS_CACHE` low on the gateway narrows the window but does
  not close it.

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

## Only the gateway master key can be rotated (current)

`infer-stack secrets rotate` replaces `LITELLM_MASTER_KEY`. Not covered:

- **The Postgres password** (dynamic routing). Postgres stores it when its data
  directory is created, so changing the `.env` alone would lock the gateway
  out; `infer-stack env LITELLM_DB_PASSWORD=...` is refused once that directory
  exists. The database has no published port. Rotating it needs an
  `ALTER USER` inside the container first.
- **`LITELLM_SALT_KEY`**, by design. LiteLLM encrypts DB-stored credentials
  with it and cannot read them after it changes. The first rotation pins it to
  the pre-rotation master key, which is what LiteLLM had been using, and it
  never moves after that.
- **Open WebUI** may keep the old key in its persisted settings; update it
  under Admin > Settings > Connections.
- **Active leases** must be released first (or `--force`): their holders read
  the key once, at acquire.

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

### User config is authoritative; the recovery snapshot is internal

The primary workflow is `config init -> catalog suggest --apply -> acquire`.
There is no required `config publish` step in ordinary use. See
[ADR 0001](../adr/0001-user-config-is-authoritative.md).

For crash-safe recovery, leasing still persists a frozen render snapshot. The
controller advances it automatically on acquire under the publication lock:

- **Quiescent stack.** With no active lease and no managed deployment
  container, the next acquire adopts the current user settings and catalog
  wholesale.
- **Compatible catalog additions while live.** New endpoint/bundle/route
  definitions can be merged into the active snapshot without changing any
  definition already frozen for resident workloads. This is the normal
  "suggest/edit, then acquire another model" path.
- **Conflicting edits while live.** Redefining a frozen endpoint/bundle/route
  cannot be adopted into the same leasing epoch. Quiesce the managed stack
  (release/evict resident deployments) and retry; the next acquire adopts
  current config automatically.
- **Global render settings while live.** Gateway/UI/project/image-pin and other
  global render changes stay frozen for the active epoch. Compatible catalog
  additions may still proceed. Once the stack is quiescent, the next acquire
  adopts the global changes automatically.
- **Several runbooks' catalogs.** `infer-stack config publish a.yaml b.yaml`
  remains available as an advanced pre-seeding operation. A later runbook whose
  catalog is already a subset of that union does not compact sibling catalogs
  away.

`allowed_gpus` remains per caller rather than part of the recovery snapshot.
Changing backend kind (Compose to KubeAI or back) still requires tearing down
the old backend first; automatic snapshot advancement never crosses backend
kinds.


### A queued acquire holds nothing while it waits (current, by design)

With the Compose backend an acquire is admitted atomically: its lease and its
GPU allocations are committed together, or nothing is. A queued (`--queue`)
acquire that has not been admitted yet has written nothing, so it holds no GPU
and has no place in line (see "Admission is first-come" below).

Backends without strict residency (KubeAI, the null backend) keep the earlier
behaviour: the lease is committed first and placed by the render.

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

### Applies run one at a time and are not coalesced (current)

Every desired-state change renders and applies under one host-wide lock, so a
burst of N concurrent acquires runs N selective applies one after another. Each
caller can wait behind the others. An apply with nothing to change only reads
residency.

Measured on a host with per-shard leases (static gateway, two shards
concurrent): the lock hold for an acquire's apply was about 1 s, 6 s at worst,
and one apply per shard. At that scale serialisation is not what makes a run
slow — model load (minutes) and generation dominate. Dynamic routing, which
adds route reconciliation to the hold, and higher concurrency are not yet
measured.
If lock wait becomes the bottleneck, skip an apply whose render is identical to
the last successful one; do not reintroduce a separate apply lock.

### Recovery after an interrupted apply is a settle check, not full quiescence (current)

A timed-out or interrupted apply marks the pending change `interrupted`. Before
the next apply, the controller waits (about 60 s, plus at most one bounded
Docker query) until two consecutive
samples of the project's containers show the same ids and states and none is
`removing`. If the runtime cannot be read, or does not settle, the apply is
refused and the change stays pending.

This sees only container ids and states. Daemon work that has not yet changed
either (an image pull, a container create still in flight) is invisible to it.
Selective apply then treats what it finds by state: a `created` container is
replaced, a `removing` one aborts the apply (the change stays pending), and a
`restarting` one is left to Docker.

### Without the gateway, model host ports shift with the live set (current)

With `litellm` off, each model publishes a host port assigned by its position in
the live set. Adding or removing a model renumbers the others' ports, which
changes their fingerprints, so selective apply recreates them. Behind the
gateway (the default) upstreams publish no host port, and this does not happen.

### Leased engines keep `restart: unless-stopped` (current)

A crash-looping engine is detected from its restart count and its log rather
than by giving leased engines a restart budget (`on-failure:N`). A budget would
let the container die so it could be distinguished by state alone, but it also
stops a resident keep-warm model from recovering on its own after a transient
failure -- and that recovery is exactly what `classify_engine_log`'s transient
class protects: an unreachable hub or a truncated download is left to the
restart policy, however often it has restarted, while an unrecoverable error is
fatal on the first crash.

What remains blunt is the UNRECOGNISED crash, which still waits for two
restarts and is then declared fatal. A model that crashes for a reason no
signature covers, but that a restart would have fixed, is condemned. The
signature lists are the place to fix that, with evidence from a real log; if
that proves insufficient, weigh the restart budget against keep-warm residency
rather than changing it in isolation.

### `stack up` and `stack down` are raw Compose escape hatches (design boundary)

`infer-stack stack up`/`stack down` (and the TUI's Down button) run
`docker compose up -d --remove-orphans` / `down` directly. They bypass the
publication lock, ownership, selective apply and the GPU barrier, and they do
not touch leases. Use `infer-stack apply` (the TUI's Up button) to bring the
desired state up safely.

### The Docker target is the local default daemon (design boundary)

infer-stack does not inherit `DOCKER_HOST`, and runs every Docker command with
`DOCKER_CONTEXT=default`, which overrides a context selected with `docker
context use`. Every operation and every recovery therefore talks to the local
default daemon. Driving a remote or non-default daemon would need the target
recorded in the published profile, and is out of scope.

### Idle keep-warm residency is optional and never started (current, Compose)

With the Compose backend, an idle keep-warm deployment is a placement candidate
only while its container is uniquely resident. It keeps its GPUs while nothing
needs them, yields them to any admitted demand (it is then *displaced*, and its
container removed by the next apply), and is never started by infer-stack.
Displaced models are not re-warmed. This fixed the fault where idle residents
starved new leases.

- **Deployments from before allocations existed.** A LIVE deployment in such a
  ledger adopts the GPUs of its running container on the first render. If it
  has no single running container (for example a GPU reservation, which runs
  nothing), it stays *unresolved*: it keeps working, but no new GPU is
  allocated to anyone until its lease is released.
- **Other backends.** On backends without strict residency (KubeAI, null),
  idle keep-warm deployments stay in the desired set, as before. A failed
  acquire whose rollback cannot read residency can then leave a leaseless idle
  candidate that the next apply starts (workaround: `infer-stack evict
  <deployment>`).

### Known fault: the gateway can route a model's traffic to another container (until `network migrate`)

When a model's container is removed and another container receives its IP
address, the LiteLLM gateway can keep a pooled connection to that address and
send the removed model's requests to the other model indefinitely, while traffic
continues. This was reproduced: every request over 11 minutes was misrouted,
although Docker DNS was correct.

**Fix:** `infer-stack network migrate --subnet <cidr>` (while no leases are
active; it recreates every container once) puts every service on a fixed subnet
at an address no other service ever receives. `infer-stack network check`
reports a routing fault distinctly from "not ready". Stacks that have not been
migrated keep Docker's dynamic addresses.

**Mitigations on a stack that is not migrated:**

- avoid recreating a `stop`-policy model seconds after another container was
  removed;
- after a misroute, stop traffic for about five minutes before retrying;
- setting `AIOHTTP_TTL_DNS_CACHE` low on the gateway narrows the window but does
  not close it.

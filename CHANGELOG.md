# Changelog
We [keep a changelog](https://keepachangelog.com/en/1.0.0/).
We aim to adhere to [semantic versioning](https://semver.org/spec/v2.0.0.html).

### `infer-stack clean`: a clean slate, dry run by default

`clean` releases every active lease (whoever owns it), tears down every
deployment including keep-warm ones, and removes unmanaged containers in the
project. The gateway stays up. Like `git clean`, it only reports what it would
do unless given `-f`; `--no-orphans` leaves unmanaged containers alone and
`--json` prints the plan or the outcome. It composes `release --all --evict`
and `gc --orphans`, so it adds no new teardown path. The use case is a batch
scheduler with its own GPU accounting: a lease held outside it, manual or
keep-warm, occupies a GPU the scheduler believes is free, and the job it
places there cannot start.

### The gateway is its own module

The front door (LiteLLM config and service, route registry, dynamic-route
reconciliation, the managed keys, Open WebUI, the reverse proxy) moved out of
`leasing/compose.py` into `leasing/gateway.py`, with a `Gateway` object that
`ComposeBackend` owns (`backend.gateway`). The service naming rules moved to
`leasing/naming.py`. Backends now hand the gateway their route rows rather
than the gateway reading backend state. Rendered output is unchanged
(verified byte for byte), and `compose.py` is about a third smaller. Code
that imported gateway names from `infer_stack.leasing.compose` should import
them from `infer_stack.leasing.gateway`; the public naming helpers are still
importable from `compose`.

### The kubeai backend fails fast on an engine that cannot start

The crash diagnosis (restart count, exit code, and the engine log classified
as fatal, transient or unknown) was Docker-only, so on a cluster a model that
could never start held its lease for the whole timeout. It now lives in
`infer_stack/leasing/diagnosis.py` and both backends use it. The kubeai
backend reads pods through a new strict `residency()` (a kubectl failure
raises instead of reading as "nothing running") and quotes the crashed run's
log (`kubectl logs --previous`). A not-ready wait names the pod's reason, such
as `Unschedulable` or `ImagePullBackOff`. vLLM rejecting a flag (`error:
unrecognized arguments`) is now recognised as fatal on both backends; before,
it waited for two restarts.

`runtime.env` now works on the kubeai backend (the Model's `spec.env`, with
the same templates and reserved names). The served-name rule lives in one
place, `leasing.models.served_name`. Two gateway route builders fell back to
the deployment id where the engine used the first served alias, so a
deployment without `served_model_name` routed to a name its engine did not
serve.

`gc --orphans` and `network migrate` now refuse a non-compose backend
explicitly. They had used "the backend has `residency`" to mean compose.

### The kubeai backend puts the LiteLLM gateway in front of the cluster

On the kubeai backend a client had to name a model by its KubeAI Model name
(a DNS slug of the served name), not the endpoint alias. Cards send the
alias, so a card that ran on the compose backend got HTTP 404 on a cluster
(verified on k3s). The kubeai backend now runs the same LiteLLM gateway,
routing each alias to its Model: one `OPENAI_BASE_URL`, the managed key, and
the alias as the model name on both backends. `secrets rotate` works on it.
`--no-litellm` keeps the old direct access. New setting:
`kubeai_gateway_upstream`, for a gateway that cannot reach the cluster
Service's IP.

The profile-era KubeAI renderer (`infer_stack.backends.render_kubeai_artifacts`)
is removed. Nothing called it since the profile path was excised; the kubeai
backend is the one renderer for KubeAI.

`dev/kubeai_e2e.sh` now sends the alias as a card does, and fails when the
request fails. Before, a failed generation fell through to PASS: the check
sat in a `&&` list, where `set -e` does not apply.

### Custom container launches are catalog data, not recipes

`runtime.serve_recipe` is gone. An endpoint whose image has its own launcher
now describes it with generic fields: `command` (replaces the stock `vllm
serve` arguments), `env` (container environment, with `{max_model_len}`,
`{gpu_memory_utilization}`, `{served_model_name}` and `{port}` filled in) and
`mounts` (persisted under the runtime data dir). The renderer has no
model-specific branch. The HyperQwen suggestion is ordinary data, and its
long-context mode is an edit (`max_model_len: 150000`, `SPEC: mtp`,
`CTX: long`), not a new recipe. The TUI endpoint editor shows and edits the
image, command and environment, and keeps what it does not show.

Compatibility:

- An existing `serve_recipe: hyperqwen-3090-single` still works. It is read
  as the fields it meant and renders the identical container. Editing and
  saving the endpoint in the TUI rewrites it in the new form. Any other
  recipe name is an error, as before.
- `extra_args` are now deployment identity, so two endpoints that differ only
  in them no longer share one process. An endpoint with `extra_args` gets a
  new compatibility key: after upgrading, its next acquire starts a new
  deployment and an idle one left from before is displaced. Upgrade while
  nothing is leased to avoid a second copy of a model holding GPUs.
- `extra_args` that repeat a flag infer-stack sets from its own fields
  (`--served-model-name`, the parallel sizes, `--max-model-len`) are refused.
  Other repeats still work, with the extra value winning.

### `catalog show` suggests what you probably meant

`infer-stack catalog show NAME` for a name that is not in the catalog now
offers the likely intended entries, with their sections: `'gpt-oss' not
found. Did you mean: gpt-oss-20b (model, endpoint), gpt-oss-120b (model,
endpoint)?`. A name that is part of a longer one is matched by containment,
which plain edit distance misses; typos are matched at a 0.75 similarity
cutoff. `catalog endpoint show`, `catalog model show` and the unknown-endpoint
error from `acquire` use the same matching.

### The TUI's Logs tab shows download progress as it happens

A model's first start downloads weights with a progress bar that redraws one
line with `\r`. Docker's log driver stores output line by line and holds a
partial line until its newline, so both `docker compose logs -f` and
`docker logs -f` showed nothing for many minutes and then everything at once.
The Logs tab now reads each container's recent history with `docker logs
--tail`, then its live output with `docker attach --no-stdin
--sig-proxy=false` (ending that never touches the container). It shows at
most one redraw every 2 s. Containers created while the tab is open are
picked up within 3 s, and a restarted container is followed again.
`infer-stack logs -f` still goes through Compose and still has the delay.

### A first-use image pull shows its progress

An acquire whose engine image was not yet on the host sat silent for as long
as the download took (20 minutes on a real host), because `docker compose up`
pulled it with its output captured. Missing images are now pulled before
`up`, with a line each time a layer finishes: "pulling IMAGE: 7 of 23 layers
downloaded, 4.1 GB of 9.8 GB" (sizes from the registry manifest). It goes to
the log and, in the TUI, to the status bar and the TUI log tab. The pull also
runs before any container is removed, so a failed pull changes nothing.

### The TUI log shows the CLI command for each action

Acquire, release, evict, apply, stack down, and the catalog edits (add model,
add or edit an endpoint, remove, suggest) now log the equivalent
`infer-stack ...` command in the TUI log tab. Tests check that each one parses
and that the logged `catalog endpoint add` writes the same entry the TUI did.
The TUI's cleanup action has no CLI counterpart yet, and its log says so.

### The HyperQwen suggestion is offered wherever it fits

`catalog suggest` offered `qwen3.8-27b-dbirks-hyperqwen` only on cards named
RTX 3090, and pinned it to that card. It now appears on any Ampere-or-newer GPU
with 24 GiB, placed like any other model. Measured unchanged on an RTX PRO 6000
Blackwell (sm120): ~300 tok/s single stream against ~27 for the bf16 base, and
4 concurrent requests batched without queueing. Pre-Ampere cards are still
excluded (new pool field `requires_ampere`).

### `infer-stack secrets rotate` replaces the gateway's master key

It writes a new `LITELLM_MASTER_KEY`, restarts the gateway with it through a
normal publication, and checks that the new key is accepted and the old one
rejected. Refused while leases are active unless `--force`.

LiteLLM encrypts the routes dynamic routing stores in Postgres with
`LITELLM_SALT_KEY`, falling back to the master key when that is unset, so
changing only the key would have made them unreadable. The first rotation pins
the salt to the old key. `env LITELLM_MASTER_KEY=...` now does the same and
rejects a key without the `sk-` prefix (previously it was silently replaced on
the next render). `env LITELLM_DB_PASSWORD=...` is refused once Postgres has
initialised its data directory, because Postgres keeps the password it was
created with.

### The TUI's engines log view no longer shows the gateway

"(engines — no litellm)" streamed litellm lines. Three causes, each enough:
the selector was built with "(all services)" as its value, so mounting it
switched the stream to every service; each refresh that saw the service list
change rebuilt the options, which posted a change carrying `''` -- the value
of "(all services)"; and a view opened before any engine existed fell back to
every service and kept that stream once engines appeared. Refreshes no longer
count as a selection, the engines view follows services as they come and go,
and with no engine it says so instead of showing the gateway.

### An interrupted readiness wait releases its lease

Ctrl-C during `acquire`/`run`'s readiness wait left the lease ACTIVE and its
deployment LIVE. Nothing released it, so the GPU stayed claimed, `evict` could
not reclaim it (eviction is for *idle* deployments), and because the stack was
no longer quiescent every later acquire was refused for redefining an endpoint
frozen by the leasing epoch -- until the TTL expired. Measured on a real host:
one interrupted `run` blocked all leasing for 22 minutes. An interrupted wait is
an acquire that did not deliver, so it now rolls back exactly like a timeout.

### The VRAM floor stops double-counting duplicate weight sets

A repository that ships the served weights plus a second complete copy --
`original/`, `metal/`, `consolidated/`, or in-tree GGUF/MLX variants -- had
every copy added together. Measured on a host: a root quantised set and a
same-sized `original/` bf16 set reported 121.5 GiB for a model that serves in
60.8 GiB, which exceeded every GPU and made it permanently unplaceable ("the
pool can never satisfy that", so waiting could not help either).

The engine loads one set of weights, so the floor is now the largest set, not
the total: weight files are grouped by directory within the snapshot and by
format family, and the largest group wins. This is the rule the floor already
applied across snapshots, applied within one. Sharded weights in one directory
are still one set and still add up.

**A recorded measurement now also beats the floor.** `infer-stack measure`
records the real footprint of an exact serve, and clamping that with a
heuristic lower bound could only make a model that demonstrably ran look
unplaceable. A hand-declared `min_vram_gib` is a guess and is still clamped,
which is what the floor is for.

### Redefining an endpoint nothing is running no longer blocks the host

Reported from a real host: two endpoints were re-registered with
`catalog endpoint add --force` while unrelated keep-warm deployments were
resident, and afterwards every acquire was refused until someone ran
`evict --all`.

Only definitions a resident workload is actually running now stay frozen.
When the current catalog collides with the recovery snapshot, the snapshot is
first pruned to the endpoints and bundles that LIVE deployments, and idle
keep-warm deployments whose container is resident, are running. Redefining
anything else simply replaces the stale definition. A collision that does
matter names the endpoint and what to free (`infer-stack evict <alias>`)
instead of asking for the whole stack to be quiesced.

### A transient engine failure still fails fast if nothing will retry it

`classify_engine_log` spares an engine whose log shows an unreachable hub or
a truncated download, because `restart: unless-stopped` will try again. If
nothing will try again -- the container exited under `restart: no`, or an
`on-failure` budget is spent -- there is no next attempt to wait for, so the
lease is released at once with the cause reported. Docker's own restart policy
and retry cap, now carried on the residency snapshot, decide which case it is.

### An engine that cannot start fails fast, with its own error

Reported from a real run: a model whose architecture the vLLM build does not
implement exits at once, `restart: unless-stopped` restarts it forever, and
`observe()` counts only *running* services -- so the acquire held its GPU for
the whole 1800 s timeout, and the engine's error never reached the operator.

- **Readiness can now be fatal.** A probe consults Docker's own bookkeeping:
  a container restarted twice, or exited non-zero and not restarting, is not
  loading. `Readiness.fatal` ends the wait immediately, so the GPU is released
  in seconds rather than at the timeout.
- **The engine's own words are reported.** The diagnosis quotes the last log
  lines and names a likely cause when it recognises one: missing
  `trust_remote_code`, an unimplemented architecture, an unreadable config, a
  gated repository (set `HF_TOKEN`), or CUDA OOM.
- **`acquire`, `run` and `measure`** print it, and `acquire --json` carries a
  `failures` list.
- **The engine's log decides, not the restart count.** `classify_engine_log`
  separates errors a restart can never fix -- a rejected config, an
  unimplemented architecture, a gated repo, CUDA OOM -- from ones it can: an
  unreachable hub, a truncated download, a port still held by the container
  being replaced. An unrecoverable error is fatal on the FIRST crash; a
  transient one is never fatal, which is what `restart: unless-stopped` is for;
  an unrecognised crash keeps the blunt two-restart budget.
- Unreadable Docker, a container still starting, a clean exit, or a single
  unrecognised restart are all left alone: they may still be loading.

### Refused TUI actions pop up instead of doing nothing

An action the TUI declines -- `Edit` on an endpoint that is actively served,
`Acquire` with nothing selected, a catalog action without a catalog path --
now raises a popup, keeps its message in the status line, and is recorded in
the `TUI log` tab. Previously it wrote one status line that the next refresh
tick wiped, so a refusal was indistinguishable from a dead button. This covers
all 24 refusal paths; the ones that report a failure are logged as errors.

### The TUI header shows the running version

The header now reads `infer-stack` / `0.7.1 · leasing dashboard`, so a
screenshot or a bug report says which version produced it.

### The TUI reports its own failures, in a `TUI log` tab of its own

A failing action used to look like nothing happened: Textual runs actions and
background workers on its own message pump, so an exception reached no
terminal, and the TUI had no handler for worker failures.

- **A `TUI log` tab**, separate from the docker Logs pane (which streams
  container output and sits inside a collapsed section). It records what the
  TUI did, every status message, and every failure with its traceback, and it
  names the file each error is also appended to.
- **An error is impossible to miss:** the tab label turns red with a count
  (`⚠ TUI log (2)`), a toast appears, and the status line keeps the message
  instead of being wiped by the next refresh. Press `l` for the tab.
- **`<data root>/tui-errors.log`** keeps tracebacks for after the TUI exits.
- This covers button handlers and all 16 background workers (the catalog
  editor, acquire, release, the API probes). A worker cancelled because
  another action in its group started now says so, instead of looking dead.
- An action that declines to act (no catalog path, endpoint actively served)
  logs that reason, so "nothing happened" is never the whole story.

### Restore the three-step leasing UX; isolate TUI log streams

- **User config is authoritative again.** The normal workflow is explicitly
  `config init -> catalog suggest --apply -> acquire`; the persisted leasing
  profile remains only an internal crash-recovery snapshot. Acquire advances
  that snapshot automatically instead of requiring a separate `config publish`
  step.
- **Compatible catalog additions can join a live epoch.** While models are
  resident, previously frozen definitions and global render settings remain
  stable, but new non-conflicting catalog definitions are merged under the
  publication lock. Conflicting redefinitions wait for release/eviction; once
  quiescent, the next acquire adopts current user config wholesale.
- **Explicit `config publish` is advanced-only.** It remains useful to pre-seed
  several runbook catalogs or preview/pre-pull a profile, but it is no longer a
  required configuration surface in ordinary operation. ADR 0001 records the
  invariant.
- **TUI named-service logs are generation-isolated.** Switching from LiteLLM to
  a specific vLLM service invalidates the previous follower before clearing the
  pane. Buffered lines from the terminated worker are discarded, fixing stale
  LiteLLM output leaking into a vLLM-scoped log view.

### TUI GPU pinning and Qwen3.8-27B on a 3090

- **Endpoint GPU affinity is part of Edit endpoint in the TUI.** vLLM endpoints
  can opt into an exact `placement.gpu_indices` override there, while `auto`
  keeps the existing VRAM-aware scheduler. Pins are validated against tp×pp×dp,
  shown in the endpoint table, and kept distinct in deployment compatibility;
  there is no separate GPU-pin action to learn.
- **`suggest from my GPUs` recognizes the measured Qwen3.8-27B/RTX 3090 path.**
  The official `Qwen/Qwen3.8-27B` metadata supplies the 262K context/27.78B
  model identity; the runnable source is the ~19.5-GB W4A16 AutoRound body that
  HyperQwen prepares for a 24-GiB 3090. The suggestion pins the matched 3090,
  uses HyperQwen's immutable per-commit image, and starts its speed-first
  DFlash2 + prefix-cache single-user profile at 64K context. Its catalog name
  is `qwen3.8-27b-dbirks-hyperqwen`; the unsuffixed name is reserved for the
  official checkpoint rather than conflating the two artifacts.
- **Endpoint activation is reliable from mouse, keyboard, and Acquire.** The
  endpoint table uses Textual's native click chain when available and a
  same-endpoint timing fallback when a driver does not provide one; neither path
  reconstructs rows from screen coordinates. Enter and the Acquire button use
  the same acquire action. Mutation results stay visible through the immediate
  table refresh instead of being replaced by a row-relationship hint.
- **Endpoint Edit stays usable on short terminals.** The editable fields scroll
  inside the modal while Save/Cancel stay fixed and visible, including after the
  GPU-placement field was added.
- **Named vLLM serve recipes are explicit and compatibility-safe.** The
  `hyperqwen-3090-single` recipe owns its nonstandard entrypoint, persistent
  prepared-model/cache mounts, and launch environment. Stock vLLM endpoints
  keep their existing structural compatibility shape; KubeAI refuses the
  Compose-only recipe rather than silently rendering the wrong engine command.
- **HyperQwen is credited as related work.** The README names the upstream
  project and makes the ownership boundary explicit: HyperQwen provides the
  3090-specific preparation/runtime tuning; infer-stack provides placement and
  lifecycle integration around it.

### Publication side effects, `.env` races, raw controls (review)

- **`config publish` previews purely.** On Compose it now previews the
  candidate in memory, commits the profile and its approved digest, and only
  then renders for real. A crash before the commit can no longer leave
  candidate routes in the append-only route registry, or candidate addresses
  in the address table.
- **`infer-stack env KEY=VALUE`** now takes the publication lock and replaces
  the file atomically. A render in progress sees one consistent `.env`, and
  concurrent writers no longer lose each other's keys.
- **Fingerprints** hash only the `.env` variables a service interpolates, so
  an unrelated key no longer recreates it.
- **The TUI's Up button** now runs `apply` through the controller, instead of
  a raw `docker compose up --remove-orphans`. `stack up`/`stack down` are
  documented as raw escape hatches.
- **`status` and the TUI's served-endpoint view** show TTL expiry virtually.
- **A subnet change** checks for foreign containers attached to the old
  network before removing anything, so it cannot take the stack down and then
  abort.

### Approval digests and subnet changes (review)

- **`config publish`** writes the profile and its pending marker (with the
  approved digest) in one transaction. A crash can no longer leave a published
  profile whose approval nothing records.
- **The approved digest now always describes the pending state.**
  - `acquire --no-apply` records none, so a staged lease stays discardable by
    `release`.
  - A rollback of a failed acquire drops the digest of the state it abandons.
  - An acquire's digest is written in the same transaction as its lease.
- **Changing an already-migrated subnet** now recreates the Docker network.
  Compose never changes an existing network's IPAM, so once the old containers
  are confirmed gone the apply removes `infer-stack-net`, and Compose recreates
  it on the new subnet. A container still attached blocks it, with the change
  left pending. This is checked on every apply, so an interrupted migration
  completes later.

### Migration and publication fixes (review)

- **Unresolved legacy deployments.** A LIVE deployment from before allocations
  that is still unresolved can no longer accept a new lease by coalescing; the
  acquire is refused, naming it. Renewing its existing lease still works.
- **`network migrate`** previews the migrated render and takes approval before
  writing anything. It then switches the subnet, resets the address table and
  marks the change pending in one transaction, with the approved digest. A
  crash can no longer leave the old subnet with an empty address table, which
  could have given a service another service's address.
- **Crash-safe acquire approval.** The approved render's digest is now stored
  in the pending marker before the lease commits, so a recovery after a crash
  (or an upgrade) cannot silently apply a different render. The digest is
  dropped once the approved render reaches Docker, so a route-verification
  retry does not need re-approval.
- **`config publish` pre-pull** derives its image set from the candidate
  profile and its catalogs: infrastructure images the candidate enables, the
  engine image of every published endpoint, and per-endpoint image overrides
  (including Ollama).
- **Re-running `network migrate`** on its current subnet no longer reports the
  host route of infer-stack's own network as an overlap.
- **Ledger transactions** also roll back on `KeyboardInterrupt`.

### Health view: degraded, displaced, unresolved, orphans, pending changes (P10)

- **`infer-stack leases`** now ends with a `health:` block, and `--json`
  includes a `health` object. It is read-only: it never writes the ledger. It
  reports:
  - the pending change (staged or apply requested, interrupted, approval
    guard);
  - unknown residency;
  - each deployment's condition: `ambiguous`, `degraded`, `displaced`,
    `unresolved` or `not-running`;
  - orphan containers;
  - profile drift;
  - the stable address table;
  - leases that have expired but not yet been reclaimed.
- **Degraded deployments end to end.** A LIVE deployment whose committed GPUs
  are no longer valid is reported `degraded`, is neither started nor removed,
  and refuses coalescing; releasing its lease makes it removable.
- **Refusals name the contested GPUs.** An admission refusal lists which
  admitted deployments hold which GPUs, and their owners.
- **GPU column.** In admission mode, `leases` shows committed allocations and
  resident GPUs rather than a hypothetical legacy plan.

### `config publish` pre-pulls images; approved renders are guarded (P4)

- **Pre-pull.** `config publish` pulls every image the new profile references
  before publishing, outside the lock, so steady-state applies never wait on a
  registry. A failed pull publishes nothing; `--no-pull` skips the step.
- **Approved-digest guard.** A publication records the digest of the render
  the operator approved. If a later process renders something different
  before that change is applied (for example infer-stack was upgraded in
  between), the apply is refused and the change stays pending until
  `infer-stack apply` re-approves it explicitly.

### Selective apply waits for health and for removals (review)

- **Health-conditioned dependencies.** `up --no-deps` skips Compose's
  `depends_on: condition: service_healthy`, so selective apply now waits
  (bounded, 180 s) for such a dependency, for example Postgres before the
  dynamic-routing gateway, to report healthy before starting its dependent,
  and aborts with the change pending otherwise.
- **Removals are confirmed.** After removing containers, the apply confirms
  from strict residency that they are gone (bounded, 60 s) before starting
  anything on their GPUs or addresses.

### Stable per-service addresses: `infer-stack network migrate` and `network check` (P7)

- **`network migrate --subnet <cidr>`.** Puts the project on one named
  network with a fixed IPAM subnet, and every service on a static address.
  - Addresses live in an append-only ledger table: a service keeps its address
    across recreation, and no other service ever receives it. Allocation skips
    the network, gateway (`.1`) and broadcast addresses.
  - A subnet overlapping an existing Docker network or host route is rejected.
  - Every container is recreated once, so the command is refused while leases
    are active unless `--force`.
  - This fixes the reproduced gateway misroute after container recreation.
- **Address holders.** An unmanaged container holding a service's address
  aborts the apply.
- **`network check`.** Probes each model upstream by name from inside the
  gateway (`docker exec` with `python3`), and reports `healthy`, `not-ready` or
  `routing-fault`. It exits 4 on a routing fault.

### Selective apply: ownership labels, fingerprints, GPU barrier (P8)

The Compose backend no longer runs `docker compose up -d --remove-orphans`.

- **Labels.** Every rendered service carries `infer-stack.service` and a
  behavioural `infer-stack.fingerprint`. The fingerprint covers the canonical
  stanza, the generated files the service mounts, and the managed `.env` when
  the stanza interpolates from it.
- **What an apply does.**
  - It keeps a managed container whose (service, fingerprint) is wanted, is
    unique, and is running, restarting or paused. A required paused container
    is unpaused; an optional one stays paused.
  - It removes other managed containers, except those of degraded deployments.
  - It reports containers it does not manage (orphans), and never removes them
    implicitly.
  - It starts only missing, non-optional services, with `up -d --no-deps`, level
    by level in dependency order.
- **GPU barrier.** Nothing is started on a GPU while another container holds
  it. A managed occupant is removed first; an unmanaged or degraded one aborts
  the apply (the change stays pending). So do duplicate containers for a wanted
  service, and a container still being removed.
- **Upgrade.** Containers from before the labels are adopted once, when they
  match infrastructure in the render or a LIVE/resident deployment on the same
  GPUs. Adoption recreates nothing.
- **Orphans.** `infer-stack gc --orphans` lists unmanaged project containers
  and removes exactly those, after confirmation or with `--yes`.
- **Residency** now lists the whole project, infrastructure included, with each
  container's service, fingerprint and ownership.

### Admission fixes (review): approval before commit; unresolved rows are never placed

- **The diff is approved before anything is committed.** The admission preview
  renders exactly the files the post-commit render will write and asks for
  approval then; the render after the commit does not ask again. A declined
  acquire leaves no lease, no added alias on a shared deployment, and no route
  registry change. The registry is now persisted only after approval.
- **Every claimed deployment is checked.** Admission refuses a candidate if any
  deployment it claims is unplaced or unrenderable, including an existing one
  it only coalesces onto.
- **Unresolved legacy deployments are never placed.** A LIVE deployment from
  before allocations, with no unique running container, is neither freshly
  placed nor given an allocation; it only blocks new allocations until
  released. Crash recovery in admission mode no longer commits allocations.
- **Adoption now runs.** Migration adoption of pre-label containers was
  unreachable (it sat after a `return`), and runs now.

### Admission: committed allocations, atomic acquires, optional warm residency

For the Compose backend, this fixes the incident where idle keep-warm models
starved new leases while GPUs sat empty. It covers plan steps P5, P6 and P9.

- **Hard allocations.** A LIVE deployment holds a committed allocation, the new
  `deployments.assigned_gpus` column, added to existing ledgers automatically.
  The allocation is released in the same transaction as any transition out of
  LIVE.
- **Atomic admission.** An acquire is previewed in memory, both placement and
  render, against the ledger, strict residency and the published profile. The
  lease and its allocations are committed together, or nothing is written.
  - A queued caller that is not yet admitted holds nothing.
  - Two callers racing for the last GPU cannot both win.
  - A ledger change between preview and commit makes the acquire retry.
  - While Docker residency is unknown, only requests needing no new GPU are
    admitted.
- **Optional warm residency.** An idle keep-warm deployment is only a
  candidate while its container is uniquely resident. It yields its GPUs to
  demand (reported as `displaced`), and is never started.
- **Reuse.** Making an idle deployment LIVE again adopts its resident
  container's GPUs, or places it fresh.
- **Renew.** When every deployment is already LIVE, renew is lock-free and
  TTL-only. Otherwise the lease is re-admitted under the lock, and the renew
  fails explicitly if that is impossible.
- **Upgrade.** LIVE deployments without an allocation adopt their running
  container's GPUs. Ones that cannot (for example GPU reservations) stay
  unresolved and block new allocations until released.
- **Planner.** In admission mode it ignores soft sidecar pins, so a new
  placement stays within `--allowed-gpus`.
- **CLI catalogs.** A `--catalog` (or `INFER_STACK_CATALOG`) the caller names
  that is missing or invalid is an error, even once a catalog union is
  published.
- **`config publish` refuses to change the backend kind.**

### Profile publication fixes (review)

- **`config publish` on a fresh ledger** no longer freezes the invocation's
  settings first. A declined or failed preview now leaves no profile and no
  pending marker.
- **A declined recovery render keeps the crashed acquire's GPU scope.**
  Previously the scope was cleared anyway, so a later caller could place that
  deployment within its own GPUs.
- **The CLI refuses a catalog that is not one of the published sources** (new,
  or edited since publishing), instead of resolving names against the old
  published definitions.
- **KubeAI** now freezes its catalog union into the profile too.
- **Switching backends:** a profile for another backend kind is reported on the
  first mutation rather than when the controller opens, so
  `config publish` can switch backends while quiescent.

### Planner admission mode (keywords only; no caller uses it yet)

`plan_placement` accepts `required_ids`, `hard` and `optional_hints`. Committed
allocations are validated against the whole physical pool; an invalid one is
reported in `degraded` and never re-placed. Required deployments are placed
next. Optional idle residents keep their GPUs only if still free, and are
otherwise reported in `displaced`; they are never newly placed. Without the
keywords, plans are unchanged. This is step P3 of the admission plan.

### Renders use a published profile; `infer-stack config publish`

Lease operations no longer render from each caller's flags and settings.

- **Frozen on first use.** The first mutation against a ledger freezes a
  profile. It holds backend, project, gateway, UI, dynamic routing,
  display-GPU policy, reverse proxy (a BYO nginx config is snapshotted by
  content), image pins, ports, state paths, and the catalog union. Later
  operations, including recoveries by other processes, render from it, and
  warn once when their own settings differ.
- **Acquire checks the published catalog.** An endpoint missing from the
  published catalog union, or defined differently there, is refused before
  anything is written, naming `config publish`. The CLI resolves endpoint names
  from the published union, so a runbook can acquire any published endpoint
  whatever `--catalog` it passes.
- **`infer-stack config publish [catalog ...]`** replaces the profile, previews
  the diff, and publishes. It merges several catalogs into one union
  (identical definitions deduplicated, conflicting ones refused), and works
  only while no lease is active and no deployment container exists.
- **`--allowed-gpus` stays per caller.** An acquire stores it with its pending
  change. If the acquire dies before its first render, the next operation
  places that deployment within the original caller's GPUs.
- **Upgrading.** On an existing host the next lease operation freezes the
  current settings and catalog. If runbooks use different catalogs, run
  `infer-stack config publish` with all of them while the stack is idle.

### Docker commands run with an explicit environment

Every Docker command infer-stack runs now gets an allow-listed environment:
`PATH`, `HOME`, `USER`, `LOGNAME`, locale, `TMPDIR`, `XDG_RUNTIME_DIR` and
`TERM`. This covers the backend, the `stack` commands, and the TUI's runner and
log follower. It affects existing setups:

- **`HF_TOKEN`.** An exported `HF_TOKEN` no longer reaches Compose.
  `${HF_TOKEN:-}` resolves only from the managed `.env`; set it with
  `infer-stack env HF_TOKEN=hf_...`.
- **Docker daemon.** `DOCKER_HOST` is not inherited and `DOCKER_CONTEXT` is
  forced to `default`, so neither a caller's shell nor `docker context use` can
  point one operation at a different daemon.
- **`stack` commands** now pass the managed `.env` as `--env-file`, as the
  backend does.
- **TUI.** Its Docker runner is now the same bounded runner as the CLI, instead
  of an unbounded `subprocess.run`.

### `routes seed` and `routes prune` publish through the controller

Both commands used to write the route registry and then reconcile as two
separate steps, outside the controller's lock. They now run as one serialised
mutation through `Controller.publish_change`. `prune` still previews and
confirms outside the lock. Under the lock it recomputes, and drops only routes
that were confirmed and are still unneeded. Both report `publication_pending`
in `--json`, and exit 3 if the apply did not fully take effect.

### Read-only views no longer write the ledger

`leases`, `wait`, `measure`, `evict`'s target lookup, `apply --wait`'s view,
route annotation and TUI polling used to call `sweep()`, which writes TTL
expiry into the ledger outside the controller's lock. They now read
`Ledger.status(virtual_expiry=True)` instead. It reports a lease past its TTL
as `expired`, and a deployment left without a protecting lease as `idle`,
without writing either. Expiry is recorded only by controller operations,
including `gc`.

### Releases and cleanup go through the controller

`infer-stack release` (single, `--all`, `--evict`) and the TUI's release,
release-all and cleanup actions now use `Controller.release_leases` and
`Controller.prune`. A batch still releases in one publication, so there is at
most one diff prompt. The CLI and TUI previously changed the ledger directly and
reconciled afterwards, outside the lock. `release` now reports
`publication_pending` in `--json`, and exits 3 if the apply did not fully take
effect.

### `renew` goes through the controller

`infer-stack renew` now calls `Controller.renew`, under the host-wide lock. A
renew that makes an idle deployment LIVE again is a desired-state change: it
takes the publication marker and applies. A TTL-only renew takes no marker and
runs no apply. It could previously run concurrently with another process's
render and apply. The CLI prints the revived deployments, and exits 3 if their
apply did not fully take effect.

### Failed applies no longer leak leases or overlap daemon work

Fixes found in review of the serialised-publication change:

- **No hidden lease on a failed acquire.** If Docker fails or times out while
  an acquire is applying, the lease is released in the ledger and the error is
  re-raised. The rollback re-renders but does not apply again, since runtime
  state is unknown; the release stays pending.
- **`--no-apply` never applies**, even when an earlier operation left an apply
  pending. This covers both the acquire and its rollback.
- **Settle before retrying.** An interrupted apply marks the pending change
  `interrupted`. The next apply first waits, bounded, for the project's
  containers to stop changing, and refuses (`RuntimeUnsettled`) if they do not.
- **Rollback eviction uses strict residency.** It no longer uses `observe()`, so
  a Docker error during rollback can no longer evict a warm keep-warm
  deployment. Only a deployment with definitely no container is evicted.
- **`infer-stack apply` reports pending changes.** It exits 3, and reports
  `publication_pending` in `--json`, when the apply did not fully take effect.
- **Route failures are retried.** Reconciliation re-diffs and retries failed
  admin-API calls until the route set verifies or its deadline passes.
- **Ctrl-C stops Docker too.** An interrupted Docker command now has its
  process group killed; it runs in its own session, so it previously kept
  running.

### Render and apply are serialised under one lock, gated by a durable pending marker

Every desired-state change (acquire, release, evict, gc, rollback, `apply`) now
runs as one publication under the controller's host-wide lock. It records a
`publication_pending` marker before touching the ledger, mutates the ledger,
renders, applies that exact render, and clears the marker only once the whole
apply has succeeded. In dynamic-routing mode that includes verified gateway
routes.

This replaces the separate apply lock and the generation-based coalescing. There,
a render could rewrite the compose file while another process's `docker compose
up` was reading it, and an apply that failed or was killed left no record. Now:

- **Crashes and failures stay pending.** A crash, a Docker timeout, or routes
  that do not verify leave the marker set. The next applying operation, or
  `infer-stack apply`, re-renders and applies the whole pending desired state.
- **Compose `apply()` returns whether it fully took effect.** Route
  reconciliation gets 20 s when the gateway was already running and 180 s when
  the apply is bringing it up. `ReconcileResult.publication_pending` reports a
  change that is still pending.
- **Staged leases stay staged.** `acquire --no-apply` and `infer-stack render`
  never apply. A later applying operation still applies the whole pending
  state, including staged leases, as before.
- **Applies no longer run concurrently.** They are also no longer coalesced:
  concurrent acquires each apply in turn. Each apply is bounded (see below), and
  one is a no-op when nothing changed.

The legacy `desired_gen`/`applied_gen` counters are still maintained but no
longer read. This is part of step P2 of
`dev/tmp/plan-keep-warm-admission-2026-09-16.md`.

### Docker commands are time-bounded; route reconciliation is deadline-bound and reports success

Every Docker command run by the Compose backend now has a wall-clock bound
(queries 60 s, stop/rm/exec 300 s, compose up/down 1800 s, pulls 3600 s). On
expiry the command's whole process group is killed and `BackendTimeout` is
raised, rather than the call hanging. These calls will run under the
controller's host-wide lock (plan step P2), where an unbounded command would
block every other lease operation.

Dynamic-routing reconciliation now works against one wall-clock budget covering
listing retries, POSTs and verification, instead of 90 listing attempts whose
per-request timeouts were not counted. The worst case against an unreachable
gateway drops from roughly 18 minutes to 180 seconds. When routes change,
reconciliation lists them again to verify, and `_reconcile_routes` returns
whether the managed route set matches the desired one. Callers still treat it as
best-effort for now; gating on the result is part of P2.

### Strict residency: which deployment containers exist, and on which GPUs

`ComposeBackend.residency()` returns a strict snapshot of this project's
deployment containers, found by the `infer-stack.deployment` label in every
state, with GPUs read from each container's actual device reservation
(`HostConfig.DeviceRequests`). It is additive: nothing calls it yet.

It exists because `observe()` is the wrong tool for any decision that stops,
removes or hands over a GPU. `observe()` returns an empty set when Docker cannot
be read -- deliberately, so acquire never bricks on a stale compose file -- and it
maps containers through the render sidecar, which describes what was rendered
rather than what exists. `residency()` has the opposite contract: a Docker
failure raises `ResidencyUnknown` and is never "nothing running"; a deployment
with more than one container is reported as ambiguous with every container kept;
and a reservation that cannot be mapped to GPU indices is treated as occupying
every GPU rather than guessed. `observe()` is unchanged.

This is step P1 of `dev/tmp/plan-keep-warm-admission-2026-09-16.md`.

### `env` answers the front door, not just the secrets

`infer-stack env OPENAI_BASE_URL` and `infer-stack env LITELLM_PORT` now work,
so everything a client needs comes from one verb:

```bash
export OPENAI_BASE_URL=$(infer-stack env OPENAI_BASE_URL)
export OPENAI_API_KEY=$(infer-stack env LITELLM_MASTER_KEY)
```

Before this, `env` knew only what was written in the managed `.env`, which is
secrets. The URL was obtainable only from a lease env-file -- so every script
that wanted both ended up looking the key up properly and hardcoding
`http://127.0.0.1:14042/v1` beside it. A hardcoded port is wrong exactly when
it matters (a second stack, a moved front door) and it is wrong silently.

Both keys are DERIVED, not stored, and they answer before any `acquire`,
because a URL needs no secret to exist. They come from `_front_door`, the same
resolution `infer-stack test` uses, so a script built from `env` cannot point
at a door `test` never knocked on.

A stored value still wins: `infer-stack env OPENAI_BASE_URL=https://gw:8443/v1`
pins it, which is how you aim a script at a gateway that is not the local front
door. `LITELLM_PORT` is then read back off that URL rather than re-derived --
a port disagreeing with the URL printed beside it is worse than no port -- and
is omitted entirely when the effective URL names none, since behind a proxy on
80/443 there is nothing to report and a made-up number is one a script would
bake in.

### `doctor --gpu` / `--sudo`: why a card looks busy, and who holds it

Twice a GPU has read 100% utilization with ~0 MiB allocated and an empty
process table, and twice the diagnosis took hours. The tools mislead in a
specific way: unprivileged `lsof`/`fuser` see only the caller's own processes,
so they report "nothing holds it" while `nvidia-smi -r` answers `In use by
another client`. Every containerd shim and Kubernetes pod is owned by root.

`--gpu` samples utilization over several seconds (it is a windowed average, so
one reading after a process exits proves nothing) and flags a card that is busy
with nothing allocated. `--sudo` runs the holder scan as root and maps each pid
to its cgroup, which is what names the container or pod — `pid 9030` is not an
answer, `gpu-feature-discovery in kubernetes pod 00397bb3…` is.

Without `--sudo` the holder check reports **not checked**, never "clear". A
false all-clear is what led to recommending a reset that could not succeed.

`nvidia-persistenced` holding every device is treated as expected, but named,
because it is why `nvidia-smi -pm 0` does not let a reset through: that turns
the mode off and leaves the daemon holding its handles.

Nothing here resets or kills. The same symptom with a different holder means
something else, and both times the card computed fine — the gauge was cosmetic.

### An impossible lease fails instead of queueing

`acquire --wait-for-placement` queued whenever a deployment was unplaced,
without asking whether the request could be satisfied at all. A lease whose
deployments cannot fit *together* — a `tensor_parallel_size: 4` answerer plus a
1-GPU extractor on a 4-GPU host — waited out the full 1800s timeout holding
whatever it had already placed, so a request that could never succeed blocked
the ones that could.

The planner's permanent-failure branch does not catch this: each deployment is
placeable on its own, and only the set is impossible.

`acquire` now re-plans the lease's deployments alone on an idle host before it
queues (`ComposeBackend.plan_on_idle_host`). If they do not fit *there*,
waiting cannot help, so the lease is rolled back and `PlacementError` is raised
at once. Backends without the method (null, kubeai) skip the check and queue
exactly as before — it can turn a hang into an error, never the reverse.

"Idle" means free of everything *unrelated* to the request, not empty. Pins of
the requested deployments are kept; every other pin is dropped. Under Slurm a
requested deployment may already be running on a GPU outside this call's
`allowed_gpus` — a shared extractor another job started — and that is reusable
as it stands. Dropping its pin would force it back inside our own slice, count
it against our budget, and reject a lease that was only waiting for a card to
free.

### Tensor-parallel deployments can place again

`min_vram_per_gpu` returned the weight-bytes floor unchanged, but that floor is
a WHOLE-MODEL figure while the function's contract is per-GPU. A
tensor-parallel deployment therefore demanded the entire model on each of its
cards — so tensor parallelism was unusable exactly when it was needed, since
the only host that could satisfy it was one where a single card could hold the
whole model anyway.

Observed: `qwen2.5-72b` at `tensor_parallel_size: 2` asked for 135.43 GiB on
each of two 95.59 GiB cards and reported "the pool can never satisfy that",
where ~68 GiB per card is the real requirement. It starved every job that
needed it as an extractor.

The floor is now divided by `weight_shard_count()` — `tensor_parallel_size ×
pipeline_parallel_size`. `data_parallel_size` is deliberately excluded: it
replicates the model, so each replica needs the whole thing. A declared
`placement.min_vram_gib` is untouched, being per-GPU by convention.

### Documented the Slurm compatibility model

`docs/slurm-compatibility.md`. Slurm allocates GPUs to jobs; infer-stack places
models within the allocation it was given. The two do not overlap, and the
module docstring saying multi-node placement is "Slurm territory" was easy to
read as "does not work under Slurm".

States the part that is not automatic: `$SLURM_JOB_GPUS` must be passed as
`--allowed_gpus`. Without it placement considers every GPU on the machine,
including cards allocated to another job, and the failure surfaces later as a
CUDA OOM in whichever job loses.

## Version 0.7.1 - Unreleased


## Version 0.7.0 - Released 2026-08-28

### TUI: the logs pane defaults to engines, not everything

The docker logs pane now follows every service EXCEPT the LiteLLM gateway by
default. LiteLLM emits a line per proxied request, so on a busy host it scrolls
the engine output -- which is where startup failures, OOMs and CUDA errors
actually appear -- out of the pane before it can be read.

`(all services)` and the gateway itself remain one selection away in the same
dropdown. The engines view expands to concrete service names passed to
`docker compose logs`, so it is a narrower stream rather than a filter applied
after the fact; with no engine services deployed it falls back to all services
and says so in the label.

### Added
* **Reserve-only GPU lease: `infer-stack acquire --reserve-gpus N`.** Hold N
  *available* GPUs (count-based first-fit — infer-stack picks which, never a
  pinned index) without launching any server, so an external process can run on
  exactly the reserved card under the SAME admission-queue / render-lock
  accounting as served runs (a reserved GPU is withheld from concurrent vLLM
  placements and vice-versa, because the reservation is a real ledger Deployment
  visible cross-process). Modelled as a non-servable deployment
  (`engine='reserved'`, DEDICATED so two reservations never coalesce onto one
  GPU, `reclaim!=keep-warm` so release frees the GPU at once); it renders no
  container (render_compose already skips non-vllm/ollama), is never probed for
  readiness, and reports the chosen index via the env-file's
  `CUDA_VISIBLE_DEVICES`. Honors `allowed_gpus`/`$SLURM_JOB_GPUS` like any
  placement. Claims are recorded with `kind='reserved-gpu'`. This turns the
  previously-unwired Phase-2 `reserved` scaffolding into a usable feature.
  `tests/test_leasing_reservation.py`; `tests/test_reservation_gpu_frame_e2e.py`
  is an opt-in on-host probe that the reserved index and `docker --gpus device=`
  agree on the same physical GPU.
* **TUI: "Evict all idle" + multi-select in the leases/deployments tables.**
  Clearing a pile of released-but-kept-warm deployments no longer means evicting
  one row at a time: a new **Evict all idle** button (deployments pane) flips
  every IDLE deployment to STOPPED in one action (then **Clean up** forgets
  them). Both tables also gained multi-select: **space** toggles the cursor row,
  **ctrl/cmd-click** toggles a discontiguous row, and **shift-click** extends a
  contiguous range (selection shown in a leading marker column); **Release** /
  **Evict** then act on every checked row, falling back to the cursor row when
  nothing is checked. The selection is kept by id so it survives a poll refresh
  and is pruned to live rows. Note: "Clean up" still only forgets STOPPED
  deployments + RELEASED/EXPIRED leases by design — IDLE (keep-warm) deployments
  are retained until evicted. The multi-select is a hand-rolled shim (Textual 8.x
  has no native row multi-select, only text selection); it is self-contained and
  can be dropped if Textual ships one (see textual#3606 / PR #6585). Tests in
  `tests/test_tui.py`.
* **Dynamic LiteLLM routing via the admin API + Postgres (opt-in
  `dynamic_routing`).** A new mode that manages the gateway's route table *live*
  through LiteLLM's admin API (`/model/new` / `/model/delete`) against a
  Postgres-backed model store (`STORE_MODEL_IN_DB`), instead of a static config
  file. It fixes the **same-model `--dedicated` collision**: in static-superset
  mode every dedicated deployment of one served model collapses onto a single
  `vllm-<served>` container (one GPU), but with dynamic routing each deployment
  gets its own `vllm-<served>-<id>` upstream, so N dedicated deployments run on N
  GPUs (LiteLLM load-balances the shared public alias across them). It follows
  the render/apply split: render writes the desired route set (`litellm_routes
  .json`, one entry per live `(deployment, endpoint)` with a deterministic
  `model_info.id`) and a *static* base gateway config (empty `model_list`, so the
  gateway is never recreated — no blip); apply reconciles the live gateway as an
  idempotent set-diff (`ComposeBackend._reconcile_routes`), so it coalesces,
  heals drift (routes lost to a restart reappear; stale routes are deleted), and
  leaves hand-added models (no `isr-` id) alone. Off by default (static superset
  stays the default); enable with `config set dynamic_routing true` or
  `--dynamic-routing`. Backed by `compose._litellm_routes` / `_postgres_service`
  / `ComposeBackend.db_password()`; tests in
  `tests/test_leasing_dynamic_routing.py`. NOTE: verified against the pinned
  `litellm v1.82.3` that the admin API requires a DB (it is *not* DB-less), so
  Postgres is a hard requirement for this mode.
* **Coalesced apply: one `docker compose up` serves a whole batch of concurrent
  acquires.** The controller's critical section is split into a fast RENDER lock
  (ledger write + placement + compose-file render) and a separate APPLY lock
  around the slow `docker compose up`, so a second caller can render while the
  first is still applying (acquires no longer serialize end-to-end behind each
  other's bring-up). A monotonic generation in the ledger (`desired_gen` bumped
  by each mutation, `applied_gen` published after a successful apply) lets the
  apply lock double as a coalescing wait-queue: an acquirer whose generation is
  already covered skips its own apply, so N concurrent acquires need far fewer
  than N applies. The snapshot is a guaranteed-covered floor (taken before the
  `up`), so a render landing mid-apply is re-applied next, never dropped; crash
  during apply is safe (the flock auto-releases and `up` is idempotent). Backed
  by `Controller._render` / `_ensure_applied` / `_apply_lock` and the new
  `ComposeBackend.apply()`. `infer-stack apply` uses the new
  `Controller.apply_now()` (force) so it still heals drift when nothing changed.

* **`infer-stack gc` — reclaim leaked leases and free their GPUs.** Sweeps
  TTL-expired leases (a hard-killed job — SIGKILL/OOM/reboot — never runs its
  `release`, so its lease lingers until TTL) and reconciles, tearing down any
  `stop`-policy deployment left with no demand. Run it periodically (cron) or as
  a final pipeline step; a blocking `acquire --queue` already does this
  implicitly while it waits. `--evict` also tears down idle keep-warm
  deployments (like `evict --all`). Backed by `Controller.gc(evict_idle=...)`.
* **Admission queue for `acquire` / `run` (`--queue`).** Instead of failing fast
  when every GPU is busy, `acquire`/`run --queue` (and
  `Controller.acquire(wait_for_placement=True)`) poll until a deployment frees a
  GPU, bounded by `--timeout`. Each retry sweeps the ledger first, so a crashed
  job's TTL-expired lease is reclaimed while waiting and its GPU lets the queued
  request through — queueing and leak-recovery are the same mechanism. Default
  off, so interactive use keeps its fail-fast "no GPU" error; batch/pipeline
  fan-out opts in. Queueing is currently plain (no head-of-line reservation), so
  a multi-GPU request can be starved by a stream of single-GPU ones — fine for
  the small-fleet case; reservation is a follow-up.
* **LiteLLM gateway no longer blips when the model set changes (static superset
  route table).** When the backend has the catalog, the gateway is rendered with
  one route per *catalog* endpoint addressing a *deterministic* upstream host
  (`vllm-<served>` / `ollama-<host>`, no deployment-id suffix), so its config —
  and therefore its container — is untouched as models are acquired/released:
  `docker compose up` leaves the gateway running instead of recreating it. The
  `config_hash` still recreates it when the *catalog itself* changes (new/removed
  endpoints), which is correct. vLLM/Ollama service names are now deterministic
  from the served name/host (`observe` still correlates containers via the
  `infer-stack.deployment` label, so reconcile is unaffected). The per-model
  `depends_on` on the gateway is dropped (the `router_settings` already make the
  upstream-warmup window self-healing). Without a catalog the legacy
  per-deployment config is used (and still churns). Caveat: two simultaneously
  *desired* deployments sharing a served name (an endpoint re-pointed at a new
  model while the old is live) would collide on the deterministic name — an
  interactive case unsupported under the static gateway.
* **Open WebUI can manage Ollama's own models.** Open WebUI is no longer locked
  to the LiteLLM gateway with `ENABLE_OLLAMA_API=False`. It now holds two
  connections: an **OpenAI** connection (the gateway when on, else a single
  upstream's own `/v1`) for chat, and a **native Ollama** connection pointed
  straight at any Ollama daemon, so you can pull/run/delete models from the UI
  and have the daemon load them on demand — a true drop-in for a hand-run
  `ollama` + Open WebUI stack.
* **LiteLLM gateway is now optional** — `--litellm` / `--no-litellm` (and
  `config set litellm`). With it off, Open WebUI (still on by default) talks to
  the rendered upstreams directly, and `access()` reports the UI URL. Open WebUI
  also renders without a gateway whenever there is an upstream to point at.
  Ollama tags are now pulled on `acquire` even with the gateway off.
* New tutorial: `docs/source/manual/ollama-openwebui-tutorial.md` — stand up a
  self-managing Ollama + Open WebUI box (GPU-pinned) entirely from the CLI. Adds
  a `docs/source/manual/` section to the Sphinx docs (the leasing demo moved
  there too).
* TUI **API tab** is now a proper console: shows the gateway + Open WebUI URLs
  (ctrl+click to open), **List models** (GET `/v1/models` on the gateway) beside
  Send / Test-all, a live **curl** preview with a **Copy curl** button, and an
  **Open WebUI** button. Clipboard support via Textual (OSC 52): Copy-curl, and
  `y` copies the status line (handy for the open-URL it prints).
* TUI live-feedback polish: the **endpoint wizard is labeled and
  engine-adaptive** — vLLM shows tensor-parallel / **data-parallel** / max-model-len
  / GPU-mem / max-seqs / prefix-caching / extra-args; Ollama shows host +
  free-form runtime — so the form is no longer a row of unlabeled inputs. The
  **API tester moved to its own top-level tab** (more room for the monitor);
  catalog **Add/Edit/Remove are localized** to the endpoints + models panels
  (the bottom button stack is gone; Suggest moved to Settings); the vertical
  splitter drags the **full width range** (not just the middle); and the
  endpoints↔models drag direction is fixed.
* TUI **top-level tabs** — the multipane monitor is now a **Dashboard** tab, with
  a new **Settings** tab to edit the durable settings (backend, data dir, Open
  WebUI, reverse proxy, skip-display-GPUs) and save them to `settings.yaml`
  without dropping to YAML. Textual's command palette (ctrl+p) exposes every
  action for search.
* TUI docker pane gained a **Control** tab (compose up/down + the rendered
  compose-file path), and the models table now shows **quant** + a **cached**
  flag (a cheap existence check against the HF hub cache — no slow `du`).
* TUI catalog management — the add-endpoint wizard now exposes the runtime knobs
  that matter for serving (**tensor-parallel size, max model len, GPU memory
  fraction, raw extra vLLM args** — where data-parallel etc. go — and the
  **reclaim policy**), mirroring `catalog endpoint add`. You can **edit** an
  endpoint (blocked while it's actively served) and **remove** an endpoint or
  model (with a confirm dialog; a model still referenced by an endpoint is
  refused by the validating writer). CLI parity already exists via
  `catalog endpoint add [--force]` / `catalog endpoint rm` / `catalog model rm`.

### Fixed
* **A `acquire` that timed out waiting for readiness left its lease ACTIVE,
  pinning a GPU indefinitely.** Unlike `run` (which releases in a `finally`), a
  plain `acquire --timeout` whose endpoints never became ready returned with the
  lease still held, so the deployment stayed LIVE and — combined with reconcile
  trusting the ledger as desired-state — could be re-realized on every subsequent
  converge. A readiness timeout is now the third "couldn't deliver" rollback path
  in `Controller.acquire` (alongside `ConvergeAborted` / `PlacementError`): it
  releases the lease and reconciles, tearing the deployment down per its reclaim
  policy. The `acquire` CLI prints the teardown and exits non-zero; the outcome
  carries `released_on_timeout=True`. To intentionally hold a lease while a slow
  model loads, use `--no-wait` (acquire detached) and `wait` for it separately.
* **Upstream containers blipped (and broke readiness mid-request) when an
  unrelated deployment was added or released.** Each vLLM/ollama upstream
  published a host port assigned by *position* in the live set (`BASE + i`), so
  adding or removing any deployment renumbered every survivor's port — which
  changed their rendered service specs and made `docker compose up -d` recreate
  unrelated, still-leased containers. With the gateway up, LiteLLM's route then
  pointed at a container that was restarting, so in-flight requests got
  `InternalServerError: Connection error` for the ~minute it took vLLM to reload
  — surfacing as a flaky slurm-e2e node failure when one job's `release` landed
  during another's readiness probe. Behind the gateway an upstream is internal
  (reached by compose-network DNS at `:8000`), so it now publishes **no** host
  port and each survivor's spec is byte-identical as the set changes — the same
  no-blip property the static gateway config already has. The no-gateway path
  still publishes (the readiness probe hits the upstream directly there). Also
  hardened the dynamic-routing reconcile to treat a `/model/delete` "not found in
  db" as success (a shared gateway lets another converge delete the route first).
* **`database is locked` when several processes open a fresh ledger at once.**
  Switching the journal to WAL (and creating the schema) on first open needs a
  brief exclusive lock that sqlite returns immediately as "locked" rather than
  honoring `busy_timeout` — so a batch of pipeline jobs all running
  `infer-stack acquire` against a brand-new ledger could race and crash in
  `SqliteStore.__init__`. These DDL steps now retry on a transient lock.
* **A mutating verb silently degraded to an in-process lock — which serializes
  nothing across CLIs — when the cross-process lock file couldn't be opened.**
  The render lock falls back from the (often service-owned, read-only) ledger
  dir to a host-temp file keyed by the ledger path; that fallback is shared by
  *all* users on the host, so the first user to run created it `0644` and a
  second user/uid (a different tmux or slurm session) hit `EACCES` reopening it.
  With *both* candidates unopenable the controller used to `warn` and proceed on
  a `threading.RLock`, which only serializes threads of one process — useless
  against separate CLI processes, so concurrent `acquire`/`release`/`gc`/`evict`
  could collide on the ledger (`database is locked`, stale-diff renders). Now:
  (1) a lock file/dir we create is made **group-writable** (file `g+rw`, dir
  `g+rws`, best-effort, only for paths we own) so the next session in the owning
  group can open the same flock file; and (2) when no cross-process lock can be
  obtained at all, a mutating verb **raises `LeaseLockError`** (refuses to
  mutate) with an actionable diagnosis (the exact paths tried, why each failed,
  and the `chgrp`/`chmod g+s`/`umask 002` fix) instead of silently racing. The
  in-memory ledger (tests) and the writable-fallback case are unchanged.
* **Ollama GPU pinning to a non-zero GPU** silently fell back to CPU. The docker
  device reservation (`device_ids`) already exposes only the pinned GPU and the
  NVIDIA runtime renumbers it to `0` inside the container, but the service also
  set `CUDA_VISIBLE_DEVICES` to the *host* index — so pinning to GPU 1 left the
  container looking for device `1` that wasn't there. Now pinned by the
  reservation alone (like vLLM). New e2e tier `88_gpu_pinning` exercises this on
  a real 2nd GPU; new tier `86_ollama_lean` covers the `--no-litellm` stack, and
  `85_ollama` now asserts the dual Open WebUI connection wiring.

### Removed (breaking)
* **Collapsed `serve` into `acquire` (no compatibility alias).** `serve` was a
  thin preset of `acquire` (an infinite, `manual`-owned lease) that routed
  through the identical code path; the two verbs differed only by an owner label
  and which flags they exposed. There is now a single verb: `infer-stack acquire`
  — with no `--ttl` it is an infinite (standing-service) lease, and `--ttl 2h`
  makes it a time-boxed reservation. `acquire` now carries the everyday-`serve`
  help (render→apply→wait, `--no-apply`/`--no-wait`/`--no-ui`). Migration:
  `serve X` → `acquire X`; the default lease owner is now `$USER` (was `manual`
  for `serve`) — pass `--owner manual` to keep the old label. The TUI's "Serve"
  control is unchanged (it calls the controller directly, not the CLI verb).
* Removed the pre-leasing **profile world** now superseded by catalog + leasing
  (no back-compat — pre-release): the entire `infer-stack legacy …` command
  group and its modules (`cli/commands_profile`, `cli/commands_smoke`,
  `renderer`, `benchmark`, `verification`, `contracts`, the active-profile
  `Up/Down/Purge/Deploy/Env/Ollama*` runtime verbs), plus their tests
  (~5k LOC). The ollama and kubeai *concepts* are retained (catalog
  `engine: ollama`, leasing coalescing, `kubeai_ops`/`backends/kubeai_renderer`)
  for when those backends are implemented. Follow-up pass also removed the
  pre-leasing `resolver`, `validator`, the `cli/compose` + `cli/probes` shims,
  and the superseded `backends/compose_renderer` (replaced by
  `leasing/compose`); carved `cli/context` down to `_apply_path_overrides` +
  `effective_inventory`. (The old top-level `catalog.py` + the now-dead helpers
  inside `config.py` are an internal-only follow-up — they no longer surface in
  the CLI.)
* `infer-stack status` is now a **leasing-native holistic overview** — backend,
  data/config dirs, catalog (with model/endpoint counts), settings, ledger, and
  compose-project locations, plus a leasing summary (active leases / live
  deployments) and "dig deeper" pointers (`leases`, `tui`, `stack ps`, `logs`).
  It no longer reports on the old active-profile render.

### Changed (breaking)
* Renamed the **deployment group concept → "deployment"** throughout (core
  classes, CLI, TUI, docs): `DeploymentGroup` → `Deployment`, `GroupState` →
  `DeploymentState`, `group_id(s)` → `deployment_id(s)`, ledger methods
  (`get_group` → `get_deployment`, `list_groups` → `list_deployments`, …), the
  `leases --json` key `groups` → `deployments`, and the SQLite `groups` table →
  `deployments` (+ `claims.group_id` → `claims.deployment_id`). No DB migration —
  delete any existing ledger DB and it will be recreated. Mental model: "many
  leases → one deployment."
* Renamed the lease identifier **`session_id` → `lease_id`** everywhere (the
  object is a `Lease`; the dual vocabulary was confusing). This is a hard rename:
  the env-file key is now `INFER_STACK_LEASE_ID` (was `INFER_STACK_SESSION_ID`),
  the `release`/`renew` positional/flag is `--lease` (was `--session`), JSON
  output uses `lease_id`, `read_session_id()` → `read_lease_id()`, and generated
  lease ids are prefixed `lease-` (was `sess-`). Update any scripts that sourced
  the old env var or parsed `session_id`.

### Added
* Begin the leasing/controller redesign (see
  `dev/infer-stack-redesign-critique.md` in the aiq-eval-runner repo). New
  `infer_stack.leasing` subpackage with a backend-agnostic, sqlite-backed lease
  ledger: `acquire`/`release`/`renew` bookkeeping, demand reference-counting,
  same-model coalescing (with capacity subsumption), per-daemon coalescing for
  Ollama, soft-TTL expiry, and idle-deployment reclaim computation. This is the core
  that later phases (reconciler, backend protocol, `acquire`/`run` CLI) build on.
* Serving catalog parser (`infer_stack.leasing.catalog`): the new declarative
  `models` / `endpoints` / `runtime_hosts` / `bundles` schema (replacing
  profiles as the primary unit), with cross-reference validation and
  `resolve_endpoint` / `resolve_names` that turn endpoint and bundle names into
  ledger `EndpointRequest`s (vLLM per-model, Ollama per-daemon).
* Backend protocol + controller (`infer_stack.leasing.backend` /
  `.controller`): a 4-method `Backend` seam (`realize`/`teardown`/`observe`/
  `probe_ready`), a `MemoryBackend` for tests/dry-runs, and a `Controller` that
  reconciles the ledger's desired state onto a backend (LIVE + keep-warm-idle
  deployments), enforces TTL on every reconcile, scopes readiness waits to the
  endpoints a lease requested, and exposes thin `acquire`/`release`. The Compose
  and KubeAI backends will implement the same protocol.
* Leasing CLI verbs: `infer-stack acquire` / `release` / `renew` / `run` /
  `serve` / `leases`, plus an endpoint-descriptor env-file
  (`infer_stack.leasing.envfile`, aligned with the `contracts.py` shape) and a
  dry-run `NullBackend`. `run -- <cmd>` acquires, injects the endpoint env into
  the child, and releases on exit — the kwdagger pipeline-node seam. Until the
  Compose/KubeAI backends land, `--backend null` (default) exercises the whole
  surface without serving anything real.
* Single-host GPU placement planner (`infer_stack.leasing.placement`): assigns
  GPUs across the whole live set of deployment deployments (reusing the resolver's
  `_first_fit`), honoring `allowed_gpus`, `reserved` GPUs (for Phase-2 raw-GPU
  reservations), display-GPU skipping, and `pinned` assignments so adding or
  removing a deployment does not reshuffle already-running models. This is the placer
  the Compose backend will use; multi-node/bin-packing stay out of scope.
* Compose backend (`infer_stack.leasing.compose`): a focused renderer that turns
  the live set of deployment deployments directly into a docker-compose project
  (reusing `profile_runtime.vllm_args`), and a `ComposeBackend` that converges
  the whole union on each reconcile (`docker compose up -d --remove-orphans`),
  persisting GPU assignments so reconciles don't reshuffle running models.
  Docker is invoked through an injected `run` seam (unit-tested against a fake;
  real docker/GPU path validated on a host). The controller now prefers a
  backend's `converge(desired)` over per-deployment realize/teardown. `infer-stack
  ... --backend compose` is wired up.
* Compose LiteLLM front door + readiness + converge lock: the Compose backend
  now renders a LiteLLM gateway (default on) that routes each endpoint alias to
  its upstream vLLM/Ollama service, giving one stable `base_url`. `ComposeBackend
  .access()` supplies that real base_url + per-endpoint request name into the
  env-file descriptor (the CLI prefers it over the `--base-url` placeholder).
  `probe_ready` now checks the gateway's `/v1/models` listing (model is
  routable) via an injected HTTP seam, and `converge` is serialized with a file
  lock so concurrent processes don't clobber the shared compose file. (Ollama
  tag pull/warmup is a remaining readiness follow-up.)

### Added (continued)
* Managed **Open WebUI**, on by default. The Compose backend now renders an
  `open-webui` service in front of the LiteLLM gateway whenever the front door
  is up, so `infer-stack serve chat` brings up a working chat UI (default
  `http://127.0.0.1:13000`) with no hand-run `docker run`. Its spec is
  independent of which models are live (it talks to the `litellm` service at a
  fixed URL), so `docker compose up -d` leaves it running across model
  add/remove/switch — only the gateway is recreated on a routing change, so the
  UI never blinks (the property the legacy stack worked to preserve). Chat
  history persists under the data dir. Opt out per-call with `--no-ui` or
  globally with `infer-stack config set ui false`; `serve`/`acquire` print the
  UI URL when ready.
* `infer-stack test <endpoint>` — a leasing-native smoke test. Sends one real
  chat completion to the endpoint *alias* through the gateway (managed key
  applied automatically) and prints latency + the reply, or an actionable
  failure with a non-zero exit. The concise alternative to hand-rolled `curl`
  (the demo still shows `curl` for the raw form).
* `infer-stack env` is now the single verb for the managed env-file, replacing
  both the `secrets` alias and the `secret get|set|list` modal (removed — the
  secrets live in a readable `.env`, so a separate "secret" surface earned its
  keep only by adding `set`, which folds in cleanly):
  - `infer-stack env` — print the env-file **path** (`source "$(infer-stack
    env)"` to load everything);
  - `infer-stack env LITELLM_MASTER_KEY` — print one value;
  - `infer-stack env HF_TOKEN=hf_…` — set one value (merges non-destructively,
    so a gated model's token can be set once before `serve`);
  - `infer-stack env --export` — every entry as `export KEY=value` lines.
  The argument is a `KEY` to read or `KEY=VALUE` to write — mirroring the legacy
  `infer-stack env` ergonomic in one command.
* `config set ui true|false` — a durable default for the managed Open WebUI
  (a new recognized setting alongside `backend` / `data_dir`).
* LiteLLM front-door config now carries `router_settings`
  (`num_retries`/`timeout`/`cooldown_time`/`allowed_fails`) so the brief window
  where an upstream vLLM/Ollama is still loading its model is retried and
  self-heals instead of surfacing as client `500`s / loud
  `InternalServerError: … Connection error. Received Model Group=…` logs.
  (LiteLLM doesn't wait for upstream *health* to start, so it forwards early
  requests to a not-yet-listening upstream during warmup.)
* `python -m infer_stack` now runs the CLI (a top-level `__main__.py` redirects
  to `infer_stack.cli.main`), matching `python -m infer_stack.cli`.
* Faster CLI startup. Heavy third-party deps that only matter at *runtime*
  (`requests`, `jinja2`, `rich.syntax`/`pygments`) are now imported lazily
  inside the functions that use them instead of at module import, so a bare
  `infer-stack --help` / tab-completion no longer pays for the HTTP, templating,
  and syntax-highlighting stacks (cli import time roughly −30%). The
  `cli_mod.requests` test seam is preserved via a module `__getattr__`.
* Reorganized the sprawly flat CLI (~38 top-level verbs) into noun submodals,
  keeping the leasing hot path at the top level (see `dev/cli-redesign.md`;
  `infer-stack help tree` prints the whole surface):
  - `infer-stack catalog …` — a flag-driven editor for the user catalog
    (`catalog model|endpoint|host|bundle add|list|show|rm`, plus
    `init/path/show/validate/edit`) with a validating writer, so models/endpoints
    are added without hand-editing YAML.
  - `infer-stack config …` — `init/paths/show/set/get/edit` over a new durable
    `settings.yaml`. `config init` is an interactive rich prompt (data dir +
    default backend, with a confirmation) and takes `--yes` (and auto-detects a
    non-TTY) for non-interactive scripting. `config set backend compose` and `config set data_dir <p>`
    are honored (the leasing `--backend` default and `data_root()` consult them),
    so the backend flag and storage location no longer have to be repeated/
    exported.
  - `infer-stack release --all` — release every active lease in one shot (the
    whole stack idles/tears down per each lease's reclaim policy). Makes teardown
    a one-liner instead of scraping session ids out of `leases`.
  - `infer-stack wait [NAME…]` — block until served endpoints are ready, the
    leasing-native companion to `serve --no-wait` (the old `wait-ready` was
    legacy/profile-only). Lets you fan out — `serve --no-wait a; serve --no-wait
    b; wait a b` loads models in parallel instead of back-to-back. No names waits
    for every live deployment; `--require-generation`/`--timeout`/`--interval` apply.
    (Readiness has two orthogonal knobs: `--require-generation` is the
    *criterion* — a real token vs a listed model — and `--wait`/`--no-wait` +
    `wait` are the *blocking* control.)
  - `infer-stack evict [NAME…|--all]` — force-tear-down released (idle) models
    now, overriding `keep-warm`, to free their GPUs. A keep-warm model normally
    stays resident after release (no cold-start next time) but holds a GPU; evict
    drops it. Target by served endpoint alias or deployment id, or `--all` for every
    idle deployment; live models (with an active lease) are never evicted. `release
    --evict` does the release-then-evict in one step (composes with `--all`).
    Mechanically: idle deployments are marked `stopped`, so the next reconcile
    converges them away.
  - `infer-stack leases` is rich-formatted on a terminal (lease/deployment tables
    with state colors); piped/`--json` output is unchanged.
  - `infer-stack leases` now shows **actual vs desired**, not just the ledger's
    intent. Each deployment gains a `running` column (from `backend.observe()` — what
    docker actually has up) and a `gpus` column (which GPU indices it is on, or
    `→N` *slated* for a desired-but-not-yet-started deployment). So a `state=live` /
    `running=—` row reads as "wanted, not up yet" (starting, staged, or
    unplaceable) instead of looking like a phantom. Both fields are in `--json`.
    Best-effort: a dry-run/docker-less host degrades to "unknown" rather than
    erroring.
  - Render and apply are now separate verbs, with `serve`/`acquire` as the
    combined "declare + render + apply + wait". `infer-stack render` writes the
    on-disk compose project (+ GPU placement) for the current desired set
    **without** `docker compose up`; `infer-stack apply` brings the desired set
    up (idempotent; re-renders from intent first). Both are **lease-free** — the
    duplicate-lease trap from staging-then-re-serving is gone because applying a
    staged lease no longer goes through a declare verb (so the refcount only
    climbs when you genuinely declare again). Declares refcount
    (`acquire`/`serve`); reconciles are idempotent (`render`/`apply`).
    `serve|acquire --no-apply` *stages*: declare the lease + render, skip the up
    (and the wait + diff prompt); `release` discards it. Placement still runs at
    render time, so an unplaceable request fails fast either way. Fits the
    existing ledger→controller→backend split: `converge(desired, apply=False)`
    is the render half, `ComposeBackend.plan()` exposes read-only placement (also
    what `leases`' `gpus` column uses).
  - The rendered compose carries a top-level `name: infer-stack`, so a plain
    `docker compose -f docker-compose.yml up -d` (infer-stack not involved)
    lands in the *same* project — same container names, same network — as the
    tool's own `-p` invocations. "Drop infer-stack and run docker yourself" is
    now an exact equivalent of `apply` rather than a sibling project the tool
    can no longer see. (`infer-stack stack up` remains the raw "run the on-disk
    file verbatim" hatch, vs `apply` which re-renders from intent.)
  - Optional Textual TUI (`infer-stack tui`): a multi-pane dashboard. A
    **catalog** pane (left) lists your models + endpoints — select an endpoint
    and press `s`/Enter to request a lease; **leases** + **deployments** panes show
    the live ledger (desired state vs running, GPUs), auto-refreshing; a **logs**
    pane tails `docker compose logs -f` and a dropdown points it at a specific
    service (or all). Controls: `s` serve, `d` release, `a` release-all, `e`
    evict, `r` refresh, `tab` cycle panes, `q` quit. Mutations converge off the
    UI thread so the monitor stays responsive; narration is silenced while the
    TUI owns the terminal. The log source is injectable, so the whole app is
    exercised headless via Textual's pilot in the tests. Opt-in extra —
    `pip install "infer-stack[tui]"`; without textual the command exits with an
    install hint. (This also made `SqliteStore` thread-safe —
    `check_same_thread=False` + a lock serializing write transactions — so the
    same ledger connection is usable from a converge worker thread.)
  - TUI, second pass — made it approachable and snappier for new users:
    a warm **orange / white / dark-gray theme**; an intro line plus per-pane
    help text; **add-model / add-endpoint wizards** (`m` / `n`, or the sidebar
    buttons) and a **Suggest** button (`g`) that seeds a catalog sized to your
    GPUs via `catalog suggest`; the logs pane became a **docker tab** (live
    `logs -f` *and* a `ps` snapshot). Actions are now **scoped to the pane they
    act on** — Serve sits under the catalog, Release/Release-all under the
    leases table, Evict under the deployments table — so the global footer keeps only
    truly-global controls (Refresh / Next-pane / Quit); the keys still work.
    Panes are **drag-resizable** (grab the full-height/width splitter bars) in
    addition to the `[` `]` / `-` `+` keys. Responsiveness:
    the periodic refresh + `docker compose ps` now run on a worker thread (no UI
    freeze), and docker's own `up`/`down` progress is captured into the logs
    pane instead of bleeding onto the screen. The TUI now requires a one-time
    `config init`, and tolerates a missing/empty catalog (it shows the
    empty-state + Suggest button rather than erroring).
  - TUI, third pass — the bottom pane became a **collapsible, tabbed console**
    (click the title, or `c`, to collapse it). Tabs: **Logs**; **Status · ps**
    (now with the `docker ps`-style status/uptime, created time, and container
    id, plus service + ports); **System** (live `nvidia-smi` per-GPU
    util/mem/temp, plus host load/mem/cpu from `/proc`); and an **API** tester
    that sends a prompt to a served model — or pings *all* of them with a
    latency report — through the LiteLLM gateway. Only the **visible** tab's
    data is polled (and nothing when the console is collapsed), so `ps` /
    nvidia-smi don't run when you can't see them. **Ctrl+click** a served
    endpoint (or `o`) opens it in Open WebUI (`/?models=<endpoint>`). The HTTP
    client is injectable for headless tests.
  - TUI, fourth pass — pane-local clarity + housekeeping. The describe-everything
    intro line is gone; each pane carries its own one-line description. The
    bottom **console** is renamed **docker** and its **Status** tab is now
    **Containers**; **System** and **API** are promoted to their own collapsed
    panes (so they're only polled when you expand them). The **API** model
    picker now lists *only models that are up and ready* (served by a running
    deployment), not every catalog entry. New **Clean up** action (`x`, or the button
    under leases/deployments) forgets released/expired leases and stopped deployments —
    backed by a new `Ledger.prune()` / `SqliteStore.prune()`. Default theme is
    now the stock **textual-dark** (the orange theme stays available from the
    command palette).
  - `Ledger.prune()` (+ `SqliteStore.prune()`): delete terminal ledger rows —
    RELEASED/EXPIRED leases and STOPPED deployments (and their claims) — for callers
    that want to forget history rather than keep it inspectable.
  - TUI — made the lease↔deployment (many-to-one) relationship legible: the
    leases pane gained a **deployment** column (the deployment id(s) a lease holds —
    the same id shown in the deployments pane, so the join is visible), and the deployments
    pane now shows **leases** (how many hold it) + **held by** (their owners)
    instead of the opaque "demand". Moving the cursor spells the link out in the
    status bar ("lease … → deployment …" / "deployment … ← held by N lease(s)").
    Action-bar buttons are compact (1 row).
  - Optional single-port HTTP reverse proxy (`reverse_proxy`). Enable it
    (`--reverse-proxy`, or `config set reverse_proxy true`) to front the gateway
    + Open WebUI with one nginx origin — UI at `/`, the OpenAI API at `/v1` — so
    there's one port to hit instead of remembering 13000 (UI) and 14042 (API).
    Plain HTTP, no TLS/auth (localhost / trusted networks only); the value is
    ergonomics, not security. The generated conf handles Open WebUI's websockets
    + large uploads; a `{enabled, port, config_path}` block (via `config edit`)
    sets the port or mounts a bring-your-own `nginx.conf`. Needs the gateway, so
    it's rendered only alongside litellm; `access()` reports the unified
    `proxy_url`. (TLS/LDAP — the legacy `frontends.reverse_proxy` features — and a
    remote control surface are deliberately *not* included here, since those
    require the auth this plain proxy can't provide.)
  - The front door (LiteLLM gateway + Open WebUI) is now a standing service,
    decoupled from model count. It was rendered only alongside ≥1 model, so
    `release --all`/`evict --all` left an empty desired set and `converge` downed
    the *whole* project — taking the CPU-only gateway/UI with it (the UI blinked,
    and `evict`, whose job is freeing GPUs, tore down the front door as a side
    effect). Now the gateway/UI render whenever enabled (empty `model_list` when
    no models), so `release`/`evict` only stop model containers (freeing GPUs)
    and the front door stays up — Open WebUI never blinks, even to zero models,
    and reconnects as you serve again. `infer-stack stack down` is the way to
    take the whole stack (gateway + UI included) down. The empty-set→`down`
    converge path now only triggers with the gateway off (`litellm=False`).
  - Better `--help`. Expanded the leasing verbs' docstrings (rendered as the
    argparse description) and added `__epilog__` examples to `serve` and
    `leases` plus a quickstart + mental-model epilog on the top-level
    `infer-stack --help` (catalog → serve/acquire → reconcile; render vs apply;
    desired vs running). The `leases` help now documents each deployment column.
  - Friendlier "unknown endpoint" error. You serve/acquire *endpoints*, not
    models — passing a model name (`serve qwen05`) now says so and lists the
    endpoints that run it (`Endpoints for 'qwen05': qwen05-1, qwen05-2 …`), or,
    if the model has none, points at `catalog endpoint add --model`. For a name
    that's neither, it offers a did-you-mean over the known endpoints/bundles.
  - Compose changes are now shown before they're applied, for **every** verb
    that touches the compose project — `serve`/`acquire` *and* `release`/`evict`/
    `apply`. On a terminal each renders a diff of the compose project (and
    LiteLLM routing) it's about to write and asks to confirm; `--yes`/`-y` skips
    it, and it's skipped automatically off a terminal (scripts/CI). Nothing
    mutates `docker-compose.yml` or runs docker without that gate — earlier,
    `release`/`evict` converged silently, which (with the container rename) could
    e.g. recreate a keep-warm model during a `release` with no preview. Each
    command batches its ledger changes into a single converge, so you're asked at
    most once. Declining an *acquire* rolls back the just-created lease; declining
    a *release/evict* leaves the ledger change recorded but docker untouched —
    `infer-stack apply` then applies it (consistent with the render/apply split).
  - `config init` now prompts for **every** known setting (data dir, backend,
    Open WebUI, display-GPU skipping), not just data dir + backend — driven by a
    single settings registry, so a newly added setting is asked about
    automatically. It also says which mode it's in ("initializing a new config
    from scratch" vs "editing the existing config at <path> (or hand-edit it with
    `infer-stack config edit`)"). Re-running edits in place and preserves any
    keys it doesn't manage; `--fresh` discards the existing config and resets to
    defaults.
  - `catalog model show` / `endpoint show` with no NAME now print every entry in
    that section instead of failing with `endpoint 'None' not found`; an unknown
    NAME's error now lists what's available (`… not found (have: e1, e2)`).
  - Behind-the-scenes feedback via loguru (like `aivm`): the leasing verbs
    narrate placement, `docker compose up/down`, and readiness waits to stderr
    (INFO; `$INFER_STACK_LOG_LEVEL` to change). It's kept off the `--help`
    import path (loguru is ~50ms) and silent for library/test use until the CLI
    enables it, so stdout/JSON output is untouched.
  - `infer-stack catalog endpoint add` — `NAME` is now optional and defaults to
    `{model}-{N}` (the vLLM model name, or the Ollama tag, slugified, with an
    auto-incrementing suffix). So `catalog endpoint add --model smol135` creates
    `smol135-1`, and a repeated add for the same model gets `smol135-2` instead
    of colliding — keeping the served name (what Open WebUI shows) tied to the
    model. An explicit `NAME` is still there for a stable alias decoupled from
    the model (e.g. `chat` you can re-point with `--force`).
  - `infer-stack catalog <model|endpoint|host|bundle> rm` now takes **multiple
    names** (`rm a b c`); removal is atomic — if any name is missing, nothing is
    removed.
  - `infer-stack env` — read/write the managed compose `.env` (path / `env KEY`
    / `env KEY=VALUE` / `--export`); e.g. `env HF_TOKEN=…` sets a gated model's
    token once before `serve`. (Supersedes the short-lived `secret` modal and
    `secrets` alias — see the `infer-stack env` entry above.)
  - `infer-stack stack …` — the day-2 compose wrappers (`logs/ps/restart/pull/
    start/stop/down`); `logs`/`ps` remain top-level aliases.
  - `infer-stack legacy …` — the pre-leasing profile/active-profile commands
    (`setup/init/render/switch/resolve/lock/validate/up/down/deploy/…/ollama-*`)
    grouped into a holding pen, promoted out as they gain leasing-native
    behavior and removed wholesale once empty.
  - `infer-stack help tree` — the full nested command tree at a glance
    (cf. `aivm help tree`).
* The day-2 compose wrappers (`logs`, `ps`, `restart`, `pull`, `start`, `stop`)
  now target the **leasing** Compose deployment when one exists
  (`data_root/leasing/compose`, project `infer-stack`) — so `infer-stack logs
  -f` / `infer-stack ps` work for a leasing user with no `config.yaml`. They
  fall back to the legacy rendered stack when there's no leasing deployment.
  (The `ollama-*` wrappers remain legacy: they exec a fixed `ollama` service
  that the leasing model names per-daemon.)
* Keep the legacy meta commands relevant post-refactor: `infer-stack config
  paths` (also exposed top-level as `infer-stack paths`) gained a `leasing`
  deployment showing the lease ledger, the compose state dir, and its rendered
  artifacts (docker-compose.yml, litellm_config.yaml, the secrets `.env`,
  sidecar). `infer-stack status` now prints a one-line leasing summary (active
  leases / live deployments) pointing at `infer-stack leases`.
* Managed LiteLLM secret + `infer-stack env`. The Compose backend now owns
  `LITELLM_MASTER_KEY` (reused from the state dir's `.env` if you pin one, else
  generated via `ensure_secret`), bakes it into the LiteLLM service, uses it for
  the readiness probe, and ships it in the `--env-file` descriptor as
  `OPENAI_API_KEY` — so `source`-ing the env-file fully configures an OpenAI
  client (no manual `export`). `infer-stack env [KEY]` prints the managed
  secrets (`$(infer-stack env LITELLM_MASTER_KEY)`), restoring the legacy
  `infer-stack env` ergonomic for the leasing model.
* Ollama pull/warmup readiness in the Compose backend: a daemon serves a tag
  lazily, so `probe_ready` now pulls the endpoint's tag into its daemon
  (`docker compose exec … ollama pull`, idempotent) and forces a generation
  through the front door to warm it before reporting ready. A `--require-generation`
  flag opts vLLM readiness into the same real-generation check.

### Fixed
* `infer-stack status` is leasing-aware. It no longer tells a leasing user (who
  has `catalog.yaml` / `settings.yaml`, maybe with active leases) that they are
  "Not initialized — run `infer-stack setup …`". `config.yaml` is now reported
  as `legacy config` (it belongs to the pre-leasing profile world), and the
  summary leads with backend / data dir / catalog / settings + the leasing
  one-liner. The setup hint only appears when *nothing* is set up, and then
  points at the leasing getting-started (`config init` / `catalog init` /
  `serve`); the legacy KubeAI status error now references `infer-stack legacy
  setup` (the command's real path after the CLI reorg). On a terminal the
  summary is now rich-formatted (bold labels, colored values, a styled leasing
  line); piped/redirected output stays plain (`Console.is_terminal`), so scripts
  and tests are unaffected.
* Converging to an empty desired set no longer crashes. Releasing the last
  `reclaim:stop` lease leaves zero services to render, and `docker compose up -d`
  errors with "no service selected" on a services-less file — so the release's
  reconcile raised (and `infer-stack run` surfaced it as a non-zero exit even
  though the job succeeded). Converge now tears the project `down` when there are
  no services instead of `up`-ing an empty file. (Latent until deployments were
  fully isolated; the GPU e2e `80_run_wrapper` tier caught it.)
* LiteLLM now reloads when its routing config changes. The gateway reads its
  model_list once at startup from a bind-mounted file; converge rewrote that
  file but `docker compose up -d` left the old container running (its service
  spec was unchanged), so a newly added/removed alias never became routable. In
  practice: coalescing a second alias onto a live deployment (e.g. an endpoint with a
  `public_name`) added it to the rendered config but the running gateway never
  picked it up, so that lease's readiness probe timed out. The LiteLLM service
  now carries a `infer-stack.config-hash` label derived from the config content,
  so converge recreates it exactly when the routing changes (and leaves it alone
  otherwise). Found by the `50_coalescing` e2e tier on GPU hardware. This trades
  a brief gateway blip on a model add/remove for correctness; keeping LiteLLM up
  across switches (as the legacy stack did) is tracked in
  `dev/leasing-followups.md`.
* Placement uses **every** GPU by default, including display-attached ones —
  skipping the monitor's GPU is now opt-in. A single-GPU host (whose only GPU
  drives the display) would otherwise place nothing at all, so the safe default
  is "use it". Opt in to leaving a display GPU free with `--skip-display-gpus`
  (per command) or `infer-stack config set skip_display_gpus true` (persisted).
  This flips the earlier default — the leasing verbs' `--include-display-gpus`
  flag is replaced by `--skip-display-gpus`, and `plan_placement`/`ComposeBackend`
  default `skip_display=False`. Demoed by the `45_both_gpus` e2e tier; see
  `dev/leasing-demo.md` and `dev/e2e_tests/`.
* `vllm_args` no longer emits `--disable-log-requests`, which vLLM v0.19.1
  rejects (`unrecognized arguments`) — it crashed vLLM, surfacing as a LiteLLM
  "Connection error". Engine-version-specific flags can go in `extra_args`.
* Compose `observe()` is now resilient to a stale/invalid on-disk compose file.
  `reconcile` observes (via `docker compose ps`, which validates the file)
  before `converge` rewrites it, so a bad file left by an earlier run would
  crash `acquire` before it could be fixed. `observe()` now returns "nothing
  observed" on any docker/parse error and lets converge overwrite the file
  (self-heal).
* Test isolation: the subprocess `run_cli` helpers in `tests/test_cli_meta.py`
  and `tests/test_cli_setup.py` used `env.setdefault('INFER_STACK_CONFIG_DIR'/
  'INFER_STACK_DATA_DIR', tmp_path)`, which let an ambient `INFER_STACK_*`
  exported in the caller's shell leak in — the tests then read the real config/
  data dir and failed (e.g. on a box where those were exported for manual
  testing). Force the vars to `tmp_path` instead.
* Compose GPU reservation emitted `capabilities: [["gpu"]]` (list-of-lists),
  which `docker compose` rejects ("capabilities.0 must be a string"); now emits
  `capabilities: ["gpu"]`. Found while testing the Compose backend on real
  2-GPU hardware (see `dev/leasing-test-plan.md`). Guarded against recurrence by
  a test that runs `docker compose config -q` on the rendered project (skipped
  where docker compose is unavailable) — it validates the artifact's schema, not
  just the dict we build.
* `acquire`/`serve` now fail fast when a requested model can't be placed,
  instead of hanging on readiness forever. Previously, if every GPU was already
  taken (e.g. one model already serving on the only free GPU), serving a second
  model placed *nothing* for it — placement was a silent `WARNING` — yet the
  ledger had already marked the new deployment `LIVE`, so `infer-stack leases` showed
  a phantom "live" deployment with no container behind it, the compose diff was empty
  (nothing to approve, hence no prompt), and `serve` blocked on a readiness
  probe for a container that would never start until Ctrl-C. The controller now
  detects that a just-requested deployment landed unplaced, rolls the lease back
  (matching the diff-declined path) and raises `PlacementError`; the CLI prints
  the planner's reason ("need 1 GPUs but only 0 available") plus how to free a
  GPU (`leases` → `release`/`evict`). Found running the `dev/leasing-demo.md`
  walkthrough on yardrat (two models, one free GPU).

### Changed
* vLLM compose service/container names now lead with the served model:
  `vllm-<model>-<deployment-id>` (e.g. `infer-stack-vllm-qwen05-1-grp-098e…`) instead
  of the opaque `vllm-grp-098e…`. A vLLM deployment is exactly one model in one
  container, so `docker ps` / `nvidia-smi` are now legible *without* infer-stack
  — a stated goal: you can drop the tool and the running stack still makes sense.
  The full deployment id is kept as a suffix so the name stays unique (two desired
  deployments can share a served name when an endpoint is re-pointed at a new model)
  and correlates 1:1 with the `id` column of `infer-stack leases`. The name is
  also LiteLLM's on-network upstream host, so it is slugified to a DNS-safe
  `[a-z0-9-]` label and derived from one helper used by both the service key and
  the routing config. Ollama daemons keep their deployment-id name (one daemon can
  host several models, so a model-led name would mislead). Upgrading recreates
  already-running vLLM containers once (the service key changes).
* Consolidate shared machinery so the legacy and leasing code paths reuse one
  implementation instead of duplicating it:
  - GPU-pool placement primitives (`available_gpu_indices` / `first_fit` /
    `resolve_gpu_indices`) moved to `infer_stack.hardware` and reused by both
    the resolver and the leasing placement planner (no more importing the
    resolver's private functions).
  - HTTP readiness probes moved to a new layer-neutral `infer_stack.probe`
    (`openai_ready` / `ollama_ready`) over an injectable HTTP client. `cli.probes`
    re-exports them for the legacy `wait-ready`/`switch` callers; the leasing
    Compose backend reuses the same probe (so its readiness gained the
    advertised-alias and optional-generation checks). One probe implementation
    instead of two.

## [Version 0.0.1] -

### Added
* Initial version
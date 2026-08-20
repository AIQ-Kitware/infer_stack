# Changelog
We [keep a changelog](https://keepachangelog.com/en/1.0.0/).
We aim to adhere to [semantic versioning](https://semver.org/spec/v2.0.0.html).

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

## [Version 0.7.0] - Unreleased

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
# Changelog
We [keep a changelog](https://keepachangelog.com/en/1.0.0/).
We aim to adhere to [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Version 0.7.0] - Unreleased

### Added
* Begin the leasing/controller redesign (see
  `dev/infer-stack-redesign-critique.md` in the aiq-eval-runner repo). New
  `infer_stack.leasing` subpackage with a backend-agnostic, sqlite-backed lease
  ledger: `acquire`/`release`/`renew` bookkeeping, demand reference-counting,
  same-model coalescing (with capacity subsumption), per-daemon coalescing for
  Ollama, soft-TTL expiry, and idle-group reclaim computation. This is the core
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
  groups), enforces TTL on every reconcile, scopes readiness waits to the
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
  GPUs across the whole live set of deployment groups (reusing the resolver's
  `_first_fit`), honoring `allowed_gpus`, `reserved` GPUs (for Phase-2 raw-GPU
  reservations), display-GPU skipping, and `pinned` assignments so adding or
  removing a group does not reshuffle already-running models. This is the placer
  the Compose backend will use; multi-node/bin-packing stay out of scope.
* Compose backend (`infer_stack.leasing.compose`): a focused renderer that turns
  the live set of deployment groups directly into a docker-compose project
  (reusing `profile_runtime.vllm_args`), and a `ComposeBackend` that converges
  the whole union on each reconcile (`docker compose up -d --remove-orphans`),
  persisting GPU assignments so reconciles don't reshuffle running models.
  Docker is invoked through an injected `run` seam (unit-tested against a fake;
  real docker/GPU path validated on a host). The controller now prefers a
  backend's `converge(desired)` over per-group realize/teardown. `infer-stack
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
  group showing the lease ledger, the compose state dir, and its rendered
  artifacts (docker-compose.yml, litellm_config.yaml, the secrets `.env`,
  sidecar). `infer-stack status` now prints a one-line leasing summary (active
  leases / live groups) pointing at `infer-stack leases`.
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
  practice: coalescing a second alias onto a live group (e.g. an endpoint with a
  `public_name`) added it to the rendered config but the running gateway never
  picked it up, so that lease's readiness probe timed out. The LiteLLM service
  now carries a `infer-stack.config-hash` label derived from the config content,
  so converge recreates it exactly when the routing changes (and leaves it alone
  otherwise). Found by the `50_coalescing` e2e tier on GPU hardware. This trades
  a brief gateway blip on a model add/remove for correctness; keeping LiteLLM up
  across switches (as the legacy stack did) is tracked in
  `dev/leasing-followups.md`.
* Display-attached GPUs are now usable on demand. The placer still skips them by
  default (so a workstation's monitor GPU is left alone), but the leasing verbs
  gained `--include-display-gpus`, wired to the Compose backend's `skip_display`,
  so a host whose only spare GPU happens to drive a display can place models on
  it (and spread distinct models across every GPU). Demoed by the `45_both_gpus`
  e2e tier; see `dev/leasing-demo.md` and `dev/e2e_tests/`.
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

### Changed
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
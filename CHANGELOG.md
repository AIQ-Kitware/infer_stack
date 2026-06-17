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
* Ollama pull/warmup readiness in the Compose backend: a daemon serves a tag
  lazily, so `probe_ready` now pulls the endpoint's tag into its daemon
  (`docker compose exec … ollama pull`, idempotent) and forces a generation
  through the front door to warm it before reporting ready. A `--require-generation`
  flag opts vLLM readiness into the same real-generation check.

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
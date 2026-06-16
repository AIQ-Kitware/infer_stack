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

## [Version 0.0.1] -

### Added
* Initial version
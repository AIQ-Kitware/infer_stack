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

## [Version 0.0.1] -

### Added
* Initial version
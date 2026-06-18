## 2026-05-22 21:30 -0400

Model: claude-sonnet-4-6, then claude-opus-4-7 (model switch mid-session).

User intent: `vllm-stack switch <profile> --apply` was bouncing Open WebUI
and LiteLLM on every model swap. The goal was to keep both up while only
the affected vLLM service is recreated, so chat sessions and WebSocket
connections survive the swap. Also surfaced several smaller papercuts on
the way: GPT-2 1024-context limit getting hit by Open WebUI's hardcoded
`max_tokens=1000` title-generation feature, port 14000 squatted by VSCode
Remote Tunnels' Microsoft Auth OAuth loopback listener, pre-flight port
check refusing legitimate self-owned ports, and the smoke test crashing
with a 60-line `requests` traceback when the stack wasn't ready.

What I changed:

- **Pre-flight port check now ignores self-owned ports.** New
  `our_published_ports(compose_cmd, compose_file, env_file)` calls
  `docker compose -f <file> ps --format json` and returns the host ports
  our own project publishes. `_preflight_check_ports` skips those, so
  `up`/`switch --apply` on a running stack stops false-positive-rejecting
  with "port 14042 already in use" when the holder is *our* litellm.
- **`compose_recreate_router` no longer touches open-webui.** Was
  force-recreating both `litellm` and `open-webui` so the router would
  reload the rendered YAML. Dropped open-webui from the service list:
  Open WebUI re-fetches `/v1/models` on user actions, so a brief
  stale-cache window is fine, and force-recreating logs every chat user
  out of the UI.
- **Open WebUI no longer `depends_on: litellm` in the compose template.**
  Even with open-webui removed from `compose_recreate_router`, the user
  still observed it being recreated on every profile switch. Diffing
  renders with same data dir confirmed the open-webui block was
  byte-identical between profiles. Empirically, Compose's `up -d` still
  cascade-recreates dependents when their dependency is recreated, in
  some interaction with `depends_on: condition: service_started` that
  doesn't show up in the rendered hash. Breaking the YAML dep entirely
  is the only reliable way to isolate Open WebUI from the chain. Open
  WebUI's built-in retry logic for unreachable model providers makes
  this safe — visible in the logs as `Cannot connect to host
  litellm:4000` until the next poll succeeds.
- **Live LiteLLM router refresh** (`_litellm_refresh_router_live`).
  After `compose_up`, instead of force-recreating litellm to pick up the
  new model list, diff `GET /model/info` (current state in the running
  container) against the rendered YAML and apply via `POST /model/delete`
  + `POST /model/new` with delete-before-add so a retargeted alias can
  hand over without LiteLLM rejecting a duplicate `model_name`. Falls
  back to `compose_recreate_router` only if anything goes wrong (admin
  API unreachable on cold start, master key missing, individual call
  fails).
- **`store_model_in_db: True` in the generated `litellm_config.yaml`.**
  This was the blocking issue for the live refresh: LiteLLM's
  `/model/delete` endpoint requires the model to be in postgres,
  but YAML-loaded models live in the in-memory router only. With
  `store_model_in_db: True` LiteLLM syncs YAML models into postgres on
  startup, so admin API CRUD works on them. Without it, `/model/delete`
  responds "model not found in DB" → `RouterRefreshError` → fallback
  fires every time.
- **Resolve `os.environ/VAR` references before sending to admin API.**
  LiteLLM substitutes `os.environ/...` strings at YAML-load time only.
  The admin API takes literal values. Before sending the parsed YAML
  entry to `/model/new`, walk it recursively and replace
  `os.environ/VAR` with `env[VAR]` from the rendered `.env`.
- **`LITELLM_MASTER_KEY` gets `sk-` prefix.** Extended `ensure_secret()`
  with a `prefix=` arg; the renderer passes `prefix="sk-"` for
  `LITELLM_MASTER_KEY`. If the existing key doesn't start with the
  prefix, regenerate. Stops LiteLLM's confusing "Authentication Error,
  LiteLLM Virtual Key expected. Received=AIrV…, expected to start with
  'sk-'" rejection when users authenticate with the master key. (Note:
  the real symptom was a red herring — auth actually worked for the
  smoke test path, but the same key shape would have been rejected in
  some virtual-key paths.)
- **New `vllm-stack env` subcommand.** Three modes: bare prints the path
  to the rendered `.env`, `--key NAME` prints one value, `--export`
  prints `eval`-friendly `export KEY=value` lines (with `shlex.quote`).
  Replaces the awkward `grep KEY .env | cut -d= -f2` pattern.
- **New `vllm-stack purge` subcommand.** Stops the stack then uses a
  temporary Alpine container (`docker run --rm -v <parent>:/mnt alpine
  rm -rf /mnt/<dir>`) to delete root-owned state directories that
  user-space `rm` can't touch. `--delete-cache` also wipes
  `hf-cache/` and `vllm-cache/` for a full reset; default preserves
  cached model weights.
- **Smoke test errors are now one-liners.** New `_smoke_request`
  wrapper maps the common `requests` failures onto `SystemExit` with
  remediation hints: `ConnectionError(RemoteDisconnected)` → "router
  up but upstream not ready, `vllm-stack logs vllm-*`"; `ConnectionError`
  with refused → "nothing listening yet, `vllm-stack ps`"; `Timeout` →
  "model still loading"; 401/403 → "auth key out of sync, restart with
  `vllm-stack down && vllm-stack up -d`"; 503 → "vLLM still loading".
  No more 60-line tracebacks for transient startup state.
- **Default LiteLLM port changed 14000 → 14042.** VSCode Remote
  Tunnels' built-in Microsoft Auth extension parks an OAuth loopback
  listener on a fixed port — happened to be 14000 in the user's build.
  14042 is the same character (just a tad bumped) and far enough from
  the typical ephemeral OAuth port range that it shouldn't recur.
- **`gpt2-single` profile and `gpt2` model.** GPT-2 124M
  (`openai-community/gpt2`), ~250 MB, completions-only, no HF_TOKEN —
  the smallest possible plumbing-test profile. Quickstart now starts
  here and progresses to smollm2-135m then workstation-safe.
- **`model_info` block in the LiteLLM template.** Every model carries
  `max_tokens` / `max_input_tokens` / `max_output_tokens` derived from
  `max_model_len` (50/50 split). Well-behaved clients (Open WebUI's
  title-generation pipeline being one) see the cap and don't blow past
  GPT-2's 1024 context with their hardcoded `max_tokens=1000`.
- **Qwen3.5 reasoning blocks.** All 8 `qwen3.5-*` model entries were
  missing `reasoning: {enabled: true, parser: qwen3, expose_to_openwebui:
  true}`. Qwen3.6 had it; the 3.5 family was skipped. Without the
  block, vLLM doesn't pass `--reasoning-parser qwen3`, the model's
  `<think>…</think>` block leaks into `choices[].message.content`.

State of mind / reflections:

The root cause of "Open WebUI bounces" took three iterations to fully
isolate, because each fix exposed another layer:

1. First attempt — drop open-webui from `compose_recreate_router`. Tests
   passed, but user reported "still went down". I assumed the live
   refresh would now spare litellm too, so open-webui's session would be
   undisturbed. Wrong.

2. Second attempt — find why live refresh wasn't actually replacing the
   fallback. Realised `store_model_in_db: True` was missing, so YAML
   models had synthesised in-memory IDs that `/model/delete` rejected
   with "not found in DB". Also: `os.environ/VAR` strings were being
   forwarded verbatim to `POST /model/new` instead of being resolved
   from the rendered `.env`. Fixed both. Tests passed, user reported
   "still went down" again — with logs showing
   `open-webui exited with code 0` *after* litellm.

3. Third attempt — actually diff renders. Compose was recreating
   open-webui despite identical config blocks. The remaining mechanism
   had to be `depends_on` cascade behavior that the `--config-hash`
   labels don't reveal. The user's suggestion to break the dep ended up
   being the right move; nothing else could have isolated open-webui
   without instrumenting Docker Compose itself.

This is a worthwhile benchmark candidate: the surface symptom ("Open WebUI
keeps bouncing on profile switch") had a misleading first explanation
(force-recreate in our fallback) and even after that was fixed and tests
passed, the real cause was a deeper LiteLLM-API limitation
(`store_model_in_db`) *and* a non-obvious Compose cascade behavior that
only diffing two real renders + reading container shutdown logs would
catch. Each individual step looked complete in isolation. Worth
distilling into `dev/benchmark-candidates/` once we add a few more like
it.

Design takeaways:

1. **Container recreate vs. config refresh is a real distinction the
   admin API forces you to confront.** YAML model_list is convenient
   but it makes `/model/delete` unusable without `store_model_in_db:
   True`. If you want live refresh to work, treat the DB as the source
   of truth and let YAML populate it on startup — don't try to manage
   two parallel stores.

2. **Compose `depends_on` is not free.** It changes recreate-cascade
   behavior in ways that aren't visible in the config-hash labels and
   aren't documented prominently. For long-lived "session UI"
   containers (Open WebUI, dashboards, monitoring frontends), prefer
   no `depends_on` and lean on the dependent service's own retry
   logic. Reserve `depends_on` for true ordering requirements (DB
   migration before app start, etc.).

3. **`os.environ/...` references are LiteLLM-YAML-only.** Anything
   that round-trips YAML → admin API needs to resolve them first.

4. **Pre-flight checks need self-ownership awareness.** A port-in-use
   check that doesn't differentiate "another project owns this" from
   "we own this and compose is about to recreate it" breaks `up`/
   `switch --apply` on every running stack. The fix is a one-line
   `docker compose ps --format json` query to extract our own
   published ports and exclude them.

5. **The user's first guess is often the right framing.** "Maybe we
   need to make openweb-ui not depend on litellm?" — I had been
   investigating it as a config-hash mystery for thirty minutes;
   the user reframed it as a dependency-graph problem and the fix
   was one YAML stanza deletion. Surface that level of intuition
   earlier next time by stepping back from the mechanism after one
   failed hypothesis.

Validation:

```bash
vllm-stack render --yes
vllm-stack down && vllm-stack up -d   # pick up depends_on removal + store_model_in_db
vllm-stack switch <other-profile> --apply
docker inspect open-webui --format='{{.State.StartedAt}}'  # before
docker inspect open-webui --format='{{.State.StartedAt}}'  # after — should match
```

Tests: all 65 pass after each step.

---

## 2026-05-20 15:09:24 -0400

Model: claude-opus-4-7[1m] (Opus 4.7, 1M context).

User intent: rendered Compose / KubeAI artifacts were being written to
`./generated/` inside the repo checkout, which meant two users developing
against the same checkout on one host would stomp on each other's files.
The request is to move the canonical output location off-tree and to a
machine-wide directory while still letting each user keep their own
input profiles (`config.yaml` / `models.yaml` / KubeAI local values).
Suggested default: `/data/service/docker/vllm-stack/generated`.

What I changed:

- New `output.generated_dir` section in `config.yaml`, with a default
  resolver that mirrors `_default_storage_root`: prefer
  `/data/service/docker/vllm-stack/generated` when `/data/service/docker`
  exists, else fall back to `./generated`. Override precedence: CLI flag
  `--generated-dir` → env var `VLLM_SERVICE_GENERATED_DIR` →
  `config.yaml` → default. `setup` bakes the resolved value into
  `config.yaml` so it's visible/editable.
- Plumbed `output.generated_dir` (absolute) into the resolved deployment
  dict alongside `state` and `cluster`. Both backend renderers
  (`compose_renderer`, `kubeai_renderer`), `kubeai_ops`, `exporters`,
  and `verification` read from there. Each falls back to the old
  `<root>/generated` layout when the deployment doesn't carry an
  `output` section, so direct test callers that build a deployment by
  hand keep working without needing to know about the new field.
- CLI `generated_dir()`, `plan_path()`, `kubeai_generated_dir()` now
  take cfg; threaded cfg through every call site. The KubeAI README
  emitted by the renderer references the actual rendered path instead
  of the hard-coded `generated/kubeai/...` so the printed instructions
  stay correct when the output is off-tree.
- Tests: `test_serving_profiles._cfg` pins
  `output.generated_dir = "generated"` so the resolver-populated
  deployment dict points at `tmp_path/generated` on dev machines where
  `/data/service/docker` exists. `test_cli_setup.run_cli` sets the env
  var so the subprocess flow + persisted config.yaml stay anchored on
  `tmp_path`. Added three new targeted tests covering custom output
  dir for compose, kubeai, and `normalized_output` path anchoring.

State of mind / reflections:

The fallback choice was the central design decision. The renderers
could (a) always require `deployment["output"]` to be populated, or (b)
fall back to `<root>/generated` when missing. Option (b) preserved a
lot of direct-call test surface that builds the deployment dict
manually, and matched how `normalized_state` already behaves
(callers can omit the section and get sensible defaults). The cost is
two near-identical fallback stanzas in the renderers and kubeai_ops,
which I considered consolidating into a helper but left inline because
each call site reads slightly different bits of the deployment.

What might break: existing operator workflows running on a host where
`/data/service/docker` already exists will see new renders go to
`/data/service/docker/vllm-stack/generated` even without explicit
config changes. That's the desired behavior, but anyone with a running
stack pinned to `./generated/docker-compose.yml` would need to either
take it down via the old path or override `output.generated_dir` to
`generated` in `config.yaml`. The README documents this; I didn't add a
migration script because the old `down`/cleanup flow is unchanged when
operators point the same `--generated-dir` at the previous location.

Pre-existing test failures: 13 tests in `test_cli_setup.py` fail
identically with my changes stashed — they exercise `render` without
`--yes` via subprocess, which hangs on the Rich confirm prompt. Not
in scope here; flagged but not fixed.

Design takeaways:
1. Output dir is a deployment-shaped fact, not a CLI fact. Putting it
   in the resolved deployment alongside `state` made every backend and
   downstream verifier converge on one source without each rediscovering
   it from cfg/root.
2. When introducing a config section that already has a sensible
   "anchor on root" interpretation, mimicking the existing
   `normalized_state` pattern saves a downstream surprise: relative
   paths in config behave the same way state paths do.
3. Tests that exercise renderers directly (no CLI) are a load-bearing
   constraint when changing where artifacts land. Preserving a
   `<root>/<dirname>` fallback in the renderer itself avoided a sweeping
   test refactor and kept the abstraction useful for ad-hoc tooling.

## 2026-05-27 14:25:00 -0500

Model: claude-sonnet-4-6 (via eval_audit harness).

**User intent:** Rename the package from `vllm_service`/`vllm-stack` to `infer_stack`/`infer-stack`. The motivation was twofold: (1) CLI name (`vllm-stack`) and module name (`vllm_service`) diverged — a persistent source of friction; (2) the service now supports ollama alongside vllm, making the "vllm" prefix misleading. No backwards compatibility needed.

**What changed:**
- Python package directory: `vllm_service/` → `infer_stack/`
- PyPI package name: `vllm-litellm-autoconfig` → `infer-stack`
- CLI entry point: `vllm-stack` → `infer-stack`
- All Python imports: `from vllm_service.*` → `from infer_stack.*`
- Environment variables: `VLLM_SERVICE_*` → `INFER_STACK_*`
- XDG app dirs: `~/.config/vllm_service/` → `~/.config/infer_stack/`
- Kubernetes annotation namespace: `vllm-service/` → `infer-stack/`
- Package description: "vLLM stacks" → "inference stacks"
- `manage.py`, `README.md`, all docs, tests, scripts, examples updated
- `dev/journals/` left untouched (historical record)

**Design notes:**
- `infer_stack` / `infer-stack` achieves the desired consistency: module name uses underscores, CLI uses hyphens, both share the same stem.
- "infer" is deliberately generic — accommodates vllm, ollama, and any future backends (vision, specialized runtimes) without re-renaming.
- Template files like `default-vllm-models.yaml` keep their names — those describe the vllm *backend*, not the package identity.

**Confidence:** High. The rename is purely mechanical (string substitution + directory mv). All 22 Python files compile cleanly. No residual `vllm_service`/`vllm-stack`/`VLLM_SERVICE` references remain outside historical journal entries and the tarball artifact.

**Next steps for the user:** Update `eval_audit` (the parent repo) to reference `infer_stack` instead of `vllm_service` in any imports, config references, or scripts that use the submodule. The GitHub remote rename is also handled separately by the user.

## 2026-06-16 14:43:08 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code.

**User intent:** Begin the long-discussed `infer-stack` redesign toward a
leasing/controller model so multiple users running multiple `kwdagger`
pipelines can `acquire` the models a node needs, block until ready, and
`release` after — instead of the current single global "active profile" that is
last-render-wins under a shared data dir. The full critique + staged plan lives
in the *parent* repo at `aiq-eval-runner/dev/infer-stack-redesign-critique.md`
(grounded in the two real consumers: aiq-eval-runner's Incubilate/Contextual
Drag and eval_audit's HELM/kwdagger path). User explicitly asked to start on a
new branch and bump versions; chose sqlite for the store (open question #1 in
that doc). This is Phase 1's foundation.

**What I built (branch `dev/leasing-controller`, version 0.6.1 -> 0.7.0):**
A new backend-agnostic `infer_stack/leasing/` subpackage — the *controller*
half, deliberately separate from the existing stateless compiler
(`resolver`/`validator`/`renderer`), which is kept intact.
- `models.py`: `EndpointRequest` (catalog-resolved input), `Lease`,
  `DeploymentGroup`, the `compatibility_key` (pure structural identity hash),
  `vllm_structural`/`ollama_structural` builders, and `capacity_satisfies`
  (subsumption, not equality).
- `store.py`: `SqliteStore` — autocommit + WAL + `busy_timeout`, with a
  `transaction()` context manager that issues `BEGIN IMMEDIATE` so the
  find-or-create-group critical section is race-safe across processes.
- `ledger.py`: `Ledger.acquire/release/renew/sweep/reclaimable_groups/status`.
  Demand = count of *protecting* leases (ACTIVE and not past TTL), computed by a
  SQL `COUNT(DISTINCT lease_id)` join, not stored. Groups flip LIVE<->IDLE on
  demand; teardown of IDLE groups is left to the future reconciler per reclaim
  policy.
- 17 unit tests + 3 xdoctests, all passing; ruff clean; existing fast suite
  unaffected. Ran via `uv run --extra tests --with xdoctest` (xdoctest is in
  pyproject addopts but not the tests extra — had to add it; the system python
  has no pytest).

**Key design decisions / reflections:**
- The central reframe is compiler-vs-controller. I resisted bolting acquire/TTL
  onto the resolver; the ledger is a separate stateful layer with its own store.
  This is what makes "last render wins" go away: desired state becomes the union
  of live leases, not one mutated profile.
- Coalescing grain = "the thing that gets a process": per-model for vLLM,
  per-daemon for Ollama. Modeled uniformly by making the Ollama structural key
  the *host config* (no tag), so many tag endpoints collapse to one group whose
  `served` map accumulates the tags. This kept one code path for both engines.
- Compatibility is structural-equality + capacity-subsumption, NOT a single
  equality key. A 32k deployment serves an 8k request; the reverse makes a new
  group. Sharing policy is handled in the ledger (dedicated => always new),
  deliberately kept OUT of the hash so the key stays a clean deployment identity.
- TTL is soft and is the crash backstop: protection lapses by time immediately
  in the demand query; `sweep()` only *materializes* the EXPIRED state for
  status/reclaim. A job killed by SIGKILL (no release) is recovered by TTL.

**Uncertainties / risks / what might break:**
- The ledger is pure bookkeeping with NO backend yet — nothing realizes,
  probes, or tears down deployments. `AcquireResult.groups` is the handoff
  point for the next slice (reconciler + 4-method backend protocol + readiness
  with the Ollama pull/warmup rung).
- `EndpointRequest` is hand-built in tests; the catalog.yaml parser that
  produces them (new schema: models/endpoints/hosts/bundles) is not written.
- Cross-process concurrency relies on `BEGIN IMMEDIATE` + WAL + busy_timeout; I
  reasoned it through and unit-tested the single-process invariants, but did not
  write a true multi-process stress test. If contention shows up, that's where
  to look first.
- Group `spec` is taken from the first creating request; coalesced requests'
  specs are assumed compatible by construction. Correct given the compat key,
  but worth remembering when the backend starts consuming `spec`.

**Confident about:** the object model and the coalescing/TTL/demand semantics
(well covered by tests), the layering (controller cleanly separated from the
compiler, no edits to existing modules beyond the version bump), and that this
is the right minimal foundation for Phases 1-4.

**Next steps:** catalog.yaml parser -> EndpointRequest; the Reconciler + Backend
protocol (`realize/teardown/observe/probe_ready`) with Compose first (reuse
`_first_fit`, `_wait_until_ready`, the live LiteLLM router refresh); then the
`acquire`/`release`/`run`/`status` CLI emitting the `contracts.py` env-file.

## 2026-06-16 15:05:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Continuation of
the same session; committed the ledger as `b1ebf17`, then built the next stage.

**What I built (same branch `dev/leasing-controller`):** the serving-catalog
parser `infer_stack/leasing/catalog.py` — the declarative input side of the
controller and the new replacement for "profiles" as the primary unit. Schema:
`models` / `endpoints` / `runtime_hosts` / `bundles`. `Catalog.from_dict`/`load`
parse + validate (cross-reference checks: vllm endpoint -> model, ollama
endpoint -> host, bundle -> endpoints, known engines). `resolve_endpoint` and
`resolve_names` turn endpoint/bundle names into the ledger's `EndpointRequest`s,
so the catalog composes directly with what I built in the prior commit. 15
catalog tests incl. ledger-integration (alias endpoints coalesce; an ollama
bundle collapses to one daemon group serving both tags), 32 leasing tests total,
4 xdoctests, ruff clean.

**Decisions / reflections:**
- Kept the catalog in the `leasing/` subpackage (not the legacy top-level
  `catalog.py`) — it is the *serving* catalog for the new model and should not
  be conflated with the old vllm_models/profiles catalog.
- `served_name`/`public_name` is the endpoint's exposed name; the alias case
  (two endpoint names, same model+runtime) intentionally produces the same
  `compat_key`, so distinct catalog entries still coalesce onto one deployment.
  This validates the "compat key = deployment identity, not endpoint name"
  decision from the ledger stage.
- vLLM `EndpointRequest.spec` carries hf_model_id + runtime + reclaim; ollama
  `spec` carries host/image/settings/gpu_indices + reclaim. These are the
  payloads the (not-yet-written) backend will consume to realize a process.

**Risks / unknowns:** still no backend — `spec` shapes are my best guess at what
compose/kubeai renderers will need and may shift when the reconciler lands. The
new catalog schema is not yet wired to any CLI or migrated from the existing
`config.yaml`/profiles (the legacy `_legacy_profile_to_stack` migration is
future work). No backwards-compat shim yet between the old and new catalogs.

**Next:** the Reconciler + a 4-method Backend protocol
(`realize`/`teardown`/`observe`/`probe_ready`), testable first with a fake
backend, then a Compose implementation reusing the existing placement/readiness
machinery.

## 2026-06-16 15:30:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session;
committed the catalog as `1274470`, then built the reconcile layer.

**What I built (same branch):** the backend seam + controller.
- `backend.py`: a tiny 4-method `Backend` Protocol
  (`realize`/`teardown`/`observe`/`probe_ready`, all idempotent) + `Readiness` +
  a `MemoryBackend` (records calls, configurable per-group/per-endpoint
  readiness) for tests and dry-runs. This Protocol is the single thing a real
  backend (Compose/KubeAI) must implement — and the deliberate line between
  "infer-stack coordinates" and "the backend/k8s schedules".
- `controller.py`: `Controller(ledger, backend)` with the desired-vs-actual
  reconcile loop, `wait_ready`, and thin `acquire`/`release`. 11 controller
  tests (incl. catalog integration), 43 leasing tests total, 5 xdoctests, ruff
  clean.

**Decisions / reflections:**
- Named the orchestrator `Controller` (module `controller.py`) and kept the
  reconcile *loop* as its `reconcile()` method. The design doc says "Reconciler";
  I went with Controller because the object also owns the thin acquire/release
  orchestration, and "reconciler" reads odd as the home of `acquire`. The loop
  is still literally a reconciler.
- `reconcile()` calls `ledger.sweep()` first. Without it, a group whose only
  lease has expired by time still has `state == LIVE` in the row (nothing
  flipped it), so it would stay "desired". Sweeping makes TTL self-enforcing on
  every reconcile — the crash backstop actually fires.
- Desired set = LIVE groups + IDLE groups whose reclaim policy is keep-warm.
  Teardown falls out of the diff (actual - desired). I deliberately did NOT add
  a STOPPED group state: after teardown the backend's `observe()` no longer
  returns the group, so it isn't re-torn-down, and a later acquire reuses the
  IDLE row and re-realizes it. Fewer states, same behavior.
- `wait_ready` is scoped to the endpoints the *lease* requested, not all of a
  coalesced group's served endpoints. Important for Ollama: a daemon may serve
  several tags; a node acquiring one tag shouldn't block on a sibling tag's
  health. Tested explicitly.

**Risks / unknowns:** still no real backend — `MemoryBackend` proves the
control logic but not the docker/GPU reality. The reclaim model has no
*pressure* concept yet: keep-warm idle groups stay up forever (reclaimed only
when explicitly stop/scale-to-zero, or reused). That's intended for now but the
Compose backend will need a pressure-driven reaper when GPUs are contended.
`realize` is per-group; the Compose backend renders one compose file for the
union, so it will likely converge the whole desired set at once rather than
per-call — the Protocol allows either, but I haven't validated the batch shape.

**Next:** the Compose backend implementing this Protocol (reuse `_first_fit`,
`_wait_until_ready` + Ollama pull/warmup, the live LiteLLM router refresh), then
the `acquire`/`release`/`run`/`status` CLI emitting the `contracts.py` env-file.

## 2026-06-16 16:05:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session;
committed the controller as `b512c62`, then built the CLI surface.

**What I built (same branch):** the leasing CLI verbs + the env-file/descriptor
emitter + a dry-run backend.
- `leasing/envfile.py`: the endpoint descriptor (aligned with `contracts.py`:
  base_url / api_key_env / models) and its sourceable shell env-file
  (`INFER_STACK_SESSION_ID`, `OPENAI_BASE_URL`, `INFER_STACK_ENDPOINT_<NAME>`,
  `INFER_STACK_MODELS`, optional `CUDA_VISIBLE_DEVICES`). `read_session_id`
  recovers the session from a written env-file so `release --env-file` works.
- `leasing/backend.py`: added `NullBackend` (dry-run) — `observe()` returns the
  empty set so the controller no-op-realizes everything and never tears down;
  readiness is immediate; no in-memory state, so it stays coherent across
  separate CLI invocations (the persistent ledger is the only truth).
- `cli/commands_leasing.py`: `acquire` / `serve` / `release` / `renew` / `run` /
  `leases`, registered on `ManageCLI`. `run -- <cmd>` acquires, injects the
  endpoint env into the child, runs, releases on exit, propagates exit code.
- 13 CLI tests + a real end-to-end shell smoke through `python -m
  infer_stack.cli`. 66 leasing/CLI tests total, 6 xdoctests, ruff clean.

**Decisions / reflections:**
- Named the status verb `leases` (not `status`) because the legacy profile model
  already owns `infer-stack status`. The two models coexist during the
  transition; `leases` is unambiguously the new one.
- Default `--backend null`. The whole acquire/release/run/env-file flow is
  exercisable without docker, which is the point of doing the CLI before the
  Compose backend: it locks the user-facing contract (verbs + env-file shape)
  that the consumer repos and the Compose backend then build against.
- Hit the scriptconfig smartcast trap: a `nargs='*'` positional smart-splits
  string elements on commas, which mangled `run -- python -c "import os, ..."`
  into a nested list. Fix: `type=str` on the positional/`command` values and do
  comma-splitting myself in `_collect_names`. (Worth a dev/lesson if it recurs.)
- For Ollama the env-file's request-model name is the *tag* (e.g. `qwen3.5:4b`),
  not the endpoint name — confirmed in the smoke (`INFER_STACK_MODELS=qwen3.5:4b`).
  `build_descriptor` reads it from the group's per-endpoint served payload.

**Risks / unknowns:** `base_url` in the descriptor is a dry-run placeholder
(`--base-url`, default the LiteLLM port) — the real URL must come from the
Compose/KubeAI backend once it exists, so the descriptor's `base_url` and
`api_key_env` will be backend-supplied later (likely a 5th backend method or an
access() call). `run` releases in a `finally`, but a hard kill (SIGKILL) skips
it — that's exactly why `run` defaults `--ttl 2h` as the backstop, and why a
periodic `sweep` (who runs it?) is still an open v1 question. Concurrency across
processes is still only single-process tested.

**Next:** the Compose backend implementing the `Backend` Protocol — the first
time this touches docker/GPUs (renderer adapter from DeploymentGroups, GPU
placement over the union, cross-process compose-file lock).

## 2026-06-16 16:35:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session;
committed the CLI as `0100a52`. Starting the Compose backend, which I'm building
in fully-testable sub-slices because this sandbox has no GPUs/docker.

**What I built (same branch):** sub-slice 1 — the single-host GPU placement
planner `leasing/placement.py`. `plan_placement(groups, inventory, *,
allowed_gpus, reserved, pinned, skip_display) -> GpuPlan`. Reuses the resolver's
`_first_fit` and `_available_gpu_indices`. 15 tests (simulated inventories),
71 leasing/CLI tests total, 7 xdoctests, ruff clean.

**Decisions / reflections:**
- This is the placement-over-the-union the ledger deliberately deferred. I made
  it a standalone module (not buried in the Compose backend) because it is pure,
  fully testable without docker, and Phase-2 raw-GPU reservations need the same
  `reserved` exclusion.
- Three-tier deterministic order — pinned, then explicit, then first-fit — with
  groups sorted by (created_at, id). `pinned` is the stability mechanism: a
  group already running on [0,1] keeps it when a sibling is added, so reconciles
  don't reshuffle live models. An invalid pin (GPU gone) silently re-places.
- Ollama daemons pin explicitly (their host `gpu_indices`, possibly `[]` for
  CPU); vLLM groups first-fit by `tp × dp`. One planner, both engines.
- Deferred VRAM-fit validation: the leasing catalog doesn't carry per-model
  memory yet, so placement is by GPU *count* only. The legacy validator's
  VRAM/headroom check is the reference to port when ModelSpec grows a
  `min_vram_gib` field.

**Risks / unknowns:** still no docker. The next sub-slices (render
DeploymentGroups -> compose artifacts + `docker compose up`; observe via
`docker compose ps`; teardown) will be built behind an injected runner seam and
unit-tested against fakes, but the real docker/GPU path can only be validated by
the user on the GPU host — I'll mark that explicitly. The big open design choice
for the next slice: reuse the legacy `resolve()` + compose renderer by
synthesizing a config from the groups, vs. write a focused renderer straight
from DeploymentGroups. Leaning toward a focused renderer to avoid forcing the
new model through the legacy config schema, but will decide after reading the
renderer's exact inputs.

**Next:** Compose render/realize/observe/teardown behind a docker-runner seam.

## 2026-06-16 17:10:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session;
committed placement as `3f136a3`. Built Compose backend slice 2.

**What I built (same branch):** `leasing/compose.py` — a focused compose
renderer + `ComposeBackend`, plus a controller `converge` branch.
- Decided **focused renderer over reusing legacy `resolve()`/`render_compose_artifacts`**:
  the legacy renderer is welded to the resolved-deployment v5 schema
  (providers/gateways/frontends, jinja templates, diff-gated writes); synthesizing
  that from DeploymentGroups would be a large fragile mapping. The focused
  renderer emits a compose dict straight from groups and *reuses
  `profile_runtime.vllm_args`* for the genuinely tricky vLLM flag-building.
- `ComposeBackend` is converge-style: place (pinned from a persisted sidecar) ->
  render -> write compose.yml + sidecar -> `docker compose up -d
  --remove-orphans`. observe() parses `docker compose ps --format json` and maps
  running services back to group ids via the sidecar. Docker is an injected
  `run(args)->str` seam.
- Controller `reconcile` now prefers `backend.converge(desired)` when present
  (whole-union convergence) and falls back to the per-group realize/teardown
  loop for Memory/Null backends. Computed realized/torn_down from before/after
  observe() diffs.
- Wired `--backend compose` into the CLI (`detect_inventory`, data_root state
  dir, `--allowed-gpus`). 90 leasing/CLI tests, 7 xdoctests, ruff clean. Tested
  with a *stateful* FakeDocker (`up` reflects the rendered compose file, `ps`
  returns it, `--remove-orphans` drops gone services) so converge->observe is
  coherent end to end without docker.

**Decisions / reflections:**
- Per-group `realize`/`teardown` never fit compose's one-file model; rather than
  contort the protocol I added an optional `converge` the controller detects via
  `hasattr`. ComposeBackend therefore implements converge/observe/probe_ready
  (not realize/teardown) — it duck-types into the controller's converge path.
- `--remove-orphans` *is* teardown: converge always renders only the desired
  groups, so a dropped group's service becomes an orphan and is removed. No
  separate teardown docker call needed.
- Placement stability via the persisted sidecar: converge feeds current
  assignments as `pinned`, so adding/removing a group never migrates a running
  model's GPUs.

**Risks / unknowns (for when you test on the GPU host):**
- Readiness is only "container running" this slice. `acquire --backend compose`
  returns once the vLLM *container* is up, NOT once the model has finished
  loading/serving. Real generation-probe readiness + Ollama pull/warmup come in
  slice 3.
- The env-file `base_url` is still the `--base-url` placeholder (default
  :14042). With compose-and-no-LiteLLM the real endpoint is the per-vLLM direct
  port (18000+i). The LiteLLM front door in slice 3 is what makes one stable
  base_url correct; until then pass `--base-url` explicitly or hit the direct
  port.
- Cross-process compose-file writes are not yet file-locked — concurrent
  `acquire`s from two processes could race on the compose file. The ledger's
  BEGIN IMMEDIATE covers the lease rows but not the file; a flock around
  converge is on the slice-3/hardening list.
- No real docker/GPU run in this sandbox; the FakeDocker proves the logic. You
  validate the real path.

**Next (slice 3):** LiteLLM front door (alias routing -> served names, real
base_url) + real readiness (`_wait_until_ready` generation probe, Ollama
pull/warmup) + a converge file-lock.

## 2026-06-16 17:45:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session;
committed compose slice 2 as `301d4de`. Built slice 3.

**What I built (same branch):** the LiteLLM front door + access/descriptor
wiring + converge file-lock + routability readiness, all in `compose.py` +
`envfile.py` + `commands_leasing.py`.
- LiteLLM service + config rendered from the group set (default on): one
  `model_list` entry per served endpoint alias -> upstream `openai/<served>` at
  `http://vllm-<gid>:8000/v1` (or `ollama/<tag>`). This is what makes one stable
  `base_url` possible and is why the alias (endpoint name), not the upstream
  served name, is what a client requests.
- `ComposeBackend.access(endpoints)` returns the real `base_url` (LiteLLM port),
  `api_key_env`, and per-endpoint request names. The CLI's new `_descriptor_for`
  prefers `backend.access` over the `--base-url` placeholder; `build_descriptor`
  gained a `request_names` override. NullBackend has no `access` -> CLI falls
  back unchanged (verified by smoke).
- `probe_ready` upgraded: container-running (observe) AND the alias appears in
  the gateway's `/v1/models`, via an injected `http_get` seam. Tested with a
  fake HTTP that lists whatever the rendered LiteLLM config declares — realistic
  and fully offline.
- `converge` wrapped in an `fcntl.flock` (the cross-process compose-file race I
  flagged in slice 2). 94 leasing/CLI tests, 7 xdoctests, ruff clean.

**Decisions / reflections:**
- Backend-supplied access (a 5th, optional method discovered via `getattr`) is
  the clean fix for "who knows the real base_url": the backend does, not the
  CLI. Same hasattr/duck-type pattern as `converge`. Keeps Null/Memory backends
  untouched.
- Readiness = "alias listed by the gateway" is the honest, testable middle
  between "container up" (too weak) and "generation succeeds" (needs a live
  model). A real generation probe + Ollama pull/warmup is the remaining rung.
- Caught a test-helper bug while wiring this: my `vllm()` test helper had keyed
  the group's `served` map by the *served name* instead of the endpoint alias,
  so the LiteLLM alias didn't match the probed endpoint. Fixed the helper to key
  by endpoint (group id) with served_model_name as the value — which is exactly
  the endpoint-vs-served distinction the front door depends on.

**Risks / unknowns (for host testing):** still no real docker/HTTP here — the
seams prove the logic; the LiteLLM container actually starting, vLLM upstreams
becoming healthy, and the gateway listing models is the host-validated part.
Ollama pull/warmup not implemented (a daemon serves a tag lazily; readiness
should pull + warm it). The flock only guards a single host's state dir.

**Next:** the migration bridge (`migrate-config` from legacy profiles; `switch`
as a standing-lease shim) and Ollama pull/warmup readiness; then consumer
integration in eval_audit / aiq-eval-runner.

## 2026-06-16 18:30:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session.
User redirected: drop the migration-bridge / back-compat work (no backwards
compatibility needed) and instead make the code more maintainable. They chose
"consolidate shared machinery" — fold duplicated logic into one reused core,
without deleting the legacy CLI surface.

**What I did (two tested commits):**
1. `9c8e05a` GPU primitives -> `hardware.py`: `leasing/placement` had been
   importing the resolver's *private* `_first_fit`/`_available_gpu_indices`.
   Moved those + `_resolve_gpu_indices` to `infer_stack.hardware` as public
   functions; resolver imports them back under the old private aliases (zero
   internal churn), leasing imports the public names. One home for GPU-pool
   placement.
2. HTTP probes -> new `infer_stack/probe.py`: `cli/probes.py` had
   `_ready_openai_probe`/`_ready_ollama_probe` (pure HTTP, but stuck in the cli
   layer), while `leasing/compose` had its own weaker `/v1/models` check. Moved
   the probes to a layer-neutral `probe.py` over an injectable `http` client;
   `cli/probes` re-exports them under the old names for legacy callers; leasing's
   `probe_ready` now calls `openai_ready(require_listed=True,
   require_generation=...)`, so it gained the advertised-alias check and an
   optional real-generation rung for free.

**Decisions / reflections:**
- Verified the test patch mechanism before moving the probes: the suite does
  `monkeypatch.setattr(cli_mod.requests, 'get', ...)` — patching the shared
  `requests` *module object*'s attributes, which intercepts `requests.get`
  everywhere regardless of which module calls it. So relocating the probe body
  was safe; full suite (197) confirms it.
- Kept the schema-tied helpers (`_default_model_for_deployment`,
  `_resolve_smoke_protocol_from_deployment`) in `cli/probes` — they read the
  legacy v5 deployment dict and aren't shareable. Only the pure HTTP probes
  moved.
- Layering: `probe.py` and the GPU primitives in `hardware.py` are below both
  `cli` and `leasing`, so neither consolidation created an upward import. This
  is the right shape — shared mechanism sinks to a common low layer.
- Standardized leasing's HTTP seam from a bespoke `http_get(url)->(status,body)`
  callable to a requests-like `http` client (`.get`/`.post`), matching the probe
  and the real `requests` — one seam shape across the codebase.

**Risks:** behavior-preserving by construction; the only behavior *change* is
leasing readiness getting stricter (alias must be advertised), which matches
intent. Full suite green. Still no real docker/HTTP here — seams prove it.

**Next:** Ollama pull/warmup readiness (could reuse `probe.ollama_ready`), then
consumer integration; further consolidation candidates if any surface (the
LiteLLM config template vs the leasing dict builder is the remaining duplicate,
but template-vs-dict makes it lower-value).

## 2026-06-16 18:55:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Same session.
Did the next functional step: Ollama pull/warmup readiness.

**What I built:** `ComposeBackend.probe_ready` now handles the Ollama lazy-load
gap. An Ollama daemon being up (and the alias being in the LiteLLM config) does
NOT mean the tag is resident, so for ollama groups probe_ready:
1. pulls the endpoint's tag into the daemon via `docker compose exec -T
   ollama-<gid> ollama pull <tag>` (idempotent; tracked in `self._pulled` so
   it's issued once per process; failures are retryable so the poll loop heals
   once the daemon accepts the pull), then
2. forces `require_generation=True` so the front-door readiness check runs a
   real chat through LiteLLM->ollama — which both confirms the tag loads and
   warms it.
vLLM readiness is unchanged (alias-listed by default); a new `--require-generation`
CLI flag opts vLLM into the same real-generation check.

**Decisions / reflections:**
- Kept everything through the LiteLLM front door (one base_url) rather than
  reusing `probe.ollama_ready` against the daemon directly — the only
  ollama-specific action is the *pull*, which needs no host-port tracking
  (`exec` by service name). So I did NOT end up calling `probe.ollama_ready`
  here; it stays for the legacy direct-ollama path. The unifying insight: with a
  gateway, readiness is engine-agnostic (openai_ready) and the only divergence
  is the side-effecting pull.
- Pull lives in probe_ready (not converge) on purpose: right after `up -d` the
  daemon isn't ready to accept a pull, and the poll loop is exactly the retry
  mechanism. Idempotence + `_pulled` keep it from re-pulling.
- "alias listed" is too weak for ollama (LiteLLM lists configured-but-unpulled
  tags), which is why ollama forces generation — the honest readiness signal.

**Risks:** as ever, no real docker/ollama here — FakeDocker handles `exec` and
the pull command is asserted; the real pull (which can download GBs and block
the first poll) is host-validated. 17 compose tests (3 new ollama), 88
leasing/CLI + 7 xdoctests green.

**Next:** consumer integration — wrap eval_audit's MaterializeHelmRunNode command
with `infer-stack run`, and switch aiq-eval-runner Incubilate to acquire/release.

## 2026-06-17 12:20:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. User started
testing the Compose backend on real hardware (yardrat: 2 GPUs — RTX 8000 free,
RTX 5000 display-attached; both Turing/sm_75, fp16-only).

**First real-hardware bug (F1), FIXED:** `docker compose up` rejected the
rendered file — `_gpu_reservation` emitted `capabilities: [['gpu']]` but the
Compose schema wants a list of *strings* (`capabilities: ['gpu']`). My nested
list was a python-on-whales-ism. Fixed + locked with a test asserting
`devs['capabilities'] == ['gpu']`. This is exactly the class of bug the unit
tests couldn't catch (they asserted device_ids, never the capabilities shape,
and no real `docker compose config` validation runs offline).

**Wrote `dev/leasing-test-plan.md`** — a living runbook: setup, 9 staged test
blocks (each self-contained, re-exporting its own env so blocks don't depend on
each other), a findings/fixes log, debug-capture, and cleanup. Tailored to
yardrat: Turing `--dtype=half` in every vLLM endpoint, tiny ungated chat models,
and explicit calls-out of the predicted issues.

**Predicted issues now documented as findings to confirm/fix:**
- F2 (setup): the readiness probe must send `LITELLM_MASTER_KEY`; export it or
  readiness 401s. The container reads the same var via Compose interpolation.
- F3: vLLM without `--require-generation` is false-ready (alias listed before
  vLLM loads; `depends_on` doesn't gate on health).
- F5 (OPEN): display-active GPU 1 is skipped and there's no CLI knob to include
  it; `--allowed-gpus` filters after the skip. Blocks 2-distinct-model tests.
- F6 (OPEN): no first-class dtype/protocol/image knobs — dtype via extra_args,
  readiness always chat (completions-only models fail), images pinned in config.

**Reflection:** the offline seam tests (FakeDocker) gave false confidence on the
rendered artifact's *schema* — they checked the dict we built, not whether
docker accepts it. Added the guard: a parametrized test that writes the rendered
project and runs `docker compose config -q` (skipped where docker compose is
absent). It runs against real `docker compose` even in this sandbox (~sub-second
each) and now passes for vllm-single / vllm-tp2 / vllm+ollama / no-litellm — so
F1 is verified at the schema level, not just the dict.

**Next:** keep fixing what yardrat surfaces (likely F2/F3 ergonomics next, maybe
the F5 skip_display knob so both GPUs are testable), updating the findings log.

## 2026-06-17 13:05:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. yardrat reported
8 pytest failures — all in the *legacy* `test_cli_meta.py` / `test_cli_setup.py`
(setup/render/status/config-paths), none in the leasing code, and the full suite
passes clean here. Root cause (F7, FIXED): both files' subprocess `run_cli`
helpers anchored the tmp dirs with `env.setdefault('INFER_STACK_CONFIG_DIR'/
'DATA_DIR', tmp_path)`. The user had `INFER_STACK_DATA_DIR=~/infer-stack-test/data`
exported from the test-plan blocks, so `setdefault` was a no-op and the tests
read the real data dir — which also held a live `infer-stack` compose project
(hence the `docker compose ps` table leaking into the status test). Reproduced
locally by exporting the same var (got the identical failure), fixed by forcing
the vars (`env[...] = tmp_path`), and verified the fix makes all of test_cli_meta
+ test_cli_setup pass *even with the ambient var set* (29 passed).

**Takeaway:** `os.environ.copy()` + `setdefault` is a latent isolation hole for
any test that anchors via env vars — an ambient export silently wins. Force the
vars. (The in-process `_anchor_paths` used `monkeypatch.setenv`, which already
forces, so those tests were fine — the bug was only in the subprocess helpers.)
This is a test-isolation bug, not a regression from the leasing branch; worth a
dev/lesson if it recurs.

## 2026-06-17 13:40:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. yardrat re-ran
acquire and hit the *same* capabilities error — but now during `ps`, not `up`
(F8). Root cause: the bad `docker-compose.yml` from run #1 (pre-fix, with
`capabilities: [[gpu]]`) was still on disk, and `reconcile`'s converge branch
calls `observe()` (→ `docker compose ps`, which validates the file) *before*
`converge()` rewrites it. So a stale/invalid file crashed acquire before the fix
could take effect. Fixed by making `observe()` best-effort: catch any
docker/parse error and return an empty set, so converge overwrites the file and
self-heals. Test added with a runner that raises on `ps`.

**Takeaway / design lesson:** a reconcile step that *reads* actual state must be
tolerant — `docker compose ps` doubly so, because it validates the whole file,
not just lists containers. Read-side operations should degrade to "unknown ->
empty", never abort the write that would fix the very file they choke on. The
generic shape: in a desired-state reconciler, observe() must never be able to
prevent converge().

## 2026-06-17 14:30:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. Two more yardrat
findings + an ergonomics fix the user flagged.

**F9 (FIXED):** vLLM v0.19.1 rejects `--disable-log-requests` (vllm_args emitted
it unconditionally) -> vLLM crashed -> LiteLLM "Connection error". Dropped the
flag (only the leasing + kubeai renderers used it; the legacy compose template
never did; no test asserted it). Engine-version-specific flags belong in
`extra_args`.

**Ergonomics regression the user caught:** the leasing Compose backend had made
the *user* invent + export `LITELLM_MASTER_KEY` (probe read os.environ;
container got it via `${...}`). The legacy renderer instead *owned* the secret
(ensure_secret -> .env) and you fetched it with `infer-stack env`. Restored:
ComposeBackend.master_key() manages the key in the state dir's `.env` (reuse if
pinned, else generate), bakes the *literal* into the LiteLLM service (so
container + probe agree regardless of shell), uses it for the probe, and
access() returns it; the env-file descriptor now carries `OPENAI_API_KEY` (+
OPENAI_BASE_URL) so `source is.env` configures an OpenAI client outright; new
`infer-stack secrets [KEY]` prints managed secrets. Baked the literal (not
`${...}`) on purpose: the user had the var exported, and compose lets shell env
override `.env`, which would desync container vs probe.

**Wording:** user softened "never user-supplied" (a user *can* pin a key in
`.env`; ensure_secret reuses it). Aligned the master_key() docstring + test-plan
F2 to "managed: reused if pinned, else generated".

208 tests pass; ruff clean.

**Still TODO this turn:** verify `infer-stack config paths` and `infer-stack
status` are correct/relevant post-refactor (maybe elevate `config paths` ->
top-level `paths`). Doing that next.

Did it: both worked but were legacy-only. Added a `leasing` group to `config
paths` (ledger, compose dir + docker-compose.yml/litellm_config.yaml/.env/
sidecar), exposed a top-level `infer-stack paths` alias, and gave legacy
`status` a one-line leasing summary (active leases / live groups -> `infer-stack
leases`). 13 meta tests + the legacy status test pass.

**Process scar (my mistake):** to check whether some `ruff` errors in cli/ were
pre-existing, I ran `git stash && ruff && git stash pop` in a bash one-liner. A
concurrent `uv run` touched `uv.lock`, the `pop` hit a conflict and aborted
SILENTLY (`>/dev/null`), and the working tree reverted to the last commit —
making it look like all my PART-2 edits had vanished. They were safe in
`stash@{0}`; `git checkout -- uv.lock` then `git stash pop` restored them.
Lesson: never `git stash` uncommitted work in a tree where a build tool mutates
a tracked file (uv.lock) — and never hide `git stash pop` output. To check
pre-existing lint, use `git show HEAD:file | ruff -` or a worktree, not stash.

## 2026-06-17 15:30:00 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code. User: `infer-stack
logs -f` failed for a leasing user ("No config.yaml found ...").

**Root cause:** the day-2 compose wrappers (logs/ps/restart/pull/start/stop) were
bound to the legacy model — `config_for_runtime(config)` (needs config.yaml) +
`_compose_base_cmd(cfg)` (targets generated/docker-compose.yml). The leasing
model has no config.yaml and its compose lives at
`data_root/leasing/compose/docker-compose.yml` (project `infer-stack`).

**Fix:** added `_day2_compose_base(config, command)` — prefers the leasing
compose project when its file exists (no config.yaml needed), else the legacy
rendered stack (kubeai still stubs). Routed all six generic wrappers through it
(also collapsed their repeated 4-line config+kubeai+base preamble to one line).
Added `LEASING_PROJECT='infer-stack'` constant in leasing/compose.py so the
project name is shared, not duplicated. Smoke: `infer-stack ps`/`logs` now run
against the leasing compose with no config.yaml. Left ollama-* wrappers legacy
(they `exec` a fixed `ollama` service; leasing names daemons per-group — a
separate fix if needed).

**Precedence decision:** prefer leasing-if-its-file-exists. A pure-legacy user
has no leasing file -> unchanged. A user who once leased keeps the (possibly
empty) leasing compose file, so their day-2 ops point at the leasing project —
acceptable given leasing is the model; documented in the wrapper docstring.

## 2026-06-17 14:35 -0400

Model: claude-opus-4-8 (Claude Code, 1M context).

User intent: "I thought we had an e2e test in dev — write a set of e2e tests
under `dev/e2e_tests` (a developer test, not a polished unit test) that
exercises the new leasing tools, so I can rsync back a report and you can check
everything is wired up correctly, efficiently, and ergonomically."

What I found: there was no executable e2e test — only `dev/leasing-test-plan.md`,
a copy-paste runbook. The gap was real.

What I built: `dev/e2e_tests/` — a self-contained bash harness.
- `lib.sh`: a step/assert DSL (`step` / `run` / `expect_rc|out|re|file*` /
  `count_out` / `note` / `skip` / `end_step`). Every step appends one JSON
  record (id, title, verdict, duration, per-assertion results, log path) to
  `results.jsonl`; the harness never uses `set -e` so one report covers the
  whole run.
- `render_report.py` (stdlib only): doubles as the per-step JSON emitter (so the
  bash side never hand-rolls JSON) and the final assembler — `report.md` with a
  pass/fail/skip summary, a Correctness/Efficiency/Ergonomics "wiring axes"
  rollup, failures-first (with log tails), a full per-section table, and the
  environment capture.
- `catalog.yaml`: committed (not heredoc'd — that bit us before with editor
  merges mangling the runbook's `cat <<YAML`). Validated via `Catalog.load` +
  `resolve_names` (bundle + dup both resolve).
- `tests/*.sh`: tiered. Non-serving (no `--gpu`): environment, dry-run
  (acquire→leases→env-file→release→bundle on the null backend), ergonomics
  (paths/secrets/status/day-2 fallback), negatives (friendly, no traceback).
  Serving (`--gpu`): single-vllm + real chat, coalescing (one container,
  demand 2), dedicated/F5, TTL+reclaim, run-wrapper (env inject + exit-code),
  ollama pull/warmup, concurrency (compose stays schema-valid). `99_cleanup`
  always runs (downs the project) so a run never leaks containers.
- `run.sh`: fresh `INFER_STACK_DATA_DIR` per run inside the results dir (so the
  ledger + rendered compose/litellm/.env travel with the report), `--only`,
  `--keep-running`, prints the exact `rsync` line.

State of mind / confidence: smoke-ran the non-serving tiers locally against the
real null backend via the project venv — 16/16 pass, report + JSONL + skip
records all valid. Confident in the harness mechanics and the dry-run/ergonomics
assertions (they ran for real). The `--gpu` assertions are written against the
documented CLI/JSON shapes and the runbook's verified behavior but have NOT been
executed on hardware yet — that's the point: yardrat runs them and rsyncs the
report back. Risks: a couple of GPU steps assert on substrings (`"demand": 2`,
`"ready": true`, `vllm`/`litellm` in `ps`) that depend on exact CLI output; if
those drift the report will show a clean FAIL with the log tail, which is the
intended feedback loop, not a silent miss.

Takeaways: (1) a runbook and a harness are different artifacts — the runbook is
for a human debugging, the harness is for "run it, rsync the verdict"; keep both,
cross-link them. (2) Put the report emitter and assembler in one stdlib python
file the bash DSL shells out to — bash stays declarative, JSON stays correct.

## 2026-06-17 16:10 -0400

Model: claude-opus-4-8 (Claude Code, 1M context).

User intent (mid GPU e2e run, no errors through step 50): two follow-ups —
(1) an e2e test that exercises BOTH GPUs, and (2) a companion to
leasing-test-plan.md that is demo-minded (not test-minded): a real-world
walkthrough on yardrat that does a real `setup` of the user config + the real
docker-mounted storage location, then deploys a big model behind Open WebUI.
Copy-pastable, no cross-block env deps, and "set up the important stuff once so
the commands are concise — if they aren't, that's a design smell to consider."

What I did:
- Closed F5 (was OPEN). The placer's `skip_display` had no CLI knob, so GPU 1
  (display-attached) was unreachable and "both GPUs" was impossible. Added a
  `_DisplayGpuMixin` (`--include-display-gpus`, alias `--include-display-gpus`)
  to the leasing common flags and wired `_make_backend` to pass
  `skip_display=not include_display_gpus`. Unit-tested the wiring by stubbing
  ComposeBackend + hardware.detect_inventory (213 passed, was 212). Updated the
  test-plan F5 entry to FIXED and CHANGELOG Fixed.
- Added e2e tier `45_both_gpus`: acquires two distinct models with
  `--include-display-gpus`, then asserts the backend sidecar
  (`leasing-compose-state.json`) placed them on two DISTINCT gpus (`[0, 1]`) and
  that both answer through the one gateway. Reads the sidecar rather than
  nvidia-smi because the assignment is the thing under test and it's
  deterministic.
- Wrote `dev/leasing-demo.md`: §1 one-time setup (persist INFER_STACK_DATA_DIR
  in ~/.bashrc as the storage knob + catalog at the default config root so
  serve/acquire need no --catalog), §2 `serve` a 14B model (ungated, fp16/Turing,
  fits the 48 GiB GPU 0), §3 talk to the stable LiteLLM front door using
  `$(infer-stack secrets …)`, §4 a hand-run Open WebUI container pointed at the
  gateway via host.docker.internal with persistent history under the data dir,
  §5 switch models around (the stated point of the tool), §6 teardown.

State of mind / the smell audit: the user explicitly asked me to treat clunky
steps as design smells, so the demo ends with five, ranked: (1) `--backend
compose` repeated on every call (no persisted default backend), (2) storage
location is env/flag-only for leasing — legacy `setup` baked state paths into
config.yaml but the leasing Compose backend reads data_root() directly, so the
durable user config can't express "where my weights live"; that's why §1a writes
~/.bashrc, (3) no endpoint-addressed teardown for standing `serve` leases (must
copy a session id), (4) Open WebUI is unmanaged in the leasing model (legacy
rendered it), (5) HF_TOKEN has no managed slot next to LITELLM_MASTER_KEY. I
believe 1–3 are the cheap, high-value wins and said so. I did NOT implement them
this turn — the ask was to "see first," and they're design changes worth a
decision, not reflexes.

Uncertainty: the demo's GPU/Open WebUI blocks are written to verified
paths/ports/verbs but have not been run on hardware (yardrat run pending);
the 14B first-serve downloads ~28 GiB so the §2 timeout is 3600s. Confident in
the F5 wiring (unit-tested) and that 45_both_gpus skips cleanly off-GPU
(exit, not return — last turn's bug).

Takeaway: when a walkthrough forces you to repeat a flag or hand-run a step the
tool "should" own, that repetition is the spec for the next ergonomic feature —
write the demo first, let it surface the smell, then decide.

## 2026-06-17 16:55 -0400

Model: claude-opus-4-8 (Claude Code, 1M context).

User intent: mid GPU e2e run, `50_coalescing/coalesce-acquire` failed after a
1200s timeout and they had no report (interrupted, no rsync). Two harness asks —
(1) a trap so Ctrl-C still writes the report + rsync line, (2) the rsync must
exclude the heavy cache dirs. They then rsynced the logs so I could diagnose.

Diagnosis from the synced log (a real product bug, not a test flake): alice's
`qwen-small` came up ready on group grp-…; bob's `qwen-dup` (which has
`public_name: qwen-small`) correctly coalesced onto the same group (demand 2,
one container), but its readiness probe waited on alias `qwen-dup` while the
running LiteLLM gateway only knew `qwen-small` → never ready → 1200s timeout.
Root cause: the ledger merge DOES add `qwen-dup` to group.served and converge
re-renders litellm_config.yaml with it, but LiteLLM reads that bind-mounted file
only at startup, and `docker compose up -d` doesn't recreate the litellm service
because its *spec* didn't change — only the mounted file did. So the live
gateway kept the old routes. (Confirmed the controller re-reads groups via
ledger.status() before converge, so the rendered file is correct — only the
restart was missing.)

Fix: stamp a `infer-stack.config-hash` label (sha256 of the rendered config) on
the litellm service. The spec now changes iff the routing changes, so converge
recreates litellm exactly when needed and is otherwise idempotent. Unit test in
test_leasing_compose asserts the label tracks the model_list (changes on a 2nd
alias, stable for identical input). 214 passed.

Also: `coalesce-one-container` falsely failed counting 2 vllm groups — the 2nd
was a reclaim:stop group from tier 40 lingering as state=stopped in the ledger.
Tightened the e2e check to count only LIVE vllm groups. (Whether the ledger
should prune/hide stopped groups from `leases` is a separate question — noted,
not changed.)

Harness work: run.sh now has a `finish` EXIT trap (idempotent) that assembles
the report, writes a `rsync-back.sh`, and prints the summary + rsync line; an
INT/TERM trap exits so finish runs on Ctrl-C. Verified both paths (normal +
SIGINT mid-run → partial report). The data dir lives inside the results dir, so
it also holds the multi-GB HF/kernel caches; the printed rsync line and
rsync-back.sh carry --exclude globs that drop *-cache/, ollama/, open-webui/,
postgres-*/, runtime/ (keeping leasing/: ledger + compose + litellm config +
.env). Moved per-step scratch (.lastout/.asserts/.notes) out of logs/ into a
.scratch/ dir (also excluded) so the rsync'd logs/ holds only real .log files.

State of mind: confident in the litellm-reload fix — the mechanism matches the
symptom exactly and is unit-tested; the recreate cost is a brief gateway blip
only when routing actually changes, which is acceptable under converge. The
"stopped groups linger in the ledger" smell is real but I left it alone (out of
scope, and `leases` showing history may be intentional). Next GPU run should get
50_coalescing fully green and, with the trap+excludes, always return a
reviewable report even on Ctrl-C.

Takeaway: a bind-mounted config is not part of a container's compose identity —
if a sidecar reads its config only at startup, give its service a content-hash
label so converge restarts it when the content changes. Otherwise "I rewrote the
file" silently diverges from "the running process sees the file".

## 2026-06-17 17:25 -0400

Model: claude-opus-4-8 (Claude Code, 1M context).

User intent: GPU e2e run (git b030487, with the between-tier reset but before the
speed commit) — 80_run_wrapper failed; asked me to look. Also asked earlier this
session to make tiers faster (handled in 2abdd3d) and answered the keep-warm /
group-id design questions.

Bug found (real, product): `ComposeBackend.converge` always ran `docker compose
up -d --remove-orphans`, but when the desired set is empty (release the last
reclaim:stop lease → zero services rendered), `up` errors "no service selected"
on a services-less file. So release's reconcile crashed; `infer-stack run`
surfaced it as a non-zero exit even though the chat had already succeeded
(80_run_wrapper run-injects-env got a valid completion, then died on release).

Why it only showed now: my between-tier reset (b030487) isolated tiers, so for
the first time a release actually converged to *empty*. Before, a leftover
keep-warm group always kept ≥1 service and masked it. And only tier 80 catches
it because `infer-stack run` propagates the release rc — the other tiers swallow
release errors (`>/dev/null 2>&1; true`), so they passed while silently failing
to tear down.

Fix: converge tears the project `down` when there are no services, else `up`.
`down` targets the project, so it works on the empty file and is idempotent. Unit
test asserts converge([]) issues `down` and never `up`, and observe() is empty.
215 passed (was 214). CHANGELOG updated.

State of mind: this is the second latent bug the e2e harness surfaced only after
state isolation (first was litellm-reload; both were masked by leftover state).
Reinforces that the reset was the right call — it makes each tier a real
clean-slate test. Confident in the fix; `down`-on-empty is standard compose.
Remaining unrun-on-GPU after the user pulls (speed + this fix): 90_concurrency
(was interrupted), and a full green pass. Group-id compat-key work still queued
behind a green baseline.

Takeaway: "up the union" converge has an edge at the empty union — the teardown
path is not just "up with fewer services", it's a different verb. Test the
convergence to zero, not just to N-1.

## 2026-06-17 21:40 -0400

Model: claude-opus-4-8 (Claude Code, 1M context).

User intent: the demo must use the smol models; and the sprawly flat CLI (~38
top-level verbs) should be reorganized into submodals (aivm-like, "but better"),
with an ergonomic `catalog` editor (no raw YAML), `help tree`, `init`->`config
init`, and a `legacy` holding pen for anything not carried forward. Decisions
(via AskUserQuestion): noun-verb grammar; full phased reorg now.

Did it in 6 committed phases (each with tests + green suite):
1. `catalog` submodal (commands_catalog.py): model/endpoint/host/bundle
   add|list|show|rm + init/path/show/validate/edit. A validating writer
   (Catalog.from_dict) refuses to persist a catalog the leasing path would
   reject. `help tree` walks the modal registry. (Verified scriptconfig 0.9.1
   does 3-level nesting first.)
2. `config` submodal + settings.yaml in paths.py (load/save/get_setting, no
   import cycle). data_root() honors `data_dir`; leasing `--backend` default is
   now None and resolves explicit > `config set backend` > null. Kills two
   ergonomic smells.
3. `secret` get/set/list; `secret set HF_TOKEN=…` writes the managed .env
   (merge-preserving), set before serve. `secrets` kept as alias.
4. `stack` day-2 group (+ new `stack down`); logs/ps kept top-level.
5. `legacy` modal — moved ~25 profile-world verbs under it. Top level is now the
   leasing loop + submodals. Kept tests working by auto-prefixing `legacy ` in
   the two subprocess run_cli helpers (one edit each, not per call-site).
6. Docs: rewrote leasing-demo.md to use catalog/config/secret (no heredoc, no
   per-block exports), marked the resolved smells; cli-redesign.md ->
   implemented; followups + CHANGELOG; README banner pointing at leasing +
   `legacy` prefix (full README rewrite deferred).

State of mind: confident — every phase kept the suite green (215 -> 230) and the
new code is ruff-clean (the only ruff hits are pre-existing I001 import-order in
paths.py/options.py that predate me). The catalog validating-writer is the piece
I'm happiest with: the editor literally cannot write a catalog acquire would
choke on. Risk/uncertainty: the GPU e2e hasn't run against the reorged CLI yet —
but the e2e/demo only use top-level verbs that stayed (acquire/serve/run/
release/leases/secrets/status/logs/ps/paths) so it should be unaffected; worth a
yardrat pass to confirm. The README is now accurate-by-banner but not rewritten.

Takeaway: when reorganizing a CLI built on a modal framework, a registry-walking
`help tree` + a `legacy` bucket let you move fast without a flag-day — and
honoring persisted settings (backend/data_dir) in the *resolution* layer (None
default -> setting -> fallback) avoids the explicit-vs-default ambiguity cleanly.

## 2026-06-18 14:53:48 -0400

Model: claude-opus-4-8 (Claude Code, "fast"/Opus 4.8). Config: default tools,
running from the aiq-eval-runner superrepo against the infer-stack submodule on
branch dev/leasing-controller.

User intent (one prompt, four threads): (1) Open WebUI should be on by default
again — bundled, and crucially NOT torn down when models switch (the legacy
stack worked to keep the UI from blinking); user judged this easier here than
the LiteLLM case. (2) Rename `infer-stack secrets` -> `infer-stack env`,
path-first with optional KEY — secrets live in a readable `.env` so no reason to
hide the path; mirror the legacy `env` ergonomic. (3) Bring back a concise smoke
test so the demo doesn't have to curl (but keep curl shown too). (4) Diagnose a
`litellm-1 InternalServerError: Connection error. Received Model Group=chat`
seen during the demo.

Diagnosis of (4): startup noise, not a real failure. Compose `depends_on` for
litellm waits for the upstream to *start*, not be *healthy*, so during vLLM's
model-load window litellm forwards probe/early requests to a not-yet-listening
`vllm-…:8000` and logs the connection error; once vLLM is up the same route
200s (matches the user's trailing log lines). I deliberately did NOT switch
litellm to `condition: service_healthy` — that would couple every litellm
recreate (which happens on each routing change) to the slowest upstream's
health, delaying routing for already-up models. Instead I added
`router_settings` (num_retries/timeout/cooldown_time/allowed_fails) so the
warmup window is retried/self-healing rather than surfacing as client 500s.

Design decisions worth recording:
- Open WebUI stability across switches falls out of making its compose service
  spec *independent of the model set*: it points at the `litellm` service at a
  fixed internal URL with the (stable) baked master key, so the rendered dict is
  byte-identical every converge. `docker compose up -d` then leaves it running
  while only litellm (config-hash label) is recreated. Verified by a test that
  asserts `open-webui` is unchanged but `litellm` differs after adding a second
  model. This is the same lever the config-hash fix used, applied in reverse:
  put churn in the spec only where you *want* recreation.
- `ui` resolution mirrors the backend/data_dir pattern: tri-state flag
  (`--ui/--no-ui` default None) -> `config set ui` -> default True, resolved in
  `_resolve_ui`. Keeps explicit-vs-default unambiguous.
- `infer-stack test` reads the front door straight from DEFAULT_PORTS + the
  managed `.env` rather than building a ComposeBackend, so it's cheap and needs
  no GPU detection.
- Name collision caught: `EnvCLI` exists in both commands_runtime (legacy) and
  now commands_leasing; imported the leasing one `as LeasingEnvCLI` so the
  legacy modal's `env` keeps resolving to the runtime class.

Risks/uncertainties: WEBUI_AUTH=False is fine for a single-user workstation but
must gain an auth/port knob before a shared host (noted in followups). The
managed UI binds host port 13000 by default now on every compose serve — a
behavior change a user could be surprised by, mitigated by --no-ui/config.
Tests pass (240, +8) and ruff is clean on touched files; the GPU e2e hasn't been
re-run against this on yardrat yet — the open-webui service is new on the serving
path and only validated via the fake-docker render/converge tests.

Takeaway: to make one service in a converged compose project immune to churn
while another recreates on change, encode "what may change" exclusively in the
spec (labels/env) of the service you *want* recreated, and keep the stable
service's spec a pure function of inputs that don't change — recreation is then
a derived property, not a special case.

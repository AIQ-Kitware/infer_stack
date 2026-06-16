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

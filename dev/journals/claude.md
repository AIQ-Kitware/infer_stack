## 2026-06-27 09:04:20 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code.

User intent: in the TUI they cleaned up leases, then "Clean up" on deployments
did nothing — a pile of IDLE deployments with no container (running/gpus "-",
0 leases) just sat there. Diagnosed first: "Clean up" = `ledger.prune()`, which
only forgets STOPPED deployments + RELEASED/EXPIRED leases; IDLE keep-warm
deployments are deliberately retained (still in the desired set — an apply would
re-spawn them), so prune correctly skips them. The retire path is Release →
Evict (IDLE→STOPPED) → Clean up. The "-" containers are drift (torn down
out-of-band, e.g. the port-churn fix earlier today, but the ledger still lists
them as warm-pool entries). User then asked for two ergonomics fixes.

What I changed (TUI):
- **Evict all idle** button in the deployments pane → `controller.evict(None)`
  (evicts every IDLE deployment in one go). One click takes the whole warm pool
  to STOPPED so a following Clean up forgets it.
- **Multi-select** in the leases + deployments tables. Space toggles the cursor
  row; **ctrl/cmd-click** toggles a discontiguous row; **shift-click** extends a
  contiguous range from the anchor (`_click_select` is pure over (tid,row,mods)
  so it unit-tests without synthesizing mouse events; `on_click` reads the
  already-moved `cursor_row` and delegates, mirroring the existing ctrl+click-to-
  open path). Selection rendered in a leading marker column; Release/Evict act on
  every checked row via `_target_ids()` (checked set wins; else the cursor row,
  so old single-row behaviour is unchanged). Held by id in `_lease_sel`/`_dep_sel`
  so it survives a poll refresh, pruned to live rows on refill, cleared after an
  action.

Course-correction worth recording: the user first asked for "normal" ctrl/shift-
click; on finding Textual 8.x has **no native row multi-select** (only text
selection — `anchor`/`get_selection` are for copy/paste; the row API is the
unreleased PR #6585, see discussion #3606) they said drop it rather than carry a
hand-rolled surface, then "if you almost have it just finish it." It was nearly
done, so I finished it but boxed it as an explicitly **removable shim** (one
marker column + two sel-sets + `_click_select`/`_repaint_marks`, all behind the
`_table_sel` accessor) with a code/CHANGELOG note to delete it wholesale if/when
Textual ships the native API. Takeaway: when you must shim a missing framework
feature, isolate it behind one seam and name the eventual native replacement so
the removal is mechanical.

Design choices: kept prune's conservative semantics (never auto-forget a
deployment the system still wants) — the fix is discoverability (a bulk evict +
clearer button/desc text), not changing what "Clean up" means. I deliberately
did NOT (yet) auto-heal the "IDLE-but-container-gone" drift to STOPPED; that's a
separate reconcile decision I flagged to the user. Also left the two confusingly
duplicated "Clean up" buttons as-is for now (raised earlier as a rescope option).

Gotcha worth remembering (now a lesson): named the selection helper
`_action_targets` and it blew up with `'set' object is not callable` — Textual's
`App.__init__` binds `self._action_targets` to a set (action-namespace
resolution), an *instance* attribute that shadows a class method. Renamed to
`_target_ids`. `hasattr(App, '_action_targets')` is False because it's per
instance, which is exactly what made it shadow.

Confident: full suite 281 passed incl. new TUI tests (evict-all flips IDLE→
STOPPED; multi-select releases both checked leases + clears selection; space
toggles off again; `_click_select` covers ctrl-toggle / shift-range / plain-
clear). Low risk — additive UI, single-row paths unchanged. Testing note: I
first added a pixel-offset `pilot.click(..., control=True)` wiring test; it
passed alone but flaked under the full suite (offset geometry), so I dropped it —
the pure `_click_select` unit test plus the robust `space` keypress test cover
logic and wiring without the brittleness. `space` relies on the DataTable not
consuming it; the headless space test confirms the app-level binding receives it.

## 2026-06-27 08:38:01 -0400

Model: claude-opus-4-8[1m] (Opus 4.8, 1M context), Claude Code.

User intent: debug a slurm-e2e failure from the dynamic-routing run the user
rsynced back (`tests/infer_stack_pipeline_e2e/_runs/20260627T080504-slurm`, still
running). One `smol_135_01` node failed; the rest passed.

Investigation (the interesting part — the converge log alone lies): the node's
acquire succeeded and readiness passed, but the probe then got 20× `litellm
.InternalServerError: Connection error … Model Group=smol-135`. The converge log
just said "up -d 5/6 services" each time — nothing looked wrong. The GPU-memory
**timeline** (`diag/timeline.log`) was the tell: all four upstreams loaded
(08:10:51), then *every* GPU dropped to ~2 MiB at 08:12:38 and reloaded over the
next ~2 min. So an unrelated `gpt2` *release* at 08:12:26 recreated the still-
leased smol/gpt2-tp2 containers. Root cause: `render_compose` assigned each
upstream's published host port by enumeration index (`BASE + vllm_i`) over the
live set, so removing one deployment renumbered the survivors' ports → changed
their service specs → `docker compose up -d` recreated them. LiteLLM's route then
pointed at a restarting container → connection errors for the reload window.

Fix: behind the gateway an upstream is internal (reached by compose-network DNS
at :8000), so publish **no** host port — removing the only set-dependent field,
so a survivor's spec is byte-identical as the set changes (the same no-blip
property the static gateway config already has). No-gateway path still publishes
(the readiness probe hits the upstream directly). Also made the dynamic-routing
reconcile treat `/model/delete` "not found in db" as success (a shared gateway
lets a concurrent converge delete the route first — saw that warning in the log).

Considered alternatives: (a) stable per-deployment port via id-hash — rejected,
collisions reintroduce set-dependence and it keeps useless host-port pressure
with many `--dedicated` upstreams; (b) persist a port on the deployment record —
heavier (ledger/migration) for a debug-only convenience. "Internal services
don't need host ports" is the simplest invariant and gives *guaranteed* zero
churn (no field to renumber), so it won.

Confident about: the root cause (placement is sticky — verified — so device_ids
were stable, leaving the port as the only changing field; arithmetic matches the
two renders 18000/18001/18002 → 18000/18001). Tests: full suite 277 passed incl.
a new no-churn regression (`test_dynamic_upstreams_have_no_host_ports_and_survive
_set_change`) and the delete-race test. Risks/uncertainties: (1) the *first*
converge after deploying this on the real stack will recreate upstreams once (the
ports disappear from the on-disk file) — a one-time blip, then stable. (2)
Separately observed a worker-propagation race — the successful jobs' probes saw a
few `404 model does not exist` right after `/model/new` before settling; the
probe's retries absorb it, but if LiteLLM runs multiple workers the admin-API add
propagates asynchronously, so readiness can pass on a warm worker while a client
hits a cold one. Not what failed this run; noted as a watch-item. I implement
here; the user re-runs on aiq-gpu.

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
(grounded in the two real consumers: aiq-eval-runner's evaluation cards
and eval_audit's HELM/kwdagger path). User explicitly asked to start on a
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
with `infer-stack run`, and switch an aiq-eval-runner card to acquire/release.

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

Follow-up same session: user asked whether `secret` is needed at all given
`env`. It wasn't — `env` already covered get/path/list; `secret` only added
`set`. Folded `set` into `env` as a `KEY=VALUE` positional (`env KEY` reads,
`env KEY=VALUE` writes, merging non-destructively), removed the `secret` modal
and its three classes entirely. One verb, unambiguous by the presence of `=`.
Reinforces the rename's premise: secrets in a readable `.env` don't warrant a
separate "secret" surface. Updated demo/CHANGELOG/followups and tests (240 pass,
ruff clean).

## 2026-06-18 17:31:31 -0400

Model: claude-opus-4-8 (Claude Code, Opus 4.8). Config: default tools, working
the infer-stack submodule from the aiq-eval-runner superrepo.

User intent (one bundled prompt, five threads): (1) keep leasing-demo current;
(2) rich-format `infer-stack leases` (and "other CLIs"); (3) `release --all` to
make teardown concise; (4) restore the lost "show me the compose diff before you
change it, --yes to skip" approval; (5) add loguru narration (aivm-style) so the
behind-the-scenes process is legible.

Design decisions worth recording:
- **loguru off the hot path.** loguru imports at ~50ms — we'd spent real effort
  getting `infer-stack --help` to ~0.2s, so adding it at module scope would undo
  that. Solution: a private `_log.py` that is imported only at the leasing
  *runtime* chokepoint (`_open_controller`) and inside converge/reconcile, never
  by the help/catalog/config surface. It also `logger.disable('infer_stack')` on
  import, so library and test use is silent until the CLI calls
  `configure_logging()`. Narration goes to **stderr** so stdout (JSON,
  `$(infer-stack env KEY)`) stays clean. Verified loguru count == 0 in the
  `--help` importtime trace.
- **Diff-approval lives in the backend's converge, but the policy lives in the
  CLI.** ComposeBackend gained `assume_yes`; converge renders, computes the
  changed files vs disk, and (when not assume_yes) calls the existing
  `diff_prompt.confirm_writes`. The *decision* of when to prompt is the CLI's:
  only the additive verbs (acquire/serve) prompt, and only on a TTY without
  `--yes` (`_resolve_assume_yes(interactive=...)`). release/leases/run never
  prompt — for a teardown, the action is the approval. This keeps `release --all`
  and the `run` pipeline-node usable non-interactively without hanging.
- **Decline must not leak ledger state.** acquire = ledger.acquire (creates the
  lease) → reconcile (converge). If the operator declines, converge raises a
  neutral `ConvergeAborted` (defined in backend.py, not compose.py, to respect
  the layer boundary), and controller.acquire rolls the lease back with a
  *bookkeeping-only* `ledger.release` (not controller.release, which would
  reconcile again). Tested with a tiny DeclineBackend rather than the real
  renderer so the test doesn't depend on a fully-populated spec.
- **release --all** enumerates active leases from the ledger and releases each;
  the reconciles collapse naturally (each release converges the shrinking union;
  the last one downs the project).

Risks/uncertainties: the diff prompt fires inside converge, which is reached
through the controller — fine for the single-shot CLI, but any future caller
that converges in a loop on a TTY would prompt repeatedly (mitigated: only
acquire/serve set interactive). loguru is a new hard dependency (added to
pyproject + uv.lock; clean 2-package diff). The compose diff/approval path is
unit-tested with a fake docker + monkeypatched confirm; the real interactive
prompt on GPU hardware hasn't been exercised yet. 250 tests pass, ruff clean.

Takeaway: when adding cross-cutting feedback (logging) or a new dependency to a
CLI you've tuned for startup latency, gate it behind the runtime entry points
rather than module scope — "imported only when actually doing work" preserves
both the fast `--help`/completion path and library silence, and costs only a few
lazy imports.

## 2026-06-18 20:46:06 -0400

Model: claude-opus-4-8 (Claude Code, Opus 4.8). Config: default tools, working
the infer-stack submodule from the aiq-eval-runner superrepo, branch
`dev/leasing-controller`.

User intent (one exploratory session, several threads, mostly driven by running
the demo on yardrat): (1) add two Qwen models to the demo (cross-family); (2)
"why didn't `serve qwen05-1` prompt / why is it `live` when nothing's running?";
(3) "show what's *actually* running + which GPUs"; (4) "separate render from
execution — set up on-disk state and see what would run before pulling the
trigger"; (5) expand `--help`; then a verb-shape brainstorm landing on
declares-refcount / reconciles-idempotent; (6) "do we ALWAYS preview before
modifying the compose file?" → gate every converge. Committed in two batches:
`fe90d2e` (1–5) and a follow-up (6).

Design decisions worth recording:
- **Fail fast on unplaceable requests.** The yardrat hang was: acquire marks the
  group LIVE in the ledger *before* reconcile, placement silently dropped it (no
  free GPU → only a WARNING), and `serve` then blocked on readiness for a
  container that never rendered. Fix: the backend records `last_unplaced`;
  controller.acquire rolls the lease back and raises `PlacementError` if a
  *just-requested* group landed unplaced. This also dissolved the "phantom live"
  display — the bad lease never persists. Key realization: `live` is *desired*
  (ledger), never *observed*; the two were conflated in `leases`.
- **leases shows desired vs actual.** Added `running` (backend.observe) and
  `gpus` (`plan()`, read-only placement; `→N` = slated) columns. Backed by a new
  `ComposeBackend.plan()` that the converge path now reuses.
- **Legible names.** vLLM service/container is `vllm-<served-model>-<group-id>`
  (was `vllm-<group-id>`). The group-id suffix is load-bearing: two desired
  groups can share a served name (an endpoint re-pointed at a new model), and it
  keeps `docker ps` ↔ `leases` correlatable. Ollama keeps its id-name (one
  daemon, many models). One-time container recreate on upgrade (service key
  changed) — which is exactly what bit the user later (see below).
- **Render/apply as the separating seam.** `converge(desired, apply=False)`
  renders the on-disk project without `docker compose up`. Surfaced as a flag on
  the declare verbs (`serve|acquire --no-apply`, stage) plus lease-free `render`
  / `apply` reconcile verbs. The crux we reasoned through with the user: `apply`
  writes *no* infer-stack tracking state — all of it (ledger intent + sidecar
  placement) is written at declare/stage time — so "drop the tool and `docker
  compose up` yourself" is a true equivalent. Cemented that by baking a top-level
  `name: infer-stack` into the rendered compose, so a bare `docker compose -f
  <file> up` lands in the same project (was the one real gap — default project
  name would've been the dir name).
- **Verb shape: declares refcount, reconciles are idempotent.** The user found
  that `serve --no-apply` then `serve` makes *two* leases (demand 2). We
  explicitly rejected idempotent serve/acquire (serve *is* `acquire --ttl inf`;
  a refcount that sometimes doesn't count is the confusing thing). The real fix
  is structural: a lease-free `apply` means "apply a staged lease" no longer goes
  through a declare verb, so the refcount only climbs when you genuinely declare.
  Idempotency belongs on render/apply (pure functions of desired state), not on
  the refcount verbs.
- **Gate every converge (supersedes the 2026-06-18 17:31 decision).** That entry
  said teardown verbs don't prompt ("the action is the approval"). The user's
  GPU run disproved it: `release --all` silently rewrote the compose file *and
  recreated a keep-warm model* (the one-time rename migration happened to land on
  a release). So now `release`/`evict`/`apply` also build `interactive=True` and
  gate the diff (`--yes` skips). To keep it to one prompt, `release`/`evict` now
  batch all ledger mutations then converge once (was: per-lease reconcile). New
  asymmetry: declining an *acquire* rolls back; declining a *release/evict*
  records the ledger change but leaves docker untouched (`apply` later) — which
  is the honest plan/apply semantics, not a bug.

Risks/uncertainties: none of this has run against real docker/GPUs since the
changes (only fake-docker units) — the new gating prompt, the one-time container
recreate, render/apply, and the `name:` field are unexercised on hardware. The
batched release changes decline semantics (ledger released even if docker
declined); defensible but a behavior change. The container rename will recreate
every running vLLM container once on upgrade (acceptable, noted in CHANGELOG).
267→ tests pass, ruff clean on changed files (pre-existing I001s elsewhere).

Takeaways: (1) separate *desired* (ledger) from *observed* (backend) in both the
data model and the UI — conflating them produced both the "phantom live" bug and
the silent-placement hang; surfacing `running` vs `state` made the system
honest. (2) When a tool drives another tool, make the generated artifact
self-sufficient (`name:` in the compose file) so "bypass the tool" is a
first-class equivalent, not a divergent sibling — that property is also what let
us reason that `apply` hides no state. (3) Idempotency is a property of
reconcile operations, not of intent-declaring ones; put it where it's natural
rather than special-casing a refcount.

## 2026-06-19 00:09:12 -0400

Model: claude-opus-4-8 (Claude Code, "fast" Opus). Continuation of the leasing
TUI work.

User intent (compressed): make the TUI usable by a newcomer and stop it feeling
janky. Concretely: (1) an orange / white / very-dark-gray theme; (2) "feel
faster or more responsive"; (3) a real bug — `docker compose` progress was
overwriting the top-left of the screen for a moment when serving; (4) turn the
logs pane into a docker *tab* (logs **or** ps); (5) require `config init` before
launch, and if there are no models offer a button that runs the new `suggest`
tool to populate the catalog; (6) wizards to add models/endpoints; (7) more
explanatory text so new users can intuit what's going on; (8) drag-to-resize
panes.

What I did (all in `infer_stack/tui.py`, `tests/test_tui.py`, plus a small
`TuiCLI` change in `cli/commands_leasing.py`):

- Theme: registered a custom `textual.theme.Theme` (`infer-orange`) — dark-gray
  canvas, white text, one warm orange accent that only shows on the focused
  pane's border. Replaces the borrowed `nord`.
- Bleed bug: root cause was `_default_docker_run` = `check_output` (captures
  stdout, lets **stderr** through), and `docker compose up -d`/`down` write all
  progress to stderr → straight onto the full-screen terminal. Fix lives at the
  TUI layer (didn't touch the CLI's runner, where stderr-to-terminal is
  desirable): on mount I wrap `backend.run` with a `capture_output=True` runner
  that routes the noisy verbs' stderr into the logs pane and swallows `ps`
  polling. Felt cleaner than changing the shared default.
- Responsiveness: the periodic refresh used to call `backend.observe()`
  (`docker compose ps`) **on the UI thread** every 3s — a real freeze. Split
  into `_collect()` (thread-safe, no widgets) + `_render()` (UI thread);
  `on_mount` does one synchronous `_refresh_now()` for an instant first paint
  (and so the headless tests still see data after a single `pause()`), while the
  interval + `r` + post-mutation refresh go through a `@work(thread=True)`
  worker. The bleed fix doubles as responsiveness (no terminal corruption).
- Docker tab: logs pane is now a `TabbedContent` — "Logs" (the existing Select +
  RichLog, IDs preserved) and "Status · ps" (a DataTable fed by a best-effort
  compose-ps parse in the collect worker).
- Onboarding: `TuiCLI` now hard-requires `settings.yaml` (i.e. `config init`)
  and *tolerates a missing/empty catalog* via a new `_load_catalog_for_tui`
  (returns an empty Catalog + the path it would write). Empty state shows a
  help line steering to Suggest/add. Suggest (`g`/button) calls the other
  agent's now-committed `leasing.suggest.suggest_catalog` + `commands_catalog`
  load/save helpers (imported, never edited), merges, reloads in place.
- Wizards: `_AddModelScreen` / `_AddEndpointScreen` ModalScreens with Inputs;
  results written through `commands_catalog._load_raw/_save_raw` (which validate)
  and the catalog reloaded. `m` / `n` or sidebar buttons.
- Drag-resize: a tiny `_Divider(Static)` that captures the mouse and reports
  axis-delta to a callback (`_drag_sidebar` / `_drag_logs`), alongside the
  existing `[` `]` / `-` `+` keys. Vertical bar between sidebar/main, horizontal
  bar between tables/docker.

State of mind / risks: as always, none of this has been *eyeballed on a real
terminal* — headless pilot proves it composes, tabs, resizes, serves, streams,
and writes the catalog, but the orange theme's actual feel, the drag ergonomics,
and (critically) whether the stderr-capture fully kills the corner-bleed are
only verifiable by running `infer-stack tui` on a GPU host (yardrat). The
suggest path in particular only runs under real hardware detection. Confident
about: the threading split (clear win, observe was blocking), the
backend.run-wrapping approach (localized, reversible), and that I stayed off the
other agent's files (their suggest CLI is committed at 4ef5147; I only import).
8 TUI tests + full prior suite (303) green; ruff clean on changed files.

Takeaways: (1) when a full-screen UI drives a subprocess, capture *both* streams
at the UI boundary — stderr is the usual screen-corrupter, and the fix belongs
at the UI layer, not in the shared runner where stderr is wanted. (2) Keep a
synchronous first paint even after moving refresh to a worker: it makes the app
feel instant *and* keeps headless tests deterministic without `wait_for_complete`
gymnastics. (3) For a tool newcomers land in, downgrade "missing config" from an
error to an empty-state-with-a-button — the absent catalog is the start of the
funnel, not a failure.

## 2026-06-19 00:40:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Same TUI session, two follow-up
fixes from the user: (1) drag-to-resize still didn't work; (2) the global bottom
menu listed pane-specific actions (serve/release/evict) that don't apply
everywhere — wanted them scoped to their panes, keeping only truly-global items
global.

Drag root cause (found by a headless harness, not guessing): the `_Divider`
`Static` had no explicit cross-axis size, so `Region` was `width=1,height=1` — a
1×1 dot. The drag *logic* was fine (posting a captured `MouseMove(delta_x=5)`
moved the width 38→43 in the harness), there was simply nothing to grab. Fix:
`#vsplit { height: 1fr }` and `#hsplit { width: 1fr }` so each bar spans its
cross axis (verified: vsplit now 1×36, hsplit 79×1). Lesson worth keeping: when
a Textual mouse interaction "doesn't fire," check the widget's `region` first —
an auto-sized helper widget can collapse to an un-hittable cell even though its
handlers are correct.

Scoped actions: moved s/d/e/a/g/m/n bindings to `show=False` (keys still work)
and added per-pane Buttons — Serve under `#endpoints`, Release/Release-all under
`#leases`, Evict under `#groups`, plus the existing catalog buttons. Footer now
shows only Refresh / Next-pane / Quit. `on_button_pressed` became an id→action
map. Intro line updated to point at the per-pane buttons + drag bars.

Tests: added grab-area, real-drag (mouse_down + posted MouseMove + mouse_up →
width changes), and scoped-button-serve cases. 11 TUI tests green, ruff clean.
Confident in the drag fix (measured). Still unverified on a real terminal: the
actual feel of grabbing a 1-cell bar in tmux (motion reporting under tmux can be
finicky) — if dragging is still flaky on yardrat, the next move is widening the
hit target or enabling a drag affordance glyph.

## 2026-06-19 01:25:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Third TUI pass. User asks:
(1) system info à la nvidia-smi/btop; (2) a small API-test tab (send a prompt to
a model, or test all models); (3) ctrl+click a served endpoint to open it in
Open WebUI; (4) click-to-collapse panes so hidden data isn't polled; and a
follow-up: (5) the `ps` view should carry uptime/status, created time, and
container id like `docker ps`.

Approach: turned the bottom pane into a `Collapsible` wrapping a `TabbedContent`
with four tabs — Logs, Status·ps, System, API. The collapse is the user's
"click to collapse," and it does double duty as the polling gate: `_collect()`
now only fetches the *active* tab's expensive data (`docker compose ps` for the
ps tab, `nvidia-smi` for System) and nothing when collapsed. `_active_tab` is
tracked via `TabbedContent.TabActivated` and `_console_collapsed` via
`Collapsible.Toggled`; both are plain attrs read by the worker thread, set on the
UI thread — cheap and race-free enough for a monitor. Switching tabs fires a
refresh so the newly-shown tab fills immediately.

System tab: `nvidia-smi --query-gpu=… --format=csv,noheader,nounits` parsed into
a table; returns `None` when nvidia-smi is absent so the UI shows a clear "not
found" row instead of looking broken (this dev box has no GPU — only yardrat
will populate it). Host line from `/proc/loadavg` + `/proc/meminfo` + cpu_count
(no psutil dep, which isn't installed). API tab: a model Select (from catalog
endpoints), a prompt Input, Send / Test-all buttons, and a RichLog; calls go to
`http://localhost:{litellm_port}/v1/chat/completions` with the backend master
key. Made the HTTP client injectable (`http=`) so a fake drives it headless —
otherwise this tab would be untestable without a live gateway. ps columns now:
service · status(uptime) · created · container-id · ports, pulled from the
compose-ps JSON (`Status`, `CreatedAt`/`RunningFor`, `ID[:12]`).

Open-in-browser: `action_open` (key `o`) builds `…:{ui_port}/?models={endpoint}`
and `webbrowser.open`s it; ctrl+click routes through `on_click` →
`screen.get_widget_at` → walk to `#endpoints`. Over SSH/tmux `webbrowser.open`
will no-op, so the status line always prints the URL to copy.

Risks/uncertainties: the big one remains *unverified on real hardware* — the
Open WebUI `?models=` deep-link param depends on the running Open WebUI version
(it may just land on the home page; acceptable fallback). nvidia-smi parsing
assumes the standard CSV columns. Collapsible + TabbedContent + my fixed
`#docker` height composes cleanly in pilot, but I haven't watched it animate on
a terminal. ctrl+click hit-testing via `get_widget_at` is the part I'm least
sure survives tmux mouse quirks; the `o` key is the reliable fallback and shares
all the logic. 16 TUI tests (added: ps-parse, system-no-smi, collapse-gating,
open-url, injected-http API send), full suite 316 green, ruff clean.

Takeaways: (1) make the collapse state and active-tab the *same* signal that
gates polling — "don't poll what you can't see" falls out for free instead of
being a second mechanism. (2) Any pane that hits the network or a subprocess
needs an injection seam (proc_factory, http) or it's simply not testable
headless; build it in from the first line, not after.

## 2026-06-19 02:40:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Fourth TUI pass + a leasing-core
addition. User asks: (1) API tab should list only up-and-ready models;
(2) a way to clean up old released/evicted entries; (3) drop the global intro,
make descriptions pane-local; (4) rename console→docker, the ps tab→"containers";
(5) promote System and API to their own panes; and a late one: (6) default theme
to stock "textual dark".

Core add: `SqliteStore.prune(lease_states, group_states)` + `Ledger.prune()`
(deletes RELEASED/EXPIRED leases and STOPPED groups, claims first to dodge the
groups FK). This is the first ledger *deletion* path — sweep/evict only ever
transitioned state, leaving a growing tail. Kept it explicit (prune, not
auto-gc) so history stays inspectable until you choose to forget it; the TUI's
Clean-up button is the first caller.

"Ready" models: I define ready = endpoints served by a group that is *observed
running* (g.id in the observe() set), computed in `_render` and pushed to the API
selector. Cheap (reuses the placement view the groups table already needs) and
matches the "running" column the user sees. Not a true vLLM readiness probe —
documented as such; a container can be Up but still loading. Good enough for "let
me poke a model that's actually there," and avoids per-refresh HTTP.

Layout: System and API became their own `Collapsible` panes, collapsed by
default — which doubles as the polling gate (`_collapsed` dict + `_active_tab`):
nvidia-smi only runs when System is expanded, `docker ps` only when docker's
Containers tab is visible. leases/groups got wrapped in bordered Verticals so
each can hold a description + table + its own buttons.

Sharp edge found + fixed: in Textual 8.2.7 the Select no-selection sentinel is
`Select.NULL`, and `Select.BLANK` is literally `False` — so my `value is
Select.BLANK` checks were always False (a latent bug, dormant only because the
other Selects always carry a real value). Swapped all four to `Select.NULL`.
Lesson candidate: don't assume a framework sentinel's name across versions —
verify with a one-liner. 18 TUI tests (added: docker/containers tabs, ready-only
API list, collapse-gates-system, cleanup-prunes-released), full suite 318 green,
ruff clean.

Uncertainties: same standing one — unverified on real hardware (System/API/ps
need a live GPU + gateway). The ready-model definition may surprise someone who
expects a just-served-but-still-loading model to be immediately queryable; the
status line + "only ready models listed" desc set that expectation.

## 2026-06-19 12:30:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Large multi-request turn; Jon
explicitly OK'd staging. Locked four design decisions via a question prompt
(saved to agent memory): group→deployment, session_id→lease_id (HARD), TUI
top-level tabs (Dashboard/Settings/Catalog) + command palette, keep both
monitors; and the standing rule that the CLI must be a superset of TUI features.

Shipped two commits:
1. (08353d3) Made the lease↔deployment many-to-one legible — the thing Jon most
   wanted to "click." Leases pane gained a `deployment` column (the literal group
   id, matching the deployments pane's id column, so the join is eyeball-able);
   deployments pane shows `leases` (count) + `held by` (owners) instead of the
   opaque `demand`. Cursor movement narrates the link in the status bar. Buttons
   compacted to 1 row.
2. (0ab64db) session_id→lease_id hard rename: env key INFER_STACK_LEASE_ID,
   --lease flag, JSON lease_id, read_lease_id(), id prefix lease-. Verified no
   parent-repo consumers before breaking it. 319 tests green.

Deliberately deferred the group→deployment rename: 558 code sites + a SQLite
`groups` table is too big to do safely alongside features. It's its own next
pass. Open sub-decision I'll default on unless redirected: keep the SQLite table
named `groups` (purely internal, avoids a ledger-DB migration) while renaming all
Python symbols + user-facing strings to deployment — i.e. rename the concept
everywhere a human sees it, leave one hidden DB identifier alone.

State of mind: good progress, low regression risk so far (rename was contained,
full suite green twice). The remaining work is mostly additive TUI (tabs,
settings, catalog mgmt, edit/remove, docker control, model cache info, all-panes
collapsible + drag bars everywhere) plus the big mechanical group rename. Biggest
judgment call ahead is the reorg: top-level tabs change the whole compose() tree,
so I want to land the group rename first (so new UI is built on final vocabulary)
unless Jon prefers to see the reorg sooner.

## 2026-06-19 12:50:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Jon's directive: clean break,
no back-compat (pre-release, no users), delete-and-recreate the ledger DB rather
than migrate, and rip out superseded legacy later — but PRESERVE the ollama +
kubeai conceptual paths and any reusable logic. Names must agree everywhere; flag
any place a presentation name should legitimately differ from a backend name (I
found none in the leasing surface).

Did the group→deployment deep rename. Approach that kept it safe despite 558
sites: first proved the concept is isolated — `CacheGroupSpec`/`cache_groups`
(KV-cache, experimental/), `_is_group` (CLI command-group, commands_meta), and
regex `.group()` all live OUTSIDE the 12 leasing-concept files — so I scoped an
ordered sed (DeploymentGroup→Deployment, GroupState→DeploymentState,
group_id(s)→deployment_id(s), groups→deployments, group→deployment,
Group→Deployment) to exactly those 12 src + 6 test files, plus the SQLite table
`groups`→`deployments`. Two traps caught by the test suite: (1) the all-caps
`GROUP_LABEL` constant (my rules were case-specific) and (2) the rename clobbered
Textual's `@work(..., group='logs')` worker-group kwarg → `deployment='logs'`,
which `work()` rejects — reverted those 10 decorators. Also `ruff --fix infer_stack/
tests/` over-reached and autofixed ~13 unrelated files' pre-existing lint; I
`git checkout`'d everything outside scope to keep the diff focused. test_cli_leasing
+ docs + CHANGELOG updated to the new vocabulary (protecting the LiteLLM "Model
Group=" error quote and the "grouped into a holding pen" verb). 319 tests green.

State of mind: confident — the rename is mechanical and twice-verified green, the
DB table rename is fine since we're nuking state. Deliberately did NOT keep the
SQLite table name (Jon: no migration, delete the DB). Remaining: legacy-code
removal (profiles/old hookups, keeping ollama+kubeai concepts), then the TUI
top-level-tabs reorg + catalog mgmt/edit-remove + advanced endpoint params +
docker control + model cache info + all-panes-collapsible/drag. Those build on
the now-final vocabulary, so no rework.

## 2026-06-19 13:30:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Legacy removal, Phase 1, with
Jon's exception: keep + improve StatusCLI into a holistic `infer-stack status`.

Mapped the import graph first (the only safe way to delete in a web of shared
modules). Found the pre-leasing profile world is gated behind an explicit
`legacy` command group whose own docstring said "removed wholesale" — so that
group + its exclusive modules were the target. Deleted (~5k LOC, no back-compat
per Jon — pre-release, delete-and-recreate state): the `legacy` group;
cli/commands_profile, cli/commands_smoke; renderer, benchmark, verification,
contracts; the active-profile runtime verbs (Up/Down/Purge/Deploy/Env/Ollama*)
+ the _ensure_rendered/RenderCLI hook inside commands_runtime; and 4 legacy test
files (serving_profiles, ollama_stack_graph, reverse_proxy_ldap, cli_setup) +
2 legacy tests in test_cli_meta. Suite 319→222 green, and 110s→15s — the deleted
legacy tests were the slow ones.

Rewrote commands_runtime.py clean: the `stack` day-2 wrappers now target ONLY
the leasing compose project (dropped the legacy config.yaml fallback +
config_for_runtime/_compose_base_cmd/_kubeai_stub), and a new holistic StatusCLI
reports backend, data/config dirs, catalog (+counts), settings, ledger, compose
locations, a leasing summary (active leases / live deployments, read-only — no
sweep so status never mutates), and "dig deeper" pointers. Kept the ollama +
kubeai *concepts* (kubeai_ops.py, backends/kubeai_renderer.py, catalog
engine:ollama) per Jon, even though currently unwired.

Deliberately deferred (Phase 2): resolver/validator/old top-level catalog.py +
the legacy plan helpers in cli/context.py (build_plan/resolve/validate/
load_config/config_for_runtime) and cli/compose.py. These are load-bearing for
context.py (which kept code imports for _apply_path_overrides/effective_inventory),
so they need per-function surgery, not deletion — its own careful pass. Also
Phase 2: ConfigPathsCLI still lists legacy artifacts (config.yaml/models.yaml/
kubeai_generated_dir); trim once config.py's legacy paths go.

Risk/uncertainty: the new StatusCLI is unverified against a real populated stack
(only the leases-present unit test). The deferred cluster is where the remaining
breakage risk lives. Confident the deletion is clean (graph-driven, suite green).

## 2026-06-19 14:30:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Legacy sweep Phase 2.

2a: deleted resolver.py (1053) + validator.py (314) — imported only by
cli/context — after carving context down to the two helpers the leasing surface
actually uses (_apply_path_overrides, effective_inventory). Also deleted the
cli/compose + cli/probes shims (only the deleted commands used them) and trimmed
cli/__init__'s context/compose/probes re-exports. Rewrote ConfigPathsCLI to
report current locations only (config_root, settings.yaml, catalog.yaml,
data_root, leasing ledger/compose) instead of the old config.yaml/models.yaml/
generated_dir/state groups; updated its tests. ~2700 LOC.

2b: deleted backends/compose_renderer.py (superseded by leasing/compose) and
slimmed backends/__init__ to just the kubeai renderer (kept as the kubeai
concept per Jon). The whole backends package is now unimported except itself —
kubeai_renderer is deliberately retained for when that backend lands.

Deliberately deferred (internal-only, no CLI surface): the old top-level
catalog.py and the now-dead helper chain inside config.py (builtin_*_catalog,
the catalog loader, initial_config/deep_merge/load_yaml/etc. — all 0 external
callers). config.py stays for normalized_output/normalized_state/
default_state_paths + DEFAULT_PORTS/PINNED_IMAGES/KUBEAI_GENERATED_SUBDIR; the
dead chain is interwoven with initial_config and would need careful
function-boundary excision. Judgment call: at ~7800 LOC removed across Phase 1+2
with the suite green throughout, the risk of nicking a kept config function for
a maintainer-internal tidy wasn't worth doing at the tail of a long session.
The "two catalogs" (top-level catalog.py vs leasing/catalog.py) confusion is the
one remaining legacy-naming smell; flagged for a focused follow-up.

Confident: every deletion was import-graph-driven and verified green. Next up is
the TUI top-level-tabs reorg + catalog management (edit/remove, advanced
endpoint params) + docker control + model cache, all on the clean base.

## 2026-06-19 16:00:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). TUI feature sweep on the clean,
post-legacy, final-vocabulary base.

- Advanced endpoint wizard: exposed tensor-parallel / max-model-len / gpu-mem /
  extra-args / reclaim (mirrors `catalog endpoint add`), plus Edit (blocked while
  served) and Remove for endpoints + models with a confirm dialog. The
  validating writer refuses removing a model still referenced by an endpoint.
- Top-level tabs: wrapped the multipane in a TabbedContent → Dashboard +
  Settings. Settings reads/writes settings.yaml (backend, data dir, Open WebUI,
  reverse proxy, skip-display-GPUs). Key trick: extracted the dashboard body into
  `_compose_dashboard()` and `yield from` it, so the big block kept its exact
  indentation (no risky re-indent) — verified by the 26 existing tests staying
  green after the wrap.
- docker Control tab (compose up/down + compose-file path) driving the captured
  runner so output lands in Logs. Models table gained quant + a cheap 'cached'
  flag (existence check vs state.hf_cache/hub — no du).
- Drag bars: added the two splitters Jon explicitly called out — endpoints|models
  (#csplit) and leases|deployments (#lsplit) — reusing the fixed-height-neighbor
  + 1fr-neighbor pattern (a divider that drags one pane's height while the other
  flexes). All four splitters now grabbable.

Remaining: (1) "every pane collapsible" — docker/system/api collapse;
leases/deployments/endpoints/models are now drag-resizable instead (collapse +
fixed-height-divider fight each other; drag covers the space-management need).
Full per-pane collapse is a further polish. (2) The internal config.py dead-chain
gut + old top-level catalog.py deletion (no CLI surface) — still deferred.

229 tests green throughout; each TUI feature landed as its own committed,
test-backed increment. Confidence high on the mechanics; the on-terminal feel
(tab switching, drag ergonomics under tmux) remains the standing
unverified-without-a-real-terminal caveat.

## 2026-06-19 18:30:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Live-testing feedback round +
the two deferred remainders. Jon is eyeballing each change.

TUI fixes from live testing:
- Endpoint wizard was an unlabeled blank form in edit mode (prefilled inputs hide
  placeholders). Rebuilt it: a Label per field, and engine-adaptive groups — vLLM
  shows tensor-parallel / data-parallel / max-model-len / gpu-mem / max-seqs /
  prefix-caching / extra-args; Ollama shows host + free-form KEY=VALUE runtime.
  Mapped to the canonical leasing runtime keys (verified against
  leasing/compose._service_dict), so no CLI change needed — data-parallel etc.
  were always reachable via runtime keys / --runtime / --extra-args; the wizard
  just surfaces them.
- API tester promoted to a top-level tab (frees the monitor column). Catalog
  Add/Edit/Remove localized to the endpoints + models panels; the bottom button
  stack removed; Suggest moved to Settings.
- Fixed the endpoints|models drag inversion (fixed-pane-below ⇒ subtract delta).
- Vertical splitter now drags the full width range (clamped to terminal width),
  not just the middle (min-width was 26; lowered to 10, clamp uses self.size).

Per-pane collapse: made leases + deployments Collapsibles (docker/system already
were). Discovered + documented a real Textual constraint via a scratch:
setting an explicit height defeats Collapsible's collapse (height wins), and
clearing to None didn't restore shrink — so collapse and a fixed-height drag bar
can't share a pane. Chose collapse for the four monitor panes (dropped the
leases|deployments lsplit) and kept drag for the catalog (endpoints|models
csplit), since you act on the catalog constantly and want it sized, not hidden.

config.py gut + old catalog.py deletion (the last "two catalogs" smell): traced
the call graph — the builtin_*_catalog → file/overlay/merge → normalized_catalogs
chain + initial_config were all dead externally (only normalized_output/
normalized_state/default_state_paths + the port/image constants are used, by
leasing/compose + kubeai_renderer). Excised the chain (kept deep_merge, used by
normalized_cluster), dropped the `from .catalog import normalize_*`, deleted
infer_stack/catalog.py, and removed the now-orphaned profile templates
(default-*.yaml + the legacy .j2 renderer templates; kept suggestion-pool.yaml).

232 tests green throughout; ruff clean. Standing caveat unchanged: the
on-terminal feel is Jon's to confirm (he is, live).

## 2026-06-19 21:30:00 -0400

Model: claude-opus-4-8 (Claude Code, fast Opus). Live-test fixes + the suggest
sizing bug (Jon green-lit touching suggest.py — the other agent has been idle the
whole session).

- API tab: added gateway + Open WebUI URLs (ctrl+click / Open button), a List
  models check (GET /v1/models), a live curl preview + Copy curl, and clipboard
  via Textual OSC 52 (Copy curl; `y` copies the status line). Caveat surfaced to
  Jon: OSC 52 needs terminal/tmux `set-clipboard on`.
- Crash fix: the URL line used Textual `[link=http://...]` markup, which rejects
  the ':' in the URL and threw MarkupError in get_content_height. Switched to
  plain text (ctrl+click + button still open it).
- suggest.py sizing bug (the real one): vLLM OOM'd at startup for smollm2-1.7b
  because suggest overrode the pool's hand-tuned gpu_memory_utilization (0.4,
  sized for the 8192 KV cache) with a bare footprint ratio (min_vram*1.3/host =
  0.22 on a 24 GiB GPU) — below what the KV cache needs. Root insight: the pool
  default is a *floor* (it encodes the KV requirement); the footprint estimate
  should only *raise* it on smaller GPUs. Fix = max(footprint, pool_default),
  still clamped [0.2, 0.92]. smollm2-1.7b now → 0.4. Added a regression test.
  Note for Jon: existing catalogs already have the bad 0.22 baked in — re-run
  `catalog suggest --apply --force`, or Edit the endpoint, to pick up 0.4.

236 tests green. The suggest fix only changes future suggestions; the standing
on-terminal caveat (clipboard/drag under tmux) is Jon's to confirm live.

## 2026-06-26 22:13:37 -0400

Model: claude-opus-4-8[1m] (Claude Code, fast Opus). Jon asked to implement two
things we'd designed in conversation for the leasing controller: (A) split the
single mutate-lock into a fast render lock + a separate apply lock, and (B)
coalesce the slow `docker compose up` across concurrent acquires. Motivation:
under the shared-stack model every `acquire` serialized end-to-end behind the
previous one's bring-up (the global flock wrapped render AND `docker compose up`),
and concurrent slurm jobs each ran their own redundant `up`.

Design (now in code + CHANGELOG): a monotonic generation pair in the ledger
`meta` table — `desired_gen` bumped inside the same `BEGIN IMMEDIATE` as each
desired-set change (acquire always; release/evict/sweep when they actually idle
something), `applied_gen` published after a successful apply. The render lock
(the old `_global_lock`, kept reentrant for acquire->render nesting) guards
ledger-write + placement + compose-file render only. The new `_apply_lock` (a
second flock, `.apply.lock`) serializes `docker compose up` and doubles as the
coalescing wait-queue: `_ensure_applied(g_target)` loops `while applied_gen <
g_target`, takes the apply lock, re-checks (someone may have covered it), else
snapshots `g = desired_gen` BEFORE the up and publishes that floor after. Floor-
before-up is the key correctness point — a render landing mid-apply leaves
applied_gen < its g_target so it re-applies, never silently dropped. `apply()`
deliberately does NOT take the converge lock (that would re-serialize renders
against the slow up); compose-file writes are now atomic (`os.replace`) so a
concurrent render can't be half-read by an apply. `infer-stack apply` routes to
a new `apply_now()` (force) so the manual re-sync still heals drift when the
generation hasn't moved.

Reflections / what I'm confident vs not: confident the mechanism is correct —
the coalescing proof is a *deterministic* test (`_ensure_applied` with the gen
pre-advanced: exactly one apply; the apply-lock serializes so late winners
re-read applied_gen and break). I burned real time on test flakiness that turned
out to be a *genuine product bug*, not test noise: concurrent first-open of a
fresh ledger raced on `PRAGMA journal_mode=WAL`, which sqlite returns as
"database is locked" instead of honoring busy_timeout — exactly the batch-of-
slurm-jobs pattern this work targets. Fixed by retrying the WAL switch + schema
DDL in `SqliteStore.__init__`. Also replaced an inherently racy "render-not-
blocked-by-apply" threaded test with a deterministic structural one (assert
`_flock_depth == 0` and the render-lock file is `LOCK_NB`-grabbable during
apply). Tradeoff accepted: every acquire now forces at least one (coalesced,
idempotent) apply even when joining an already-live model — chosen for drift-
healing + simplicity over a "skip apply if already up" optimization (noted as a
possible future tweak). What might break: backends that have `converge` but no
`apply` and *honor* `apply=False` by not realizing are now invalid (render leaves
nothing up, no apply step to finish) — updated `BudgetBackend` in the queue test
to model the render/apply contract (record placed in render, realize in apply).

264 tests green (added `tests/test_leasing_coalesced_apply.py`). Not yet
exercised on real docker/GPU — the compose `apply()` path is covered only by
fakes here; the slurm e2e tier on aiq-gpu is where it meets real `docker compose
up` under contention.

## 2026-06-26 23:18:49 -0400

Model: claude-opus-4-8[1m] (Claude Code, fast Opus). Jon's directive: "we are
going to do the dynamic routing work. And get this done elegantly and properly."
The trigger was the slurm e2e tier exposing a real product bug: `--dedicated`
same-model deployments place on distinct GPUs in the ledger but the Compose
backend names every vLLM service `vllm-<served>`, so they collapse onto ONE
container/GPU — and worse, the test stays green (everyone routes to the one
container that answers). The static-superset gateway can't fix this because its
no-blip property *depends on* that name being derivable from the served model
alone (so a catalog route can address it without knowing the live deployment).

Before touching code I verified the load-bearing fact against the pinned
`litellm v1.82.3` source: `/model/new` (and `/model/delete`/`/model/update`)
HTTP-500 unless a DB is connected AND `STORE_MODEL_IN_DB=true`. So "the admin-API
direction" is NOT DB-less — it *is* the postgres-litellm revival (the old
follow-up #2 the maintainer had parked as "tried it, hit issues"). I surfaced
this fork explicitly rather than silently reintroduce Postgres; Jon chose
admin-API+Postgres and confirmed "having the litellm db makes a lot of sense."
Correcting that misconception in the doc was important — an earlier draft implied
plain `/model/new` worked in-memory, which would have led a future agent astray.

Design (in code + CHANGELOG + docs/litellm-gateway-routing.md): an opt-in
`dynamic_routing` mode that honors the project's render/apply split. RENDER:
each deployment gets a unique `vllm-<served>-<id>` service (so dedicated spreads
across GPUs), a `postgres-litellm` service is added, the litellm service gets
`DATABASE_URL`+`STORE_MODEL_IN_DB` env + `depends_on` DB-healthy, the rendered
config is a *static* base (empty `model_list`) so its hash never moves (no blip),
and the desired route set is written to `litellm_routes.json` (one entry per live
(deployment,endpoint), each with a deterministic `model_info.id`). APPLY: after
`docker compose up`, `_reconcile_routes()` lists `/v1/model/info`, diffs by id,
and POSTs `/model/new`/`/model/delete`. The deterministic id is the crux — it
turns reconcile into a pure set-diff that is idempotent (so the controller's
coalesced apply is correct for routes too), drift-healing (lost routes reappear,
stale ones deleted), and co-existent (only `isr-`-prefixed ids are ever deleted,
so a UI-added model is safe).

Why this shape over alternatives: I considered (a) a DB-less dynamic config file
— rejected, reintroduces the per-change gateway recreate/blip static-superset was
built to avoid; (b) direct per-deployment URLs in the env-file for dedicated —
rejected, splits addressing into two modes and abandons the gateway's value.
Admin-API+Postgres is the only path that gives per-deployment routing AND no
blip, and the env-file descriptor is unchanged (clients keep one gateway
base_url; LiteLLM load-balances the shared alias across the dedicated upstreams).

Confident: the rendering + reconcile logic is unit-covered (10 new tests in
`tests/test_leasing_dynamic_routing.py` incl. the dedicated-collision fix, the
no-blip invariance, deterministic routes, and add/delete/idempotent/co-existence
reconcile against a RecordingGateway fake); full suite 274 green; static mode is
byte-for-byte unchanged (default off). Not yet exercised against a real LiteLLM
+ Postgres on a GPU host — the reconcile is fakes-only here. Risks/uncertainties
to watch on aiq-gpu: (1) gateway-startup timing — `_reconcile_routes` retries the
initial `/v1/model/info` (30×2s) while litellm boots behind the DB healthcheck;
if a first-ever image pull makes that window too short, routes get added on the
next converge instead (acquire's wait_ready could then time out on the very first
cold acquire). (2) LiteLLM load-balancing a public alias across N dedicated
upstreams is the intended behavior but unverified end-to-end. (3) every verb must
agree on the mode or service names diverge — hence the persisted `dynamic_routing`
setting is the primary switch (the flag is a per-call override), mirroring how
`catalog` is resolved for release/gc.

## 2026-06-30 10:55:23 -0400

Model: claude-opus-4-8[1m] (Claude Code, fast Opus). Directive: "A not-ready
acquire should not keep its lease ACTIVE if the acquire times out." This is the
first bullet of the leaked-active-lease cluster the GPU e2e suite surfaced (see
`dev/leasing-followups.md`), where the harness had to wipe the ledger between
tiers to mask it. The user explicitly rejected preserving the old behavior behind
a flag — "buggy behavior is not something to preserve with parameters" — so the
fix is unconditional, no `--keep-on-timeout` escape hatch.

The bug was an asymmetry: `Controller.acquire` already rolls the just-created
lease back on two "couldn't deliver" paths — operator-declined converge
(`ConvergeAborted`) and unplaceable request (`PlacementError`) — but the readiness
*timeout* was the one failure that returned `wait.ready=False` with the lease
still ACTIVE. The deployment then stayed LIVE, and because reconcile trusts the
ledger as desired-state and never prunes a LIVE group whose lease leaked, one
timed-out acquire could re-spawn a GPU-hogging container on every later converge.
`run` happened to dodge this (it releases in a `finally`); the `acquire`/`serve`
CLI leaked it; the TUI uses `wait=False` so it never reached the path.

Fix: make readiness-timeout the *third* rollback path in `acquire` — on
`not wait_result.ready`, call `self.release(lease.id)` (which reconciles, tearing
down per reclaim policy) and set a new `AcquireOutcome.released_on_timeout=True`.
Design fork I weighed: raise a symmetric `ReadinessTimeout` exception (matches
`PlacementError`) vs. a return-marker. Chose the marker — `acquire` already
returns `wait.ready=False` to *describe* the timeout, so "and I released it" is a
natural extension, and it avoids forcing a try/except on every caller and a
test-wide churn. Release is idempotent (`ledger.release` no-ops on an
already-RELEASED lease), so `run`'s redundant inline release + `finally` stay
safe; I dropped `run`'s now-redundant inline release and left only the SystemExit
message. `_emit_acquire` reports the teardown (and skips `--env-file`, since a
released lease has no standing endpoint to point a sourceable file at) and keeps
exit 2.

The escape hatch for the legitimate "hold a lease while a slow model loads"
workflow is `--no-wait` — it acquires detached (`wait_result is None`), never
reaches the timeout path, so the lease is correctly kept; you `wait` for it
separately. That's a distinct non-buggy behavior, not a flag preserving the leak.

Confident: covered by `test_wait_ready_timeout_releases_lease` /
`_keepwarm_idles_deployment` (controller: stop-policy tears down + GPU freed;
keep-warm idles but the lease is still RELEASED) and `test_acquire_timeout_*`
(CLI: rc==2, lease released, env-file not written), via a monkeypatched
`MemoryBackend(ready=False)` + `--timeout 0` (deadline passes on the first poll,
so the test is instant — no real sleep). Ran the leasing/CLI/TUI suites: 66 + 87
green. Uncertainty: the *other two* bullets of the cluster stay open and I said so
in the tracker — re-acquire spawning a new group instead of reviving an idle one
(that's the deterministic-group-id work) and observe()-driven ledger/reality
reconciliation. This change closes the leak at its source (the leaked ACTIVE
lease); it does not add the observe()-driven self-heal. Reusable takeaway: when a
function has N "couldn't deliver" exits that all roll back, an (N+1)th failure
mode that *doesn't* roll back is almost certainly the bug — grep the siblings for
the rollback idiom and match it, rather than inventing new cleanup.

## 2026-06-30 13:49:48 -0400

**Intent.** A slurm/tmux teardown (`infer-stack release --env-file …`) was
emitting `UserWarning: could not open a cross-process lock file (tried
/data/service/infer-stack/leasing/.leasing.lock and /tmp/infer-stack-…lock);
proceeding with in-process locking only`. The user's concurrency is N separate
CLI *processes* across tmux/slurm sessions, so the in-process `RLock` fallback
serializes nothing. They asked for (a) a precheck that ensures the lock dir and
file are group-writable, (b) group-write-by-default on new files we create, and
(c) — chosen via AskUserQuestion — that a mutating verb **raise / refuse to
mutate** rather than degrade when no real cross-process lock is obtainable.

**Model/harness.** Claude Opus 4.8 (1M), Claude Code CLI.

**Root cause (confirmed by reading + reproducing the digest).** `_open_flock`
tries the beside-ledger lock then a `/tmp` fallback named
`sha1(str(lock_path))[:16]` — I reproduced `485b0da6e53b99d5` exactly, so the
fallback is keyed on the *ledger path, not the user*. Path 1 fails because the
service-owned ledger dir is writable at the *file* level (the sqlite db) but not
for a *new* file (the lock). Path 2 fails because the first user created the
shared `/tmp` file `0644`; a different uid (the teardown) can't reopen it for
append and the sticky bit blocks replacing it. Both `OSError` → old code warned
and proceeded on `threading.RLock`.

**Change (`leasing/controller.py`).**
- `_ensure_group_writable(handle, path)`: best-effort, ownership-gated. `fchmod`
  the lock fd to `g+rw`; `chmod` the dir to `g+rwx | setgid` (so siblings'
  files inherit the group). Only touches paths we own — a non-owner *can't*
  chmod a service dir and shouldn't try; that case is the raise path. Idempotent
  (only chmods when bits differ), never raises. Called on every successful open
  so it also self-heals our own pre-existing `0600` lock files.
- `_open_flock` no longer warns/None-degrades silently; it returns the handle
  (after self-heal) or `None`.
- `_global_lock` (and `_apply_lock`) **raise `LeaseLockError`** on a `None`
  handle, gated by `_lock_path is not None` so the in-memory ledger (tests)
  still yields lock-free. Placed *before* `_flock_depth += 1`, so the raise
  leaks no depth/handle state and reentrancy is preserved. All mutating verbs
  (reconcile/apply_now/acquire/release/evict/gc) funnel through `_global_lock`,
  so one raise covers them.
- `_diagnose_lock_failure` / `_lock_path_reason`: re-inspect both candidates and
  build the actionable message (owner uid, mode, group-writable?, why each open
  failed) + the `chgrp … && chmod g+rwX … && chmod g+s …` (+ `umask 002`) fix.
- Exported `LeaseLockError`; `cli/__init__.main` catches it → `SystemExit(msg)`
  so teardown shows a clean diagnosis, not a traceback.

**Design forks.** (1) `_apply_lock` raising is unreachable in normal flow (the
render lock is taken first in the same dir, so it raises first) but I made it
raise too for invariant-consistency — "no real lock ⇒ no mutation" everywhere.
(2) Considered gating self-heal strictly on "newly created" per the user's
"new file" wording, but applying it whenever-we-own-it is idempotent and also
repairs stale locks, strictly better. (3) Setgid on the dir is slightly beyond
"group write," but it's the standard mechanism that makes a *shared* lock dir
actually work across uids, so I included it (documented in docstring/CHANGELOG).
(4) User explicitly declined the env-flag "configurable" option — hard raise.

**Tests.** Added `test_global_lock_raises_when_no_lock_obtainable` (monkeypatch
`tempfile.gettempdir` at a regular file so *both* candidates fail; asserts the
raise, message content, and no leaked `_flock_depth`/`_flock_handle`) and
`test_created_lock_file_and_dir_are_group_writable`. Full submodule suite green
(247 passed, 1 skipped). The pre-existing `test_lock_falls_back_when_ledger_dir_
unwritable` still passes — the writable `/tmp` fallback must NOT raise.

**Open / for the operator.** This makes the *code* cooperative, but the live fix
on the cluster is still ops: make `/data/service/infer-stack/leasing` group-
shared (`chgrp` + `g+rwxs`) or run all sessions as one user. The `/tmp` fallback
can never be truly cross-user (different `TMPDIR`/systemd `PrivateTmp` silently
split it) — it only ever serializes same-user/same-`/tmp` processes, and now we
fail loudly instead of pretending otherwise. Reusable takeaway: a "degrade to a
weaker lock + warn" fallback is a footgun when the weaker lock doesn't model the
real concurrency — prefer self-heal-then-refuse over silent degrade.

## 2026-07-17 12:04:27 -0400

**Model/config:** claude-fable-5[1m] (Fable), Claude Code VSCode extension;
planning session driven from the eval_audit superproject.

**User intent:** Jon is preparing to benchmark the small Qwen3.5 models
(0.8B/2B/4B) on yardrat's second GPU (Quadro RTX 5000, 16 GiB) alongside the
9B on the RTX 8000 (48 GiB), and rejected operator GPU-pinning as the
mechanism: "I really don't want to think about having to tell it which model
can go where. I'd love if infer-stack understood that a request for a
particular endpoint could only be satisfied by certain GPUs." Asked for the
full plan written into docs/planning with the objective forefront so it
survives future reconsideration.

**What landed:** `docs/planning/vram-aware-placement.md` (new dir). Plan in
one line: endpoints declare `placement: {min_vram_gib: N}` in the catalog
(validated, backward-compatible when absent); `plan_placement()` filters GPUs
by eligibility (`memory_gib >= min_vram_gib`), orders deployments
most-constrained-first, and picks best-fit (smallest eligible GPU) instead of
index-order first-fit; internals become capacity subtraction with an
exclusive-per-GPU flag so future co-hosting is a policy flip.

**Why this shape (state of mind):** The investigation found the codebase has
both halves already — inventory records per-GPU `memory_gib`
(hardware.py:67), and suggest.py owns the exact fit vocabulary
(`min_vram_gib_per_replica`, `fits_on`, `_host_gpus` picking
smallest-that-fits with a written rationale) — they've just never been
introduced to each other at placement time. So this is a wiring change along
the codebase's own grain, not new machinery. `plan_placement` being a pure
function with 20 tests makes it the ideal seam; Phase 0 is
tests-first (heterogeneous `simulate_inventory('48,16')` + semantics-pinning
cases) before touching the planner.

**Alternatives rejected (recorded in the doc):** per-runbook
`INFER_STACK_ALLOWED_GPUS` pinning (hand-encoded schedule, rots, per-host);
undocumented `runtime.gpu_indices` (manual + unvalidated); SLURM typed GRES
(relocates the mapping into job specs plus a cluster-config project;
composes later anyway via the existing `$SLURM_JOB_GPUS` design).

**Deliberate scope amendment:** placement.py's docstring says bin-packing is
"explicitly out of scope"; the plan narrows that to *multi-node* bin-packing
and preemption. Single-host eligibility + greedy best-fit is now in scope —
and greedy is adequate forever at ≤8 GPUs; the doc says so to preempt future
ILP temptation.

**Uncertainties:** exact min_vram_gib numbers must be measured on yardrat,
not derived from weight bytes (activation overhead at our max_model_len);
KubeAI backend policy for the new field (leaning warn-and-ignore); whether
pinned-tier-wins-over-new-declaration is the right stability tradeoff
(chose stability + warning).

**Next:** Phase 0 (tests) when Jon green-lights implementation. The
eval_audit-side adoption (declare requirements in the Qwen3.5 catalogs,
drop the pinning plan) is tracked in the doc as Phase 3.

**Same-session update (12:20):** Jon resolved the open questions. (1) The
numbers come from *self-measurement*: models measure themselves at first
healthy serve and the result updates their catalog via an explicit promote —
now design §3 + Phase 3. Key trap designed around: `nvidia-smi memory.used`
measures the `gpu_memory_utilization` knob (vLLM preallocates KV to fill the
fraction — a 0.8B "uses" ~41 GiB of a 48 GiB card), so measurement parses
vLLM's own profiling breakdown instead; measured values live in a data_dir
overlay, never silently rewrite catalog.yaml. (2) suggest will NOT precompute
placement numbers — Jon's tried, they're never exactly right; precomputation
survives only as the weight-bytes *floor* (sound underestimate, prevents the
9B→16GiB class of misplacement from day one with zero measurement).
(3) KubeAI/k3s: warn-and-ignore confirmed. (4) VRAM-aware reservations:
explicitly deferred, no decision. Phases renumbered (measurement = 3,
adoption = 4, future = 5).

**Same-session update (12:30):** Jon refined the measurement design:
measurement is OPTIONAL, not a pipeline stage. Normal path = operator's
declared best guess; if a serve OOMs on a GPU the declaration called
eligible, that's a *diagnosed misdeclaration* with a guided error naming the
exact `placement measure` command; the weight-bytes floor clamps unsound
guesses automatically (max(declared-or-measured, floor)). Design §3
rewritten around that flow; Phase 3/Resolution 1 aligned; floor-clamp test
case added (9B declared at 8 GiB still never lands on the 16-GiB card).
Design takeaway: the guess doesn't need to be right — it needs to fail
diagnosably and cheaply, with the fix one copy-paste away.

**Same-session update (13:05) — Phases 0–2 implemented.** Jon green-lit
implementation. Tests-first as planned: 14 failing semantics-pinning tests
authored against the agreed behavior, then made green without touching the
20 legacy tests. What landed:

- `hardware.simulate_inventory` accepts heterogeneous specs — comma-separated
  `M` or `NxM` entries (`'48,16'` is yardrat) — legacy `'4x96'` unchanged.
- `catalog.py`: endpoint-level `placement: {min_vram_gib: N}` block —
  strict-keyed (a `min_vram_gb` typo is a CatalogError, not a silent
  no-constraint), positive-number-validated, vllm-only, threaded into
  `spec['placement']` only when non-empty so existing catalogs resolve
  byte-identical specs. Deliberately NOT structural: it says where a
  deployment may land, not what process it is (test pins compat-key
  equality across differing declarations).
- `placement.py`: `min_vram_per_gpu()` = max(declared, floor_vram_gib) —
  the floor field is planner-supported NOW so Phase 3 only has to produce
  it; eligibility filter (per-shard, memory_gib >= req); fit tier ordered
  by (n_eligible-in-pool, created_at, id) = most-constrained-first with
  legacy tie-break; declared deployments take smallest-eligible-free
  (best-fit, final selection re-sorted ascending so CUDA_VISIBLE_DEVICES
  stays index-ordered); UNDECLARED deployments keep exact legacy
  index-order first-fit — an undeclared 9B still lands on GPU 0, not
  "best-fit" onto the 16er it can't run (this scoping was the one deviation
  worth writing back into the plan doc's §2). Permanent-vs-transient error
  split: "pool can never satisfy" (with copy-pasteable inventory) vs "only
  N free" (the queue case). `GpuPlan.warnings` carries honored-but-suspect
  pins/explicit indices that contradict declarations; converge logs them.

Validation: 357 passed / 2 env-skips full suite; placement 35/35 (20 legacy
+ 15 new); catalog 29/29 (20 + 9 new); doctests pass. Reusable takeaway:
when adding a scheduler constraint to a live system, scope the new
*preference* (best-fit) to participants that opted in, and give
non-participants bit-identical legacy behavior — backward compat isn't just
"old tests pass", it's "old configs cannot be made worse by the upgrade".

**Same-session update (14:00) — Phase 3 implemented.** New
`infer_stack/leasing/vram.py`: tolerant multi-version parser for vLLM's
memory-profiling log lines (uses the LAST serve in a restarted container's
log), `derive_min_vram_gib` (non-KV profile × margin + KV budget — the KV
budget is OUR serving choice, not a model fact), `weight_floor_gib` (stat-only
over the local HF hub cache; largest single snapshot, never a cross-revision
sum), `Measurements` overlay (fail-open JSON at <state_dir>/measurements.json),
and the OOM classifier (explicit allocator/vLLM signatures only — a generic
'error' match would send operators measuring after unrelated crashes).
Compose backend enriches specs at plan time (declared > measured > floor,
best-effort, never persisted); `deployment_logs()` feeds both the guided OOM
hint in acquire's not-ready paths and the new `infer-stack measure <ep>
[--record]` command (measures a live deployment in place, or acquires once
and releases). KubeAI: warn-and-ignore per Resolution 3.

Design catch during implementation: the auto-enriched floor must gate
ELIGIBILITY only, not flip an undeclared deployment into best-fit selection —
otherwise the mere act of downloading weights would move existing catalogs'
deployments (e.g. an undeclared small model hopping to the 16-GiB card).
Split `declared_min_vram` from `min_vram_per_gpu`; two tests pin it. That's
the second backward-compat subtlety of this feature and both have the same
shape: NEW information sources may only ever *restrict* where things can go
or improve outcomes for opted-in deployments — never reroute a non-consenting
deployment that was fine.

Suite: 376 passed / 3 skipped (13 new vram tests, 2 new placement tests,
4 new compose enrichment tests, MeasureCLI registered). Not exercised on real
GPUs yet — the measure command's acquire-once path and the docker-logs parse
get their first real run in Phase 4 on yardrat.

## 2026-07-31 11:09:13 -0400

Model: claude-opus-5[1m] (Claude Code, default effort).

**Intent.** "Look into https://github.com/llm-d/llm-d-inference-sim to see if
we can run it as a mock in the system, then verify that two downstream
evaluation cards work against a mock vLLM server run via infer-stack."
This follows a correction I earned: I had built our own mock optimised for
producing plausible *scores*, and defended that choice against the two
external simulators the user had suggested. The user's answer was blunt and
right — "If you had asked I would have said API fidelity matters more. That's
how we know if this thing is going to break in production or not. I don't care
what the scores are in the mock run."

**What I found.** llm-d-inference-sim is a far better vLLM stand-in than what
I wrote: real SSE streaming with configurable TTFT and inter-token delay, real
prompt/completion token accounting, `/v1/models`, `/v1/embeddings`,
`/v1/responses`, `/metrics` (vLLM-compatible subset), 404 on an unknown model,
injectable `rate_limit`/`server_error`/`context_length` failures, a
`startup-duration` window that answers 503 on `/health/ready`, and LoRA
lifecycle simulation. Published distroless image on ghcr. It cannot answer a
question correctly — responses are random sentences — which is precisely the
thing I over-weighted last time.

**The design decision.** It speaks vLLM's API but not vLLM's CLI: no
positional model, no `--host`, no `--tensor-parallel-size` /
`--gpu-memory-utilization`, and it exits on an unknown flag. Three options:
(a) a shim image translating vLLM argv, (b) a third `engine` alongside
vllm/ollama, (c) a `runtime.simulator` block on the existing vllm engine.

I took (c). (a) hides the divergence inside a wrapper and buys nothing — the
vLLM argv renderer is only really testable against vLLM. (b) is ~20 dispatch
sites across compose/placement/catalog/CLI for something that is, from every
client's point of view, an ordinary vLLM upstream: port 8000, `/v1`,
`/health`, same gateway route, same lease bookkeeping. (c) touches exactly the
three places where a simulator genuinely differs — argv, GPU count, container
healthcheck — and leaves the rest byte-identical.

Two non-obvious bits. `required_gpu_count` had `max(1, ...)`, which would make
every simulator endpoint unplaceable on the GPU-less host a simulator exists
to serve; it returns 0 now. And the healthcheck shells out to `curl`, which a
distroless image does not contain, so it must be disabled rather than
inherited — safe only because `probe_ready` gates on a real generation over
HTTP, which is strictly stronger. I did *not* add `simulator` to
`VLLM_STRUCTURAL_FIELDS`: adding a key rehashes every existing compat key and
would orphan live leases, and `image` (already structural) separates simulator
from real deployments anyway.

**On keeping our own mock.** I kept it, under `catalog-mock-oracle.yaml`, with
its role stated plainly in both catalog headers and `docs/mock-endpoints.md`:
llm-d-sim answers "does the client break", ours answers "does the math
compute". Random text cannot distinguish a working fan-in from one that
silently drops a model, because garbage is the expected input either way. That
is a real and separate job — but it is the *second* job, not the first, and
the docs now say so rather than leaving a future reader to infer my earlier
priority ordering.

**Validation.** Full leasing path exercised for real (not rendered): acquire →
placed on "(cpu)" → compose up → LiteLLM route → readiness generation probe →
command sees `OPENAI_BASE_URL`/`OPENAI_API_KEY` → non-streaming and streaming
chat with correct usage counts → release → teardown. 7 new tests in
`tests/test_simulator_runtime.py`; suite green.

**Uncertainties / risks.** The simulator has no bearer-auth option, so the
auth path is still only covered by our own mock. `--model` must not be a real
HF repo id or it tries to reach a tokenizer render sidecar and dies at
startup; I default it to the served name and pin that with a test, but a user
who sets `simulator.model` to a repo id gets a crash-loop with a message that
does not obviously say why. Docker access on this host needed `usermod -aG
docker`; that is environmental, not a repo change.

**Takeaway.** When a stand-in for a real component diverges, the question is
*which layer* it diverges at. This one is client-identical and
deployment-different, so the right seam was the deployment renderer, not a new
engine. Had it diverged at the API it would have needed to be a new engine —
or not used at all.

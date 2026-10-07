## 2026-09-16 15:45:35 -0400

Summary of user intent: make `infer-stack logs -f` and the TUI readable when LiteLLM emits very large repeated connection-failure tracebacks, while suppressing only traceback shapes that infer-stack has explicitly registered as known spam. Unknown or changed tracebacks must remain visible, and the filtering path should stay cheap enough for live logs.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I implemented the filter as a small streaming module rather than embedding policy in either UI. The central contract is fail-open: `compact_litellm_tracebacks` buffers one bounded Python traceback segment, and it removes only header/frame/source scaffolding when the terminal exception plus required frame fingerprints match one of the explicit `LITELLM_TRACEBACK_PATTERNS`. It preserves terminal exception summaries and any interleaved ordinary service records exactly as emitted. The first registry covers the seven stack segments in the captured LiteLLM -> OpenAI-compatible backend connection-refusal cascade. A missing required fingerprint, a new exception type, a non-LiteLLM service, an incomplete traceback, or an oversized/malformed candidate stays raw.

The CLI applies this only to an interactive `logs -f` display; non-follow logs, redirected/piped follows, and `--raw` keep the existing raw subprocess behavior. The TUI wraps its existing injected stdout iterator with the same compactor, so there is one registration surface for both views. I kept successful request/access-log suppression out of scope because deciding that a successful operation is uninteresting is a different policy question from removing representation-level traceback scaffolding.

Validation: the exact 257-line user-provided failure is checked in `tests/data/litellm_connection_refused.log`; direct execution reduces it to 31 lines with no traceback headers or frame lines while retaining all exception summaries, the interleaved HTTP 500, retry/fallback data, and model-health GETs. Standalone assertions for the pure filter pass, including timestamp handling, ANSI Compose prefixes, interleaved service order, EOF fail-open, and fingerprint near misses for every registered segment. Full pytest could not run in this container because the extracted checkout has no dependencies and the locked `uv` environment cannot reach PyPI; the attempted sync failed downloading `xdoctest`. Python byte-compilation passes for all changed Python files.

## 2026-09-16 16:13:54 -0400

Summary of user intent: follow up on the conservative LiteLLM traceback compactor because a different repeated `NotFoundError` cascade still floods `infer-stack logs -f`, and restore the Compose service colors that disappeared once the compacted CLI path began piping stdout through Python. Keep the allowlist/fail-open model rather than broad traceback suppression.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I kept both changes narrow. The new log capture is a repeated model-routing failure: an OpenAI-compatible backend returns 404 (`The model ... does not exist`), then LiteLLM propagates it through the OpenAI adapter and proxy/router layers. I registered exactly those three complete traceback segment shapes as `model_not_found.openai_sdk`, `model_not_found.openai_adapter`, and `model_not_found.proxy`, each with terminal-exception matching plus stack-frame fingerprints. Near misses still replay byte-for-byte. The supplied capture starts in the middle of an older traceback; that clipped prefix intentionally remains raw because there is not enough evidence to classify it. Complete traceback segments in the capture compact, reducing 415 captured lines to 91 while retaining the 404 summaries, model ID, fallback status, backend access/error records, and interleaved service output. I did not add duplicate-error rate limiting or successful health/access-log suppression because those are separate policies from traceback compaction.

The color regression was upstream of the parser: `compact_litellm_tracebacks` already strips ANSI only on a private parsing copy and emits original raw lines, but `_run_compacted_follow` connects Compose stdout to a pipe. The CLI now adds Compose's global `--ansi always` option only for the interactive compacted follow path, before the `logs` subcommand. `--no-color` remains authoritative, while raw/non-follow/piped paths retain their prior behavior. The TUI is unchanged here; it already explicitly launches Compose with `--no-color` for its RichLog surface.

Validation: standalone filter assertions pass for both captured failure families (connection-refusal remains 257 -> 31; model-not-found is 415 -> 91), each new registered segment compacts to its terminal summary, missing fingerprints fail open, and ANSI-decorated source lines retain their original escape bytes after compaction. `py_compile` passes for the changed Python files. Full pytest remains unavailable in this extracted environment because repository test collection imports `ubelt`, which is not installed here; a CLI command-shape test was added for `--ansi always` / `--no-color` and should run in the normal project environment.

## 2026-09-17 00:42:50 -0400

Summary of user intent: produce a direct filesystem overlay against infer-stack commit f9a46be that fixes the correctness issues from the latest keep-warm/admission review, without reopening the architecture: dynamic LiteLLM route verification must detect same-id semantic drift; quiescent config/network previews must match the post-TTL-sweep render they actually apply; service-affecting managed environment values must not trigger delayed restarts of actively leased models; and candidate reverse-proxy profile bytes must not leak to disk before publication commits.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I kept the repair local to the existing serialized-publication design. Dynamic route reconciliation now treats a managed route id as identity only: it compares the observable routing semantics infer-stack owns (public model name, upstream model, and API base), deletes/re-adds same-id routes whose payload drifted, and verifies the semantic table after reconciliation. Config-publish and network-migrate previews use a virtual-expiry admission view, so a past-TTL lease that has not yet been materialized by sweep produces the same files as the post-commit render. Managed runtime credentials (HF_TOKEN, LiteLLM master key, and LiteLLM DB password) are now configuration-time writes: the env command holds the publication lock and refuses to change them while any protecting lease is active; client-only env values remain writable. Finally, ComposeBackend.use_profile no longer writes the managed reverse-proxy snapshot. The snapshot content is part of the in-memory planned files and is written by converge only after approval, so a declined/crashed candidate preview cannot mutate a currently published bind mount.

Regression coverage added: same route id with a wrong model is replaced; config publish and network migration succeed when the only ACTIVE database row is already TTL-expired and no container remains; runtime-secret env mutation is rejected under active demand while a client-only URL remains writable; and a crash before profile commit leaves the old reverse-proxy snapshot bytes untouched. Python byte-compilation passes for every changed Python/test file. With a tiny temporary ubelt path stub, the four new non-CLI regressions pass directly (4/4), and the broader dynamic-routing/profile/network selection reaches 61 passing tests; its five remaining failures are only imports of CLI tests because this container lacks scriptconfig. The CLI env regression could not execute here: the offline uv cache has neither ubelt nor scriptconfig (and full uv sync also lacks charset-normalizer). The normal project environment should run the focused CLI test and full suite before merge.

## 2026-09-17 00:52:47 -0400

Summary of user intent: repair the post-overlay full-suite failures and warning flood: all TUI tests fail under the installed Textual because the blank Select sentinel is now `Select.BLANK` rather than `Select.NULL`, and Python 3.13 reports large numbers of leaked sqlite connections as `ResourceWarning: unclosed database`. Fix both at their ownership/API boundaries rather than filtering warnings.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I made the Textual blank-selection API version-independent with one `SELECT_BLANK` sentinel and an identity helper, then routed every TUI read/write of the empty Select value through it. This preserves compatibility with older Textual releases that exposed `NULL` while matching the installed release that exposes `BLANK`. The one test that asserted the old Textual implementation detail now asserts infer-stack's compatibility sentinel instead.

The sqlite warnings were genuine ownership leaks, not warnings from PyYAML, argparse, or scriptconfig: those modules merely happened to allocate while an older sqlite connection was garbage-collected. `SqliteStore` owns its connection, so it now provides context-manager cleanup and a defensive finalizer that closes the connection before sqlite's Python 3.13 finalizer can report it as leaked. Existing explicit `close()` remains idempotent and preferred at clear lifecycle boundaries. The two tests that intentionally create raw watcher connections now close those watchers explicitly. A regression forces GC under a `ResourceWarning` recorder and fails if a `SqliteStore` can still produce an unclosed-database warning.

Risk/tradeoff: finalization is a safety net, not a substitute for transactional scope; no transaction behavior was changed. A store that is still strongly referenced remains open exactly as before, and a store can still be explicitly closed. The TUI compatibility constant intentionally uses identity because Textual's blank value is a sentinel, not an ordinary selectable value.

## 2026-09-17 17:52:31 -0400

Summary of user intent: make it easy from the Textual TUI to assign a vLLM endpoint/model to a particular physical GPU while preserving automatic VRAM-aware placement as the default, and add the local `Qwen/Qwen3-8B` model as a valid `Suggest from my GPUs` candidate using Hugging Face metadata rather than a guessed footprint.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I made exact affinity an endpoint-level operator override at `placement.gpu_indices`, rather than changing the scheduler default or reusing the global `allowed_gpus` pool restriction. Catalog validation requires a non-empty, unique list of non-negative physical indices and requires its length to equal tp×pp×dp. The supported placement spelling is consumed before the legacy `runtime.gpu_indices` fallback. Exact affinity is included in the vLLM structural compatibility payload only when a pin exists, so unpinned catalogs retain their existing compatibility hashes while a repin cannot revive a deployment resident on the old GPU. The CLI exposes the same override through `catalog endpoint add --gpu ...`.

The TUI now shows a `gpu` column (`auto` or exact indices) and a `GPU pin` action. The common single-GPU case is a dropdown populated from `nvidia-smi` with GPU name and VRAM; multi-GPU runtimes accept exactly tp×pp×dp comma-separated indices. Active endpoints must be released before repinning. When an idle old deployment exists for the endpoint, changing the pin evicts it after the validated catalog write. Endpoint editing also preserves an existing `placement` block; previously the edit wizard rebuilt only the fields it exposed, which would have silently erased the new pin (and could already erase `min_vram_gib`). Ollama remains host-level GPU affinity.

For suggestions I added `qwen3-8b -> Qwen/Qwen3-8B`. The official Hugging Face metadata/model card reports 8.2B parameters and 16,381,470,720 bytes of safetensor weights; the model card describes 32,768 native context with 131,072 available using YaRN. The pool therefore records a 16-GiB footprint class, a conservative 24-GiB per-replica serving floor to leave vLLM/KV-cache headroom, and a 32,768 default/native context rather than enabling YaRN implicitly. Tests cover catalog validation/compatibility, planner consumption plus the legacy spelling, CLI merge/count validation, suggestion inclusion/exclusion and HF id, endpoint edit preservation, TUI pin write/column behavior, and active-endpoint refusal.

Validation in this extracted environment: `python -m compileall -q infer_stack tests` passes. The focused pytest command could not start under the system interpreter because `ubelt` is absent. `uv run` attempted to create the locked environment but network/DNS access is unavailable and the cache lacks `idna==3.15`, so the normal pytest suite cannot be executed here. This is an environment dependency gap, not a test failure; the focused suite should be run in the normal infer-stack development environment before merge.

Verification addendum: because full project imports are blocked by the missing offline dependencies, I loaded the dependency-light `models`, `hardware`, `catalog`, `placement`, and `suggest` modules directly and exercised the new contracts. The direct checks confirm that unpinned and pinned endpoints have distinct compatibility keys while auto omits `gpu_indices`, tp=2 rejects a one-GPU pin, the planner places an endpoint pinned to GPU 1 exactly on GPU 1, Qwen3-8B is rejected on 16 GiB and accepted on 24 GiB, and the generated suggestion points at `hf://Qwen/Qwen3-8B` with a 32,768 runtime length. `git diff --check` also passes.

## 2026-09-17 18:24:32 -0400

Summary of user intent: correct my earlier Qwen3-8B interpretation. The desired local candidate is the newly released `Qwen/Qwen3.8-27B`, and `Suggest from my GPUs` should produce a configuration that is actually practical on an RTX 3090 using the measured HyperQwen serving path rather than pretending the official BF16 checkpoint fits directly.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I replaced the accidental Qwen3-8B pool entry with a hardware-gated Qwen3.8-27B recipe. The official Hugging Face config reports a 262,144-token native context and the W4A16 AutoRound derivative documents the full 27.78B model at about 19.5 GB on disk; HyperQwen documents that the published W4A16 checkpoint still needs its BF16 embedding/lm-head/MTP tensors requantized before a 24-GiB card can serve it. Accordingly the catalog source is the actual prepared starting artifact, `dbirks/Qwen3.8-27B-W4A16-AutoRound`, not the much larger official BF16 files, while the pool metadata records the official 262K context. The suggestion is accepted only when a detected GPU name contains `rtx 3090` (also matching 3090 Ti), and the generated endpoint preserves that measured-hardware decision with an exact `placement.gpu_indices` pin instead of later allowing generic best-fit placement onto another 24-GiB device.

I added a closed-set `runtime.serve_recipe` mechanism rather than generic command/environment escape hatches. `hyperqwen-3090-single` is part of vLLM process identity only when present, preserving old stock-vLLM compatibility hashes. The Compose renderer uses HyperQwen's immutable image `ghcr.io/syv-ai/hyperqwen:sha-684e927` (the Docker-image workflow for full commit 684e9277f163d1701d6179194c7f6bc1b9175d44 completed successfully), invokes its `single` entrypoint, maps its internal port to infer-stack's standard 8000, enables the repo's recommended `SPEC=dflash2` and `PREFIX_CACHE=1` profile, keeps the measured 64K context and 0.93 GPU-utilization startup setting, and overrides the hard-coded served name through trailing `EXTRA_ARGS` so infer-stack aliases remain truthful. `/app/models` and `/cache` are bind-mounted under infer-stack's runtime state so the one-time ~20 GB download/requantization, fast variant, DFlash2 drafter, torch.compile output, Triton kernels, and HF cache survive reclaim/reacquire. KubeAI fails this recipe closed because its Model CR renderer only knows stock VLLM semantics.

One subtle correction was needed in the suggestion join: its generic 0.92 utilization clamp ran after `max(default, footprint)` and therefore could lower HyperQwen's measured 0.93 default, contradicting the existing invariant that derived values never lower a curated default. The upper clamp now rises to at least an explicit pool default. Tests cover the Hugging Face/HyperQwen metadata, 3090 inclusion and non-3090 24-GiB exclusion, exact hardware pin propagation, recipe compatibility/validation, Compose command/env/volumes, and KubeAI refusal.

Validation: `python -m compileall -q infer_stack tests` and `git diff --check` pass. Full focused pytest cannot collect in this extracted checkout because `ubelt` is absent. I therefore loaded the dependency-light modules directly: suggestion/catalog checks confirm Qwen3.8-27B selects GPU 3 on a mixed RTX 8000 + RTX 3090 host, preserves the 0.93/64K HyperQwen runtime, and is absent on a generic RTX A5000; a stubbed dependency import of Compose confirms the exact `single` command, immutable image, DFlash2/prefix-cache environment, persistent HyperQwen volumes and GPU 3 device reservation; and the analogous KubeAI direct check confirms the recipe is unrenderable with a Compose-only diagnostic.

The user additionally asked that HyperQwen be credited as related work. I added a README `Related work` entry that links the upstream project and states the ownership boundary explicitly: the specialized 3090 model preparation, patched-vLLM launch path, speculative decoding, and low-level tuning remain HyperQwen work; infer-stack contributes discovery, suggestion, exact affinity, lifecycle, and routing integration.

## 2026-09-17 18:44:17 -0400

Summary of user intent: correct the follow-up UX around the new RTX 3090 Qwen3.8 integration. The dbirks/HyperQwen derivative must have an identity distinct from the official Qwen/Qwen3.8-27B checkpoint; exact GPU placement belongs inside the normal endpoint Edit dialog rather than in a separate action; and real endpoint activation must work from both double-click and the Acquire button.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I renamed the hardware-specific suggestion to `qwen3.8-27b-dbirks-hyperqwen` and deliberately left the unsuffixed `qwen3.8-27b` identity unused for the official checkpoint. The source remains `hf://dbirks/Qwen3.8-27B-W4A16-AutoRound` and the runtime remains the pinned HyperQwen 3090 recipe. Because the previous overlay could already have emitted the ambiguous name into a live catalog, suggestion apply now has a signature-gated migration: it renames only the exact old dbirks model + same-named HyperQwen endpoint pair and refuses to reinterpret an official/user-authored `qwen3.8-27b` or a model referenced by other endpoints.

The dedicated GPU-pin control was removed. `Edit endpoint` now owns the placement field alongside tp/dp/runtime controls, shows the detected physical GPUs, accepts `auto` or exact indices, and preserves unexposed endpoint/runtime structure (including HyperQwen's `serve_recipe`, image, protocol, placement floor, and pipeline-parallel setting). A pin change still validates tp×pp×dp and evicts the old idle deployment only after the new catalog validates.

The activation path no longer reconstructs double-clicks from app-level screen coordinates and timestamps. The endpoint DataTable consumes Textual's native click-chain metadata (chain 2) and row metadata, Enter activates the current row, and the Acquire button dispatches `app.acquire` directly. I also fixed a concrete regression in the previous implementation: the per-endpoint `_acquire_inflight` guard was never cleared, so one completed or failed attempt could make later clicks inert. Acquire completion now always clears that guard, reports the actual lease id on success or the exception on failure, and refreshes the dashboard.

Validation in this extracted checkout: `python -m compileall -q infer_stack tests` and `git diff --check` pass. A dependency-light direct import check confirms the renamed pool entry is suggested only onto the matched RTX 3090, carries the exact GPU pin and HyperQwen recipe, and migrates the old ambiguous generated identity while leaving an official Qwen source untouched. The focused pytest command cannot start because the container has neither the project dependencies nor working package-index DNS; `uv run` failed while trying to fetch `requests==2.34.2`. The normal development environment should run the TUI and suggestion tests before merge.

## 2026-09-17 19:02:22 -0400

Summary of user intent: repair the four regressions exposed by the normal infer-stack test environment after the Qwen3.8/GPU-pin TUI follow-up, and provide a direct CLI path for acquiring the already-suggested dbirks/HyperQwen Qwen3.8 endpoint so backend startup failures can be observed outside the TUI. The reported failures were the old-identity migration test calling `write_text` on a string, double-click not acquiring, successful Acquire status being overwritten by a deployment relationship hint, and the endpoint editor's Save button falling outside a short terminal after GPU placement moved into Edit.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I kept the production fixes at the UI event/layout boundaries. `_EndpointTable` still reads the DataTable row from Textual cell metadata rather than reverse-mapping screen coordinates, rejects non-left clicks, and keys the gesture on endpoint identity. It now accepts Textual's native `Click.chain >= 2` when the terminal driver supplies it and also keeps a 0.5-second same-endpoint fallback for drivers/tests that emit independent chain-1 clicks. This matters because Textual 8.2.7's `Pilot.click()` explicitly bypasses `App.on_event`; two separate Pilot clicks each begin at chain 1, whereas `Pilot.double_click()`/`click(times=2)` synthesizes chain 1 then 2. The fallback also covers terminals that fail to preserve a click chain without reintroducing the old coordinate hit testing. The migration-test failure was not a production defect: its helper returns a string path, so the test now writes through `tmp_path / 'catalog.yaml'` directly.

Acquire completion now refreshes the cheap ledger-backed lease/deployment snapshot before publishing the result and gives mutation status a short sticky window. Passive RowHighlighted relationship messages respect that window, so `acquired <endpoint> — lease <id>` or an immediate acquire exception survives the repaint instead of being replaced by `deployment … held by …`. The endpoint editor is now a fixed-height modal whose form body is a `VerticalScroll`; Save/Cancel sit outside the scroll region and remain reachable at the default short test-terminal height. This retains GPU placement inside Edit without making the dialog unusable.

Validation: `python -m compileall -q infer_stack/tui.py tests/test_cli_catalog.py tests/test_tui.py` passes, changed-line whitespace/line-length checks pass, and I inspected Textual 8.2.7's upstream Pilot/Button behavior to ground the click-chain fix. This container still lacks Textual, scriptconfig, and ubelt, and package-index DNS is unavailable, so the exact headless pytest cases cannot execute here. The resulting overlay is intentionally small and should be checked with the four previously failing tests first, then the focused TUI/catalog suite in the normal development environment.

## 2026-09-17 19:44:59 -0400

Summary of user intent: restore the intended `config init -> catalog suggest --apply -> acquire` leasing UX after the recently introduced frozen-profile safety work made `infer-stack config publish` a mandatory third configuration surface, while preserving the crash/recovery race guarantee that motivated the snapshot. Also fix a TUI log-view bug where buffered LiteLLM output can appear after the user switches to a specific vLLM service.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I made the persisted render profile explicitly an internal recovery snapshot rather than user-authored configuration. Acquire now synchronizes that snapshot under the existing global publication lock. If the Compose stack is quiescent (no ACTIVE lease and no managed deployment container), it adopts current user settings/catalog wholesale before admission. During a live leasing epoch, global render inputs stay frozen but current catalog snapshots are appended only when their endpoint/bundle/route definitions are semantically compatible with everything already frozen; the `CatalogUnion` conflict check runs before the candidate is persisted. This makes ordinary compatible additions -- including a newly suggested endpoint -- immediately acquirable without `config publish` while preserving the exact definitions needed to recover existing deployments. Conflicting redefinitions fail closed until the managed stack is quiescent, then a retry adopts current config automatically. Explicit `config publish` remains available only as an advanced multi-runbook pre-seed / preview / image-prepull operation. An already explicitly seeded multi-catalog union is not compacted merely because a later runbook presents one subset: auto-synchronization runs only on real profile drift.

The user catalog is now the resolution source for ordinary CLI acquire/run paths; the recovery union is only a fallback when no current catalog exists. Long-lived TUI sessions update the controller's invocation catalog whenever the catalog is reloaded after an edit/suggest operation, so the next Acquire follows the same locked synchronization path as a fresh CLI process. I documented this as `docs/adr/0001-user-config-is-authoritative.md`, added the three-step workflow prominently to the README, and updated known limitations and backend-mismatch diagnostics so no normal error tells users to publish configuration.

For logs, the Docker command was already correctly scoped when a concrete service is selected. The leak was a stream-switch race: the old `docker compose logs -f` worker could drain buffered stdout after its process was terminated, after the pane had been cleared and relabelled for the new vLLM service. TUI log streams now carry a monotonically increasing generation. Switching/collapsing invalidates the old generation before process termination; every emitted line is accepted only if its generation is still current, and an old worker that finishes spawning late self-terminates. A regression also verifies that a named vLLM log follower passes only that exact service to `docker compose logs`, never LiteLLM. This implements the isolation part of the pre-existing WP18 log-process follow-up without broad text filtering.

Validation: `python -m compileall -q infer_stack tests` and `git diff --check` pass. Six focused recovery-snapshot tests pass, including quiescent auto-adoption, compatible live catalog merge, conflicting live redefinition, first mutation, one-time drift warning, and preservation of an explicitly seeded catalog union. Running all of `tests/test_leasing_profile.py` with the existing temporary ubelt stub reaches 26 passing tests; the five remaining failures are only CLI imports because this container lacks `scriptconfig`. Textual is also unavailable here, so the TUI regression cases cannot execute in this container and should be run in the normal project environment.

## 2026-09-22 19:58:53 -0400

Summary of user intent: keep HyperQwen serving configuration as ordinary infer-stack catalog data, but make GPU-aware suggestion useful on the reference RTX 3090 by offering the measured long and huge context profiles automatically. Do not revive named serve recipes or make apply/acquire silently retune endpoints based on hardware; the TUI must continue to edit/preserve the same generic runtime fields.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I extended the suggestion-pool data model with generic, hardware-gated endpoint variants. A variant is an additional endpoint over the same model identity: it deep-merges generic runtime overrides such as env/max_model_len, can require GPU-name substrings, and when hardware-gated is pinned to the concrete matching GPU so a later best-fit placement cannot move a measured profile elsewhere. Hardware is consulted only while suggesting; the generated catalog remains explicit and deterministic for apply/acquire.

For qwen3.8-27b-dbirks-hyperqwen, the base fast endpoint now states CTX=fast explicitly instead of relying on HyperQwen's internal default. On an RTX 3090, catalog suggest additionally emits -long (max_model_len=150000, SPEC=mtp, CTX=long) and -huge (max_model_len=245760, SPEC=dflash2, CTX=huge), both referencing the one dbirks model entry and inheriting the same command/env templates/mounts. Other Ampere-or-newer >=24 GiB GPUs still get the portable fast endpoint; the extra profiles are not injected there, though users can author the same generic runtime data manually. The legacy serve_recipe compatibility translation also gains explicit CTX=fast so old catalogs normalize to the same launch description.

Tradeoff: multiple suggested endpoints make the measured 3090 choices discoverable without adding TUI preset machinery, but they intentionally do not make HyperQwen mode selection part of runtime semantics. This keeps the control plane reproducible and lets the existing endpoint editor manipulate the variants as normal endpoints. I did not add the reproduction/batch HyperQwen modes because the user's immediate need was context capacity and the agreed suggestion concept was fast/long/huge; those can be added later as pool data if useful.

Validation: python compileall and git diff --check pass. With a minimal ubelt stub (the extracted checkout lacks installed project dependencies), tests/test_leasing_suggest.py, tests/test_leasing_compose.py, and tests/test_leasing_catalog.py pass: 159 passed, 4 skipped. The focused Textual TUI test remains unavailable in this extracted environment because Textual is not installed; no TUI production code changed, and existing tests already cover preservation/editing of command/env/mounts for custom launchers.

## 2026-09-22 20:57:32 -0400

Summary of user intent: fix the acquire failure exposed by the new generic HyperQwen launch fields. The 3090 `-huge` endpoint previews correctly (`CTX=huge`, `SPEC=dflash2`, 245760 context), but after approval infer-stack immediately renders zero deployments and rolls the lease back with “rendered state differs from what was approved at publication.” Preserve the approval guard; make semantically identical generic launch mappings render byte-identically across the ledger persistence boundary.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

The failure is an ordering mismatch, not a HyperQwen or placement problem. Admission preview renders the fresh `EndpointRequest.spec`, whose `runtime.env` / `runtime.mounts` mappings retain catalog YAML insertion order. Commit persists the deployment through `SqliteStore`, whose JSON serializer uses `sort_keys=True`; reloading therefore reconstructs those nested mappings in sorted order. Compose intentionally serializes YAML with `sort_keys=False`, so the post-commit service is semantically identical but textually different. The publication approval digest hashes those rendered file bytes, correctly treats the difference as unapproved, and the acquire rollback removes the new lease. Generic mapping-valued launch fields made this old serialization asymmetry observable.

I kept the safety invariant intact and canonicalized only semantically unordered launch mappings at the Compose renderer boundary. Custom environment variables now render in sorted key order after infer-stack's owned `HF_TOKEN`; custom runtime mounts render in sorted container-target order. Command/argument lists remain ordered. This makes fresh-catalog and sqlite-reloaded deployments generate identical Compose bytes while preserving meaningful list order and the existing approval-digest check.

Regression coverage mirrors the real boundary two ways. A render-level test takes a custom-launch deployment whose env/mount mappings are deliberately non-alphabetical, JSON round-trips its `spec`/`served` with `sort_keys=True` exactly as `SqliteStore` does, renders again, and asserts the complete Compose YAML is byte-identical. An admission-level test then acquires the same shape through `Controller`, verifying the lease remains ACTIVE and the publication marker clears after preview -> sqlite commit -> post-commit render. With a minimal temporary ubelt stub (the extracted checkout lacks project dependencies), the full Compose + admission suites pass: 115 passed, 4 skipped. `compileall` and `git diff --check` also pass. No HyperQwen-specific runtime logic was added.

## 2026-09-28 13:07:17 -0400

Summary of user intent: move from VM-only KubeAI testing toward practical real-cluster use, and consolidate the currently scattered K3s/NVIDIA/KubeAI setup knowledge into an ergonomic infer-stack surface without turning infer-stack into a general Kubernetes installer. The requested deliverable is an overlay implementing the design.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I added a deliberately narrow `infer-stack kube` modal. `kube nodes` inventories Ready state and GPU scheduling/discovery facts. `kube setup` is a read-only capability plan by default; `--apply` reconciles only the cluster integration infer-stack needs. The planner treats working externally managed GPU resources/labels or KubeAI components as satisfying the capability instead of taking them over. When infer-stack does need to manage NVIDIA support, it requires an already-functional `nvidia` RuntimeClass and then installs/reconciles the pinned NVIDIA device-plugin chart with GPU Feature Discovery; host drivers and the NVIDIA container runtime remain explicitly outside infer-stack's ownership. KubeAI setup merges discovered resource profiles into existing/operator Helm values without overwriting same-named hand-tuned profiles, preserves an installed chart version unless explicitly changed, supports an operator `--values` file, and keeps chart `secrets.*` out of both Helm argv and the persistent generated values file by carrying them only through a short-lived mode-0600 values file. An exported `HF_TOKEN` overrides the preserved Hugging Face token in that ephemeral layer.

K3s remains an explicit convenience implementation underneath Kubernetes rather than being implied by the generic kube surface. `kube k3s bootstrap` owns the former bootstrap flow, configures K3s's persistent kubeconfig mode instead of copying a second kubeconfig authority, waits against `/etc/rancher/k3s/k3s.yaml` explicitly, symlinks a clean `~/.kube/config` to that authority but never overwrites an existing kubeconfig, and installs Helm when absent; an explicit version mismatch on an already-running server refuses an implicit upgrade. `kube setup` also prints the active Kubernetes context before its capability plan so an operator can see which cluster an apply would target. `kube k3s join` reads the cluster token from a file and passes it through the installer environment rather than argv. The old `scripts/bootstrap_k3s.sh` and `scripts/join_agent.sh` are compatibility wrappers around these commands, so they no longer duplicate lifecycle logic. `scripts/install_kubeai.sh` remains available as a low-level/manual escape hatch rather than becoming a second recommended workflow.

I also factored KubeAI resource-profile generation into a shared backend helper so `catalog suggest` and cluster setup use the same mapping, updated KubeAI doctor/catalog remediation text, and rewrote the README/KubeAI backend/parity docs around the new plan/apply authority. A late review caught two important edge cases: Helm may be absent only when every needed GPU/KubeAI capability is already externally managed; if the plan has any reconciliation action, missing Helm is a required blocker. Also, preserving existing Helm values must not copy `secrets.*` (including a prior Hugging Face token) into generated YAML; those values are now split into the ephemeral secret file and preserved across upgrades without being persisted by infer-stack. The explicit K3s subcommands inherit `kw.Config`, matching the rest of the kwconf command surface.

Validation in this extracted/offline checkout: `python -m compileall -q infer_stack tests`, `bash -n` on the compatibility wrappers, and `git diff --check` pass. `tests/test_cli_kube.py` plus four focused existing KubeAI sizing/measurement tests pass (14 passed, 60 deselected). Running the complete two relevant test files reaches 67 passing tests; the remaining seven tests fail at collection/import only because `kwconf` is not installed in this container. A minimal temporary kwconf/ubelt shape stub outside the repository successfully imports the complete CLI tree and verifies the nested `ManageCLI.kube -> KubeModalCLI.k3s -> bootstrap/join` registration. Package-index DNS is unavailable, so I could not execute the real kwconf parser/help tree or the full project suite; those should be run in the normal development environment before merge.

## 2026-09-28 13:52:00 -0400

Summary of user intent: keep the new Kubernetes integration from acquiring K3s
as an architectural dependency while retaining K3s as infer-stack's first
well-supported workstation-cluster provisioning path.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I tightened the distribution boundary rather than inventing a provider
abstraction before a second provisioning integration exists. The generic
`KubeManager` and KubeAI backend diagnostics no longer mention K3s-specific
remediation; they report missing Kubernetes/container-runtime capabilities in
distribution-neutral terms. K3s knowledge remains in the explicit
`infer_stack.kube.k3s` leaf module and `infer-stack kube k3s` CLI namespace.
Package/CLI docstrings now state that separation explicitly.

I added `docs/cluster-setup.md` as the operator runbook. It distinguishes the
Kubernetes control plane from infer-stack authority, shows the initial K3s
server + agent topology (with the server also usable as a GPU worker), gives the
create/join/reconcile sequence, and gives an existing-cluster path that starts
directly at `kube nodes` / `kube setup`. README and the KubeAI backend guide
link to this runbook and identify K3s as the first provisioning target rather
than the generic backend. Focused tests assert that generic missing-kubectl and
missing-NVIDIA-runtime diagnostics remain distribution-neutral.

Validation: with the same minimal offline ubelt/kwconf stubs used for the prior
overlay, `tests/test_cli_kube.py` passes 12/12. The Kubernetes CLI suite plus
the non-parser KubeAI sizing/doctor checks pass 16/16. Running both complete
relevant files reaches 71 passed / 5 failed; every failure is a CLI-entry test
that reaches the intentional `kwconf.Config.cli` stub, so the real kwconf parser
remains the only unavailable test dependency in this environment. `compileall`,
`bash -n` on both compatibility wrappers, and `git diff --check` pass. A
cumulative archive was applied to a pristine extraction of the supplied source;
all overlay files compared byte-identical, executable modes were preserved, and
the 16-test focused verification passed there as well.

## 2026-09-28 14:42:27 -0400

Summary of user intent: make two workstations that already have working Compose-backed infer-stack installations practical for real KubeAI testing without destroying their existing local state. The machines need to be pooled into Kubernetes, then temporarily handed back to direct Compose use, and later returned to the same cluster only when the operator explicitly requests it.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

The main design decision was to avoid treating temporary Compose use as Kubernetes leave/rejoin. A worker does not need its K3s/kubelet agent stopped merely to make its GPU safe for local Compose; doing so would create more lifecycle state, and stopping the sole K3s server would also take down the cluster control plane. Instead, the generic Kubernetes layer now models a reversible scheduling handoff. `infer-stack kube node detach NODE` is read-only by default; `--yes` marks an infer-stack-owned detach, cordons the node, and drains controller-managed workloads while leaving DaemonSet/static pods and the distribution agent/control plane running. `attach --yes` waits for that same node identity to be Ready, uncordons it, and removes the marker. This is distribution-neutral because it uses only Kubernetes APIs, so it does not weaken the earlier K3s boundary.

The cordon has explicit ownership and crash recovery. Infer-stack refuses to adopt or undo an unrelated operator cordon. Detach writes a `requested` annotation before mutation and changes it to `true` only after a successful drain; an interrupted operation can therefore be retried or reversed instead of becoming an ambiguous foreign cordon. Bare/unmanaged pods are a hard stop rather than a reason to add `kubectl drain --force`. `kube nodes` now exposes schedulability, `compose` for a completed temporary detach, `pending` for an interrupted detach, and `CONFLICT` if someone externally makes a Compose-marked node schedulable.

A second issue was backend recovery state. The existing leasing design intentionally refuses to change backend kind inside one recovery ledger, so switching a Compose-configured control workstation to `config set backend kubeai` and reusing its data root would make the physical-node workflow look easy while leaving a recovery-snapshot mismatch. I preserved that invariant rather than weakening it. The documented temporary-testing path keeps the persisted Compose default/ledger untouched and runs the one KubeAI authority with `INFER_STACK_BACKEND=kubeai` plus a separate `INFER_STACK_DATA_DIR=$HOME/.local/share/infer_stack-kubeai`. The normal config/catalog root remains shared. Unsetting those variables returns the control workstation to its original Compose authority; ordinary worker nodes never need a local KubeAI authority at all.

The resulting two-machine loop is: quiesce local Compose on A/B, create/join Kubernetes once, run KubeAI from its separate authority/data root, release KubeAI workloads, detach A/B for Compose, use their old Compose installations, quiesce Compose, attach A/B, and resume the same KubeAI authority. There is no second `kube k3s join` because detach preserves membership and agent state. A K3s server that is also the sole control-plane node can be detached from workload scheduling without stopping the server; cluster workloads may become Pending when all workers are cordoned, but the control plane remains available to attach them again.

Validation in this extracted/offline checkout: `tests/test_cli_kube.py` passes 18/18 with the same minimal ubelt stub used for the Kubernetes work; the tests cover reversible detach/attach, retained DaemonSet/static pods, refusal to force-delete unmanaged pods, refusal to undo foreign cordons, interrupted-detach retry, control-plane handling, and node inventory state. Four existing KubeAI doctor/sizing tests also pass. A minimal external kwconf/ubelt shape stub imports the full CLI tree and verifies `ManageCLI.kube -> KubeModalCLI.node -> status/detach/attach`. `python -m compileall -q infer_stack tests` passes. The container still lacks the real ubelt/kwconf and ruff packages, so the real parser/help path and repository lint suite should run in the normal development environment before merge.

## 2026-10-02 11:19:47 -0400

Summary of user intent: fix the Compose GPU doctor false-negative observed on aiq-gpu after handing the node back from Kubernetes. The live reproducer showed cached non-interactive sudo working, while GNU find over /proc returned status 1 because one fd disappeared during traversal even though it produced 52 valid NVIDIA-device holder records. Those holders were nvidia-persistenced, the Kubernetes NVIDIA device plugin, and nvtop; none represented a CUDA compute workload.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I kept the fix inside the GPU diagnostic rather than weakening backend admission. The privileged holder scan now verifies sudo independently, runs the /proc traversal with a stable C locale, and accepts a nonzero find exit only when every reported error is the expected live-/proc race (a /proc path vanished with ENOENT). Valid stdout is retained. Real permission/execution errors still fail closed. A second race was handled at the pid-detail stage: if the process disappears before cmdline/cgroup can be read, that stale holder is dropped rather than formatted as a current conflict.

The holder classification now distinguishes passive/system observers from unexpected holders. nvidia-persistenced, nvidia-device-plugin, gpu-feature-discovery, and nvtop are informational: they can hold NVIDIA character devices (and may block a low-level reset) but do not by themselves reserve GPU memory or make a Compose/vLLM placement unsafe. Unknown holders still fail the check and preserve the container/pod attribution. The reset-specific note is emitted only when nvidia-persistenced is actually present.

Validation in this extracted checkout: python compileall passes. The normal pytest entrypoint cannot collect because ubelt is absent from the container, so I ran a dependency-light direct harness that reproduces the exact sudo=0/find=1+ENOENT+valid-stdout case and the observed three-process holder set; both pass. Focused pytest regressions were added for the transient /proc race, real find errors, and passive holder classification. The overlay is direct-overwrite and intentionally limited to the diagnostic, its tests, this journal, and the reusable /proc lesson.

## 2026-10-03 — review of LiteLLM context metadata patch

Reviewed commit `2ba288caad36` against the context-metadata requirements. The
main propagation was sound, but two correctness gaps remained:

1. `max_model_len` participates in capacity subsumption. A 65K endpoint can
   coalesce onto an already-running compatible 262K deployment, so deriving
   every route's `max_input_tokens` from the deployment runtime leaks the
   larger process capacity into the smaller endpoint alias and makes static
   gateway metadata change across acquire/release. The catalog resolver now
   persists each vLLM endpoint's effective context contract in its per-alias
   `served` payload. Route rendering uses that value, falls back to the current
   catalog for legacy deployments, and only then falls back to deployment
   capacity for ad-hoc aliases. Compose and KubeAI use the same helper.

2. LiteLLM may synthesize `model_info.max_input_tokens` from its bundled model
   metadata for an external route even when infer-stack never supplied that
   field. Dynamic reconciliation previously compared it unconditionally, so a
   known external model could be deleted and re-added every converge. Reconcile
   now ignores observed context only when the desired route has no
   infer-stack-owned context; managed values remain strict drift semantics.

Focused CPU-only validation in the extracted source (with only a minimal
`ubelt` import stub because the archive environment lacks project deps):
`tests/test_leasing_context_metadata.py`, dynamic-routing, and route-registry
suites pass; broader catalog/ledger/admission/routing coverage passes apart
from two Compose tests that also fail on the unmodified base in this container
and CLI tests blocked by missing `kwconf`. `compileall` is clean.


## 2026-10-03 13:18:00 -0400

Summary of user intent: document a detailed future architecture for infer-stack
that can be controlled remotely, can eventually run multiple redundant
control-plane processes without creating competing authorities, and treats
security of the remote control surface as a primary design constraint.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

I wrote a planning document rather than changing runtime behavior. The central
design decision is to distinguish one logical control plane from one physical
control-plane machine: a managed backend domain should have one authoritative
state, while one or more controller replicas may serve API traffic and
reconciliation. Existing inference is deliberately a data-plane concern and
should continue if every controller is temporarily unavailable; only mutations,
reconciliation, TTL GC, and route changes should pause.

The proposed evolution keeps SQLite for simple standalone mode and introduces
PostgreSQL for remote/server/HA mode instead of attempting network-shared
SQLite. The first HA shape is active-active API replicas plus one elected
reconciler, with monotonically increasing leadership epochs and per-object
desired generations so stale leaders cannot overwrite newer state. Remote
Compose hosts are controlled through authenticated node agents rather than SSH
or exposed Docker sockets; Kubernetes/KubeAI continues to use Kubernetes as its
scheduler/agent substrate rather than adding a competing infer-stack scheduler.

Security is intentionally part of the architecture rather than a later
hardening pass. Remote listeners default to loopback, remote access requires
TLS/authentication, normal users operate only on approved catalog endpoints,
catalog mutation is a separate privilege, and the control API must not expose a
generic exec surface or arbitrary images/commands/host mounts. The plan also
defines durable operations/idempotency, audit records, failure semantics,
backup/restore expectations, staged implementation phases, and a test strategy
for leader failover, stale-generation fencing, node partitions, and data-plane
independence.

I also linked the current "one control plane per host or backend namespace"
known limitation to this proposed successor architecture so the repository
clearly distinguishes today's supported behavior from the future direction.
This is documentation-only; no tests were required or run. The principal open
questions left intentionally unresolved are API transport, agent transport,
authentication provider, catalog authority, and whether infer-stack and LiteLLM
share one physical PostgreSQL service or only colocate operationally while
keeping separate schemas/credentials.

## 2026-10-03 14:31:00 -0400

Summary of user intent: audit and simplify the current infer-stack implementation by consolidating duplicate authorities and removing LLM-style compatibility/plumbing residue, while first reconciling that cleanup with the newly merged distributed-control-plane/server plan. The result should make the code easier for human maintainers and smaller coding models to follow without introducing a new abstraction hierarchy or removing functionality.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

The HA plan changed one part of the earlier cleanup proposal materially. The ledger generation counters should not be dismissed as obsolete just because the standalone controller now uses a richer publication marker for render/apply crash recovery. Future multi-controller reconciliation explicitly needs monotonic desired generations and a leadership fencing epoch. I therefore kept the current coarse generation state and rewrote its documentation around desired-state ownership and the future per-object migration. Similarly, recovery-profile inputs remain intentionally distinct from editable user config, desired ledger state remains distinct from live backend residency, and published-catalog routes remain semantically distinct from remembered ad-hoc route-registry rows. Those are load-bearing separations, not duplicate authorities.

The implementation cleanup focuses on false authorities. Catalog YAML mutation now has one small persistence module (`leasing/catalog_edit.py`) that loads editable source, validates it through the real `Catalog` parser, canonicalizes compatibility spellings, and atomically writes it; both CLI and TUI use that path. `served_name` is the only spelling infer-stack writes, while legacy `public_name` remains accepted and conflicting dual definitions are rejected. Catalog and CatalogUnion share plain expansion/request helpers instead of copy-pasted mechanics. Stale structural-field constants, dead CLI mixins, and historical internal function aliases were removed. The deprecated `require_generation` flag no longer becomes inert backend state, although the old constructor keyword remains accepted for Python compatibility; readiness still unconditionally verifies a generation.

The front-door boundary is now literal: Compose owns a `Gateway` and `front_door()` returns that gateway rather than making `ComposeBackend` a second operational API for keys and route-registry state. Controller/CLI/TUI code addresses front-door operations through the capability. KubeAI similarly returns the front door of its gateway realization. `ResolvedEndpoint` and `EndpointRequest` are also made conceptually distinct inside the repository: new/internal code uses `to_request()` explicitly for managed deployment demand. The pre-existing read-only request properties remain as compatibility-only derived views, so external Python callers do not lose functionality while the repository has one obvious authoritative path.

I added `docs/planning/control-plane-authority-inventory.md` as the distributed plan's Phase-0 deliverable. It classifies current catalog/config intent, frozen recovery inputs, SQLite ledger state, publication state, backend observations, gateway secrets, route registry, dynamic LiteLLM realization, render artifacts, sidecars, hardware observations, and measurements. It also states which local authorities must eventually move behind transactional state-store or secret-store boundaries. The document is deliberately explicit that file locks are local serialization rather than distributed fences, rendered files are reconstructable artifacts, the route registry is durable control state, and a generic state-store interface should be introduced when SQLite and PostgreSQL actually coexist rather than speculatively wrapping today's store.

I deliberately did not fold the TUI's Kubernetes replica-readiness view into a new generic status/view-model abstraction in this change. Compose residency, KubeAI Model declaration, and Kubernetes replica readiness are not presently the same fact, and the TUI also gates expensive cluster polling on visible panes. Forcing those through a new hierarchy now would add indirection immediately before the HA observed-state/generation design gives that boundary a durable meaning. The authority inventory makes the future direction explicit instead.

Validation in this extracted/offline checkout: `python -m compileall -q infer_stack tests` and `git diff --check` pass. A direct dependency-light harness verifies catalog canonicalization, conflicting-name rejection, and friendly malformed-YAML failure. With a minimal external `ubelt` stub, the focused catalog/admission/Compose/model-serving suite passes 187 tests with 4 skipped; two known Compose lifecycle tests are intentionally deselected because they fail identically on the pristine supplied base in this container. A broader run reached 240 passing before CLI cases hit the environment's missing `kwconf`; an attempted dependency install could not reach the package index. The supplied base journal records 1292 passed / 3 skipped before this overlay, so the real kwconf CLI suite should be rerun in the normal development environment before merge.

## 2026-10-07 19:18:12 -0400

Summary of user intent: preserve the first real two-node K3s/KubeAI bring-up
while the debugging context is fresh, answer whether the TUI belongs on worker
nodes, capture the kubeconfig/setup footguns, and finish or at least record the
cleanup work exposed by worker acceptance so a lost session does not lose the
operational lessons.

Model: GPT-5.6 Sol. Configuration: tool-enabled reasoning session.

The physical test used `yardrat` as the K3s server/admin host and
`namek.kitware.com` as an agent with one RTX 3090. The generic cluster path
worked after one provisioning defect already fixed earlier in the session: the
K3s network installer was incorrectly subject to the generic 60-second command
timeout, while the real worker download/install needed longer. With both nodes
Ready, NVIDIA device-plugin/GFD present, and KubeAI installed, targeted worker
acceptance eventually passed end to end: a fresh runtime/device probe on namek,
exact-node placement of a one-GPU vLLM replica, device UUID/product verification,
and a real OpenAI generation. The broader physical `p5_two_hosts.sh` gateway /
NodePort / secret-rotation handover is still pending.

The failed attempts were useful. The first vLLM pull crossed kubelet ephemeral
storage pressure on namek. K3s/containerd itself occupied only about 2 GiB; the
root filesystem was shared with a much larger stale Docker image/build cache.
Node events showed repeated `Evicted` events while the roughly 14 GB vLLM image
was pulled/extracted. The host's K3s defaults were 5% hard
`imagefs/nodefs.available`, 10% minimum reclaim, and a five-minute pressure
transition period, so freeing enough space to merely cross 5% did not clear the
NoSchedule taint immediately. After root free space reached roughly 162 GiB on
a 938 GiB filesystem and the transition period elapsed, DiskPressure cleared
automatically. The reusable lesson is to diagnose kubelet's own nodefs/imagefs
stats and node events; do not assume `du /var/lib/rancher/k3s` identifies the
ephemeral-storage consumer, and do not hand-remove the pressure taint.

That eviction exposed a worker-test state-machine bug. The Model polling loop
removed `Failed` pods from its active set before interpreting them, so an
`Evicted` pod changed the displayed state from "container running; model
loading" back to "waiting for scheduler" and consumed the rest of the
15-minute timeout. The repair in this change treats failed serving pods as
terminal immediately and includes status reason/message plus container
termination details. A second interrupted run left a Pending, already-bound
Model pod reserving namek's one `nvidia.com/gpu`, which made the next probe
correctly fail scheduling with `Insufficient nvidia.com/gpu`. Explicit cleanup
then hit a 180-second `kubectl wait --for=delete` timeout even though the pod
disappeared at the deadline. Cleanup now re-reads after that timeout, accepts
the deletion race if no pod remains, reports the names/phases/deletion timestamps
when resources really remain, and preserves the original acceptance error if
cleanup also fails.

The admin kubeconfig ergonomics remain intentionally open rather than being
"fixed" by weakening credentials. Bootstrap already provisions a private
`~/.kube/infer-stack-k3s.yaml` and links `~/.kube/config` when safe, but on the
real server K3s had installed its bundled `kubectl` shim; with `KUBECONFIG`
unset that shim still tried root-only `/etc/rancher/k3s/k3s.yaml`. Persisting
`KUBECONFIG` in the operator shell works, but infer-stack cannot set a parent
shell environment, should not silently edit `.bashrc`, and should not make the
root admin file world/group-readable merely for convenience. The queued design
is an infer-stack-owned cluster-target/kubeconfig setting used explicitly by
KubeAI/TUI/kube runners while leaving the operator's general kubectl context
alone.

The TUI boundary is also now explicit in the docs. Today `infer-stack tui` is
an authority UI: it opens the local settings/SQLite ledger/backend state and,
in KubeAI mode, issues Kubernetes operations from that same admin machine. A
worker does not need or want its own KubeAI TUI merely because it runs model
pods; its local Compose configuration may remain intact. Running a separately
configured TUI/ledger on each worker would create multiple infer-stack
authorities for one namespace, which remains unsupported until the proposed
remote/distributed control-plane server exists.

Validation in this extracted checkout: `python -m compileall -q infer_stack tests`
passes, and the changed files pass `git diff --no-index --check`. The focused
pytest regressions cannot collect because `ubelt` is absent; `uv run --offline`
also cannot build the environment because `pygments` is missing from the local
cache. Run `tests/test_kube_worker.py` in the normal development environment.

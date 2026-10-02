## 2026-04-18 20:48:54 +0000
Summary of user intent: refactor `submodules/vllm_service` so named serving profiles become the first-class product abstraction for HELM-audit local serving, while keeping the compose and KubeAI paths usable and adding an explicit export seam for machine-usable HELM bundles.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

I started by reading the existing catalogs, resolver, CLI, compose renderer, and KubeAI renderer before touching structure. The repo already had most of the ingredients: separate model and profile YAML catalogs, multiple backends, and HELM-oriented built-in profiles. The real problem is semantic drift. The code resolves a profile, but the resolved plan is still organized mostly around "services", model defaults, and incidental alias maps. That makes it harder to explain what the public identity is and harder to export a precise contract for helm_audit. My goal for this pass is to make the public serving-profile name intentional in the code, then keep the rest of the stack as a small compiler from that profile into backend-specific artifacts.

The main tradeoff I expect is between cleanliness and compatibility. A full rewrite into dataclasses or a brand-new manifest format would probably look tidy in isolation, but it would widen the change surface and make review harder. I’m choosing a modest refactor: keep YAML-driven catalogs and explicit Python dictionaries, add a normalization/schema layer for base models and serving profiles, and thread richer profile metadata through the resolver and exporters. That should make later documentation much easier while still preserving the narrow seam into helm_audit. The risk is that some older assumptions in templates or commands may still leak "service" terminology; I’ll try to preserve runtime compatibility while making the new naming story obvious in the resolved structures and new CLI paths.

Reusable design takeaways:
1. When a repo already has the right ingredients, prefer turning the intended abstraction into an explicit schema over inventing a new orchestration layer.
2. For reproducibility-sensitive work, transport shape and public identity belong in profile metadata, not as incidental router aliases.
3. Export seams are easiest to trust when generated artifacts are explicit, path handling is predictable, and machine-local values stay overridable instead of being baked into source templates.

Implementation outcome: I introduced a schema/normalization layer that separates base model metadata from serving-profile metadata, then pushed richer resolved profile identity into the plan, backend renderers, and a new HELM bundle exporter. I kept the compose and KubeAI backends intact, but made them consume explicit `serving_profile` and per-service identity fields rather than inferring everything from loose service names. I also added built-in audit-oriented named profiles for the active Qwen, GPT-OSS, Vicuna, and Pythia cases, plus compatibility handling for older `helm-*` names so the public CLI surface can emphasize the new names without hard-breaking prior references.

What surprised me: one unrelated legacy built-in profile referenced a missing model, and the first normalization pass caused that to poison every resolve attempt. That was a good reminder that catalog normalization should preserve inspectability rather than eagerly fail the entire repo. I changed the loader to keep invalid profiles marked as invalid and only fail when someone actually selects them. That feels like the right operational tradeoff here because it keeps the repo debuggable while still making profile-specific errors explicit.

Testing and confidence: I added focused tests for profile resolution, canonical-vs-legacy profile naming, KubeAI rendering, compose rendering, and HELM bundle export with a specific check that `gpt-oss-20b-chat` and `gpt-oss-20b-completions` export different client classes. I also ran a temp-workdir CLI sanity check for `verify-profile` and `export-helm-bundle`. I’m confident in the new named-profile flow and export seam. The main remaining risk is broader legacy-catalog cleanup: the built-in catalogs still contain older mixed/legacy profiles that were not fully redesigned in this pass, and they will deserve a later documentation and deprecation pass once the team agrees on the preferred public catalog surface.

Follow-up compatibility pass: this pass stayed narrow and corrected export contract mismatches rather than reworking the structure again. The big issue was that export behavior had drifted toward “backend implies client shape,” which produced the wrong result for KubeAI and also flattened away some useful audit-side distinctions. I changed the export path to use an explicit transport contract: GPT-OSS profiles now export the same LiteLLM/OpenAI-shaped `model_deployments.yaml` fields that the existing audit bundle expects; Qwen profiles can still export the direct-vLLM convention that the current audit-side Qwen bundle uses; and KubeAI-resolved profiles override those transport hints and always export OpenAI-compatible settings with `/openai/v1` and the public serving-profile name as the request model.

I also made the GPT-OSS migration alias more conservative. The old `helm-gpt-oss-20b` compatibility name now resolves to `gpt-oss-20b-completions`, not the chat variant, because the current audit runbook is effectively completions-first and explicitly treats chat as an opt-in secondary deployment. That change is less elegant than collapsing everything toward chat, but it matches the operational reality better and reduces surprise for existing users. I added exact-shape tests for the exported `model_deployments.yaml` files, plus checks for repo-relative versus machine-local `model_deployments_fpath` handling so the generated bundle examples align with the current `helm_audit` materialization seam instead of assuming a particular cwd.

Cross-repo ownership refactor: this follow-up moved the center of gravity back where it belongs. I added a benchmark-agnostic `describe-profile` surface in `vllm_service` that exports a generic serving-profile contract: profile identity, model identity, protocol/runtime details, backend-default access expectations, and optional additional access hints. The goal was to make the submodule explainable without mentioning CRFM HELM at all. Benchmark client classes, benchmark deployment naming, benchmark manifests, and machine-local bundle layout do not belong in the core serving-profile manager, so I did not add them to the generic contract. Instead, I left the benchmark bundle exporter in place only as a transitional compatibility path and made it print an explicit notice that the preferred owner is now the `helm_audit` integration layer.

The most delicate part was deciding what to do with the old `benchmark_transport` information. I did not want to blindly delete it, because some of those hints are still genuinely useful to external consumers. The compromise was to treat them as optional compatibility access hints rather than as the main contract. The generic contract now has a backend-default access surface plus optional additional accesses, which let the downstream benchmark adapter deliberately choose between routed OpenAI-compatible access and direct-vLLM access without forcing benchmark jargon into the core profile schema. That feels like a better long-term shape: serving profiles stay general, while integrations can still express sharp downstream choices.

I’m happier with the submodule after this change because its story is simpler: define named serving profiles, resolve them, render them, and describe them. The remaining benchmark exporter code is now clearly transitional debt rather than the repo’s identity. That is an acceptable temporary state because it preserves near-term operator usability while making the intended ownership boundary visible in both code and tests.

## 2026-04-18 22:06:28 +0000

Summary of user intent: keep the current ownership boundary intact, but harden the seam by giving `helm_audit` one small public contract-loading API to call and by preserving only serving-side access/auth hints in the generic contract.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This was a good reminder that “generic contract” does not mean “leave every consumer to reconstruct setup details.” The first contract work already moved benchmark translation out of the submodule, but the audit adapter still had to know too much about how to load config, force builtin catalogs on, simulate hardware, and resolve a profile. That meant the public seam was conceptually right but mechanically too wide. I added `vllm_service.contracts.load_profile_contract(...)` as a thin canonical entrypoint that loads config from the repo root, enables the builtin catalogs, applies optional backend and hardware overrides, resolves the selected profile, and returns the benchmark-agnostic contract. The important part is not the amount of code; it is that the policy now lives in one obvious place inside the serving repo rather than being reconstructed elsewhere.

I also tightened the access metadata slightly by making auth expectations explicit with `auth_required`. That still feels like serving-side information to me, because it answers a generic question external consumers need to know: is this access surface expected to require a credential, and if so which env-var convention goes with it? What I explicitly did not move back in are benchmark-only concepts like HELM client classes, benchmark deployment naming, or manifest logic. The generic contract remains useful for any future consumer that wants to inspect how to talk to a resolved service without inheriting benchmark assumptions.

The main tradeoff here is that `load_profile_contract` now imports config/resolution internals inside the submodule’s public contracts layer. I’m comfortable with that because the direction of dependency is still correct: public API over internals within one repo is fine, while cross-repo integration over several internals was the real smell. The tests now cover the new loader directly for the active Qwen and GPT-OSS variants so future refactors in config or resolution have a better chance of preserving the external contract shape.

Design takeaways:
1. A public contract is easier to keep stable when the same module also owns the canonical way to construct it.
2. Auth expectations are part of an access surface; benchmark client mapping is not.
3. When a lower-level repo exports a machine-readable contract, the highest-value public API is often a single “load and resolve this profile for me” function rather than a larger façade.

## 2026-04-18 22:26:38 +0000

Summary of user intent: do a narrow UX pass so a new user can follow the README and get either the Compose or KubeAI backend running without manually editing `config.yaml` or `models.yaml`, favoring one first-class setup command with flags and environment fallbacks.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This pass was mostly about making the CLI tell a more honest story. The repo had already outgrown the old “`init`, then edit YAML” posture, but the main branch UX still forced people back into it because the first real command that needed config would stop with “No config.yaml found.” The cleanest fix was not to make every command invent config on the fly, but to add a single `setup` command that writes or updates `config.yaml` from explicit flags or environment variables, then keep the common commands capable of taking a few important overrides without making users open files. That gives us one clear first-run habit while still keeping the config file as an inspectable artifact instead of hidden process state.

I chose to keep the override surface deliberately narrow and practical. `setup` accepts backend, active profile, compose command, ports, state/runtime paths, namespace, and ingress settings, with matching environment-variable fallbacks. Then `render`, `deploy`, `up`, `status`, `switch`, `smoke-test`, and the profile-inspection commands accept the most useful transient overrides such as backend, profile, namespace, ingress host, and compose command. The main tradeoff here is that not every config field became a flag, but that felt right for this stage of the repo. The goal was not to build a universal config editor; it was to make the common path copy/paste-safe and easy to explain later.

The subtle bug I had to guard against was stale renders under overrides. Once `deploy --backend kubeai --profile ...` becomes supported, a previously rendered Compose plan can look “fresh” even though it is for the wrong backend and profile. I fixed that by treating runtime overrides as a reason to re-render before `up` or `deploy`. That keeps the new override path trustworthy without changing the underlying plan/render structure. I also rewrote the README flows around `setup` and smaller chat-oriented examples for Compose so the built-in smoke test and direct curl commands align with the default request path instead of dropping users into a chat-vs-completions nuance immediately.

I’m confident in the new setup story because the added tests exercise it as a real CLI flow from an empty temp directory: `setup`, `render`, and backend-specific artifact generation, plus environment-variable fallback and transient backend/profile override behavior without persisting those overrides into the saved config. Remaining debt is mostly polish: `init` still exists as a compatibility command, and there are still many lower-signal config fields that are only file-driven. That is acceptable for now because the README no longer needs them for the happy path.

Design takeaways:
1. A copy/paste-safe setup flow is usually better served by one explicit config-writing command than by trying to make every command silently bootstrap state.
2. Allowing transient overrides is only safe if apply-style commands treat those overrides as invalidating prior renders.
3. README examples become much more trustworthy when they use profiles that naturally match the default smoke-test path instead of forcing special cases into the first-run experience.

## 2026-04-18 22:31:21 +0000

Summary of user intent: do a small follow-up UX correctness pass by making `switch` persist only the active profile while keeping one-off overrides transient, and make the KubeAI README path more honest by using a safer first-run example plus cleanup of small setup-first inconsistencies.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This pass was narrower than the prior setup work, but it fixed the most important remaining surprise in the new UX. The first setup-first design deliberately allowed transient overrides on common commands, but `switch` was still saving the fully override-expanded runtime config back to disk. That blurred the boundary between “use this override right now” and “persist this setting for future commands,” which is exactly the kind of silent mutation that makes operators distrust a CLI. I changed `switch` so it now loads the saved config, updates only `active_profile`, writes that back, and only then applies runtime overrides in-memory for the immediate build/render/apply step. That keeps `setup` as the general config-writing command and `switch` as the profile-switching command.

The interesting tradeoff was not technical difficulty, but deciding how much behavior to encode into tests. I chose one focused unit-style test around `cmd_switch` rather than more subprocess integration because the point here is semantic correctness: the saved file should keep its original backend, compose command, and namespace, while the invocation can still temporarily render/deploy with an override backend and namespace. That test gives us much better protection against accidental future regressions than another end-to-end happy path would have.

For the README, I took the safer option and changed the KubeAI getting-started flow to a smaller built-in profile, `qwen2-5-7b-instruct-turbo-default`, instead of the much heavier 72B example. That makes the first-run story more honest without turning the KubeAI section into a hardware tutorial. I also aligned a couple of small drifts introduced by the previous pass: the top-level example now shows `setup ...` followed by `render` instead of `render --profile`, the top-level `describe-profile` example uses the same smaller default profile, and the missing-config guidance points to the same Compose-first profile the README uses. That keeps the onboarding story consistent from error messages through the main command list and backend sections.

Design takeaways:
1. A command that persists state should save only the state it conceptually owns, even if it accepts extra runtime overrides for convenience.
2. When a README happy path is hardware-sensitive, a smaller built-in example is often better than a disclaimer about the larger one.
3. Small message drift matters in setup-oriented tools because users notice inconsistency before they understand the underlying model.

## 2026-04-18 23:53:31 +0000

Summary of user intent: do a focused KubeAI bugfix pass so local validation/render/deploy use the same resource-profile source as the Helm install flow, without requiring users to duplicate resource profiles manually in `config.yaml`.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This pass clarified a mismatch that had been hiding in plain sight: the repo was conceptually treating KubeAI Helm values as the thing we install, but still validating against `config.yaml.resource_profiles` unless that section happened to be present. The cleanest small fix was to introduce an explicit canonical local file, `generated/kubeai/kubeai-values.yaml`, and make the KubeAI path prefer that file as the resource-profile source of truth whenever it exists. I added a `kubeai-sync-resource-profiles --from-file ...` command, plus a `setup --resource-profiles-file ...` convenience hook, so the user can generate or handwrite one values file and then sync it into the repo-local location that `validate`, `render`, and the Helm install step all agree on.

I intentionally kept a compatibility fallback to `config.yaml.resource_profiles` when no synced KubeAI values file exists. The tests made it obvious that making the synced file mandatory would have turned this into a larger behavior change than the prompt asked for. The resulting model is: synced KubeAI values file first, config fallback second. That still solves the real bug because the new local file now overrides the stale or missing config case, which was the operational pain point. It also keeps the current setup/render path from breaking for users who have not yet adopted the sync step.

The other important detail was invalidation. Because `generated/kubeai/kubeai-values.yaml` is both the synced local source and the rendered Helm values artifact, syncing new resource profiles must invalidate any old plan so `deploy` cannot quietly reuse stale KubeAI artifacts. I handled that by removing `generated/plan.yaml` in the sync path. On the usability side, I tightened namespace messaging in `status` and `deploy` so a namespace mismatch now points users back to the configured `setup --backend kubeai --namespace ...` value instead of surfacing only raw command failure text.

The README changes stay focused on this plumbing fix. I replaced the old “HACK” section with the actual built-in profile names the repo expects, routed the generated file through `python manage.py kubeai-sync-resource-profiles --from-file values-kubeai-local-gpu.yaml`, updated the Helm install example to use `generated/kubeai/kubeai-values.yaml`, and added the preflight `helm list -n kubeai` / `kubectl -n kubeai get pods` check before the KubeAI backend flow. That makes the docs match the code path again, which is the real win here.

Design takeaways:
1. A local generated artifact can safely be the source of truth if there is an explicit sync step and render invalidates stale plans after that sync.
2. Compatibility fallbacks are worth keeping when they preserve an existing happy path without weakening the new preferred source of truth.
3. Namespace-sensitive Kubernetes errors are much more actionable when the CLI points back to the exact namespace-setting command users actually ran.

## 2026-04-19 00:03:50 +0000

Summary of user intent: revise the KubeAI resource-profile patch so the synced canonical input is no longer the same path as the generated render output, while preserving the original bugfix intent and keeping config fallback behavior intact unless the user explicitly adopts the sync flow.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This revision fixed the design seam the prior patch had introduced. Treating `generated/kubeai/kubeai-values.yaml` as both durable input and render output made `render` a hidden state transition, which is exactly the opposite of what a generated directory should mean. The new model is cleaner: `kubeai-values.local.yaml` is the explicit synced local input file, and `generated/kubeai/kubeai-values.yaml` is the rendered output artifact copied from the chosen input source for the current plan. That means a normal render no longer changes future source selection, and the old `config.yaml.resource_profiles` fallback keeps working until the user explicitly adopts the sync flow.

The important compatibility nuance was to preserve fallback semantics after render, not just before it. I changed KubeAI resolution so it prefers the synced local file only when that file exists; otherwise it continues using `config.yaml.resource_profiles`. Because the rendered artifact is now always derived from `deployment["resource_profiles_values"]`, changing config and re-rendering still updates the generated Helm values file when no synced local file has been adopted. Once a synced file exists, it wins intentionally, and that win is now explicit and inspectable rather than an accidental side effect of having rendered once in the past.

I also tightened the non-lossy requirement by preserving raw `resourceProfiles` entries from the synced values document instead of normalizing them down to only the currently understood fields. Validation still only cares about profile names, but rendering now writes the synced values document back out verbatim enough to keep extra supported or unknown keys like `extraField` intact. That felt like the right compromise: preserve structure when the user intentionally synced a Helm values file, but keep the internal config fallback simpler for the legacy path.

Design takeaways:
1. Generated output should be derivable from canonical state, not reused as canonical state itself.
2. A compatibility fallback is only real if normal output generation cannot silently disable it.
3. When syncing user-supplied Helm values, preserving unknown keys is often safer than inventing a lossy normalization layer.

## 2026-04-19 00:07:41 +0000

Summary of user intent: apply the smallest clean fix for the remaining KubeAI regression by making `deploy`/staleness detection notice changes to `kubeai-values.local.yaml`, while preserving the new source-of-truth split and the config fallback behavior.

Model and configuration: Codex (GPT-5-based coding agent), default in-session configuration.

This was a narrow but important follow-up. Once `kubeai-values.local.yaml` became the explicit canonical synced source, it also became a real input to the rendered plan, which means staleness detection had to treat it the same way it already treats `config.yaml`. Without that, `deploy` could keep reusing an older rendered plan after the canonical local file changed, which would undercut the whole point of having a synced source in the first place. The fix was intentionally small: when the backend is `kubeai`, `render_is_stale()` now compares the modification time of `kubeai-values.local.yaml` against the current rendered outputs and forces a rerender if the local file is newer.

The tests here matter more than the code size. I added one direct stale-check test and one deploy-rerender test because they pin two different promises: first, that the CLI recognizes the canonical local file as an input to render freshness, and second, that `deploy` actually consumes the updated canonical file rather than merely reporting “stale” in theory. I kept the existing config-fallback test in place so the repo still proves the legacy path is unaffected when no synced local file exists.

Design takeaways:
1. Once a file becomes canonical input, stale detection must treat it as input everywhere apply-style commands rely on freshness checks.
2. A clean source-of-truth split is only complete when freshness logic follows the same split.
3. Focused tests around stale detection are worth adding because timestamp-based bugs are easy to reintroduce during unrelated CLI cleanup.

## 2026-10-01 19:28:25 -0400

User intent: make Kubernetes inventory, detailed readiness, K3s bootstrap and
KubeAI installation first-class kwconf commands, then fix the real-machine
Compose-to-KubeAI migration failures (stack down left the old recovery identity,
gc threw ProfileMismatch, status hid the active backend, sudo diagnostics implied
an all-clear, and the vLLM pod received served-model-name twice). Validation is
on this VM with fake runtimes; the maintainer will exercise the GPU machine.

Model/configuration: GPT-6 (Codex), Default collaboration mode, standard session
configuration; the exact reasoning-effort setting is not exposed to this agent.
No sub-agents used. No real-host bootstrap/install mutations performed.

I extended the existing kube ModalCLI and KubeManager rather than introducing
another shell implementation. Independent probes retain their own errors; node
GPU facts cannot depend on installed chart configuration because they determine
that configuration. Inventory describes facts; readiness applies dependency
policy. Bootstrap is a provider adapter with a read-only default and explicit
apply; working clusters/kubeconfigs win over installation convenience. The local
K3s restart gate verifies cluster identity, and unknown runtime state never means
quiescent. NVIDIA driver/toolkit installation remains host-distribution work.
Helm 0.17.1 NVIDIA plugin/GFD behavior is retained. KubeAI upgrades share
kubeai_ops and protect tokens in temporary 0600 values files.

The migration failure clarified that runtime teardown and recovery epoch
ownership are separate operations. The chosen ledger rotate command previews
and rechecks quiescence under the publication lock, archives SQLite using its
backup API (including WAL), then atomically clears archived rows and publishes
the new configured backend snapshot in the same database. Keeping the database
inode avoids stranding already-open processes on a renamed old ledger. Archive
publication precedes reset; interruption before reset retains old state and
interruption after commit is a no-op on retry. Active leases and strict runtime
objects block rotation. Catalog/settings files remain authority and untouched.
GC reports a mismatch before mutation; status names both backend identities and
reads old rows with their frozen backend. KubeAI's engine_vllm.go injects the CR
served name before spec.args; an optional flag in the shared arg builder removes
only infer-stack's own served-name emission for KubeAI, preserving Compose and
arbitrary extra arguments.

I am confident in the node/config separation and archive transaction design;
remaining risk is integration behavior on a real K3s host, particularly runtime
PATH discovery, chart rollout timing and configured API routing. Final combined
focused validation passed 213 tests. The full suite passed 1098 tests with 7 skips
and 8 existing cleanup/ResourceWarnings. `ty check ./infer_stack`, the CI flake8
E9/F63/F7/F82 gate, scoped Ruff checks of changed implementation/new tests, and
`git diff --check` passed. Actual entrypoint help tree, all five kube leaves,
ledger rotate, status, gc and both GPU/sudo doctor forms were inspected. A
standalone simulated integration using real Controllers/SQLite and fake runtime
commands exercised acquire/refusal, clean GC refusal, preview/rotation/retry,
archive inspection and subsequent KubeAI acquire successfully. Read-only probes
on the VM's existing CPU K3s cluster retained both node facts and diagnosed the
missing plugin, unready node and unavailable configured API. No setup mutation
was performed. Remaining acceptance is the maintainer's GPU host: inventory and
doctor, bootstrap/install preview then explicit apply where needed, followed by
both detailed and operational doctor checks with the intended namespace/API URL.

Reusable takeaways: prerequisite inventory must precede configuration that
consumes it; teardown does not erase persistence ownership; SQLite history
rotation should preserve live connections and change epochs transactionally.

## 2026-10-01 20:12:00 -0400

User intent: extend the TUI with efficient Kubernetes node/cluster reporting and
controls, then prioritize the real-host review of the previous setup work before
resuming that UI task. Model/configuration: GPT-6 (Codex), Default collaboration
mode; reasoning effort not exposed. Commit attribution requested by the user is
GPT-6.1-Sol. No sub-agents. TUI work was paused and stashed during this review.

The review exposed three product boundary mistakes: admin credential access was
made too broad, provider=k3s reused whatever admin context was selected, and an
optional NVIDIA DaemonSet with no targets counted as unhealthy. Root K3s config
now remains 0600; the invoking user gets a private, atomically refreshed 0600
copy. This accepts certificate refresh maintenance in exchange for preserving
other users' privileges and unrelated kubeconfigs. Explicit K3s provisioning
always scopes reconciliation to that local copy. A worker's local membership is
reported independently from its admin context, and an active agent must match
requested server/name/version before join may return idempotently.

RuntimeClass existence is a cluster fact, not evidence of node handlers. The
inventory now reports observed nvidia-runtime pod startup separately per node;
unknown handlers block detailed readiness rather than implying success. Explicit
bootstrap/install can run tiny node-specific runtime canaries without reserving
a GPU, retaining completed pods as diagnostic evidence. This avoids taking busy
GPU capacity merely to test a handler. Successful startup proves the observed
runtime invocation, not future driver health. Kube setup is deprecated and routes
to the native install leaf so the shell-visible workflows share policy. Readiness
waits report changed states and final generation verification; KubeAI includes
node/image/replica detail with one pod list for nonfatal diagnosis.

Review validation: 207 focused tests passed; the full suite before restoring
the TUI passed 1194 tests with 3 skips and 8 cleanup/ResourceWarnings. An
additional startup-detail regression passed in the review module (15 tests).
Type checks, CI flake8, scoped Ruff, shell syntax and actual entrypoint help
checks passed. Remaining real-host checks include private credential access/
refresh, stale-EKS context isolation, active-worker config inspection, and runtime
canaries on aiq-gpu/namek. No real cluster mutation is part of this work.

## 2026-10-01 20:29:02 -0400

User intent: resume the Kubernetes TUI work after the bootstrap review and
make logical commits. Model/configuration: GPT-6 (Codex), Default collaboration
mode; reasoning effort not exposed. User-requested commit attribution:
GPT-6.1-Sol. No sub-agents.

I separated the review corrections from the TUI changes. The dashboard needs
live scheduling facts cheaply; running full inventory/Helm/API diagnosis every
ledger tick would create avoidable latency and subprocess load. kube.monitor
instead gathers three batched lists, retains independent errors, and computes
scheduled GPU requests across namespaces. It explicitly does not call those
requests utilization. Runtime evidence remains observed pod startup, never a
claim inferred from a cluster RuntimeClass. The cached Cluster tab samples at
least 15 seconds apart only while visible; ledger/instance observation has its
own cadence, and non-overlap guards prevent canceled Textual workers from
leaving duplicate subprocesses behind. Existing node GPU parsing and residency
parsing supply tables and cached Instances data. Doctor is deliberate/on-demand. Log followers pause while their tab is hidden;
returning retains backend action output captured in the meantime. Textual
Expanded/Collapsed messages need their own handlers for immediate gating;
relying on the base Toggled message delayed the update until a timer tick.

Control reuses the existing lifecycle manager: node actions preview affected
workloads, warn about emptyDir loss or Compose ownership handback, confirm, then
recheck before mutation. Apply stays with the controller; Down works for Model
renders without a local gateway file, confirms Model/gateway teardown, retains
leases, and holds the publication lock. Mutations invalidate caches. The endpoint
editor exposes resource_profile rather than physical GPU indices on KubeAI.
Compose behavior and host metrics remain available; local metrics are explicitly
labeled as local on the cluster backend. The tradeoff is bounded polling rather
than persistent Kubernetes watches: this avoids adding a second client/session
lifecycle, and count-based tests pin the request budget. Small retained runtime
canaries prove an observed invocation, not permanent future handler health.

Validation: 79 existing TUI tests and 11 new Kubernetes TUI tests passed (90 total),
including hidden log streams and captured action output. The
final combined full suite passed 1206 tests with 3 skips and 8 cleanup/ResourceWarnings.
A final N/A representation check passed 49 review/readiness tests. Type checks,
CI flake8, scoped Ruff and diff whitespace checks passed. A read-only headless
TUI run against this VM's actual K3s cluster showed two node rows and four pod
rows with no probe errors; screenshot /tmp/infer-stack-kubernetes-tui.svg.
No real Apply/Down/node/bootstrap/install action was invoked. Maintainer
validation remains the GPU hosts' credential refresh, actual agent membership,
canary startup, and confirmed node handoff with real workloads.

## 2026-10-01 20:55:21 -0400

User intent: correct the reviewed KubeAI 40-character Model-name blocker,
refresh runtime canaries on explicit apply, and make GPU-worker join verification
explicit. Keep the long e2e identity and logical commits. Model/configuration:
GPT-6 (Codex), Default collaboration mode, reasoning effort not exposed;
user-requested attribution GPT-6.1-Sol. No sub-agents.

The name budget belonged in model_name_for, shared by rendered Models and gateway
routes. Kubernetes DNS validity alone did not imply CRD validity. Short names
retain their existing identities; overlong names reserve an eight-character
SHA-256 digest of the full served identity and the existing deployment tail.
Simple truncation would merge distinct models sharing a prefix. This changes
only names which the existing KubeAI CRD could not accept. The regression keeps
HuggingFaceTB/SmolLM2-135M-Instruct and checks the two dedicated Model/route
identities together. A first test unnecessarily invoked gateway apply with a
fake HTTP implementation that never completed dynamic registration; interrupted
after 73 preceding passes and restricted this rendering regression to its actual
contract. No real workload was created by that test.

Naming validation: tests/test_leasing_kubeai.py passed all 74 tests, including
short/boundary names, static/dynamic long-prefix collisions, dedicated
separation, and the exact long e2e Model with matching gateway routes. Type
checks, scoped Ruff and whitespace checks passed. The script's long alias is
unchanged. Read-only probes found an existing CPU KubeAI chart and no managed
Models on this VM; its API needs a temporary port-forward before e2e.

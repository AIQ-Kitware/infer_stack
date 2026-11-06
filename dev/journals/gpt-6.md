## 2026-10-12 15:21:56 -0400

User intent: make `catalog suggest` produce a curated, hardware-aware Pareto-like
selection **at every useful size class**, not only at the largest size fitting a
machine; retain both GPT-2 and SmolLM2-135M as debugging fixtures, keep small
Qwen3.5 models (0.8B, 2B, 4B, 9B), HyperQwen 27B, 122B, and distinct Gemma
multimodal options, while making legacy/superseded models opt-in. Remove all
implicit keep-warm behavior: leases keep models running, and releasing the final
lease should stop a model unless its operator explicitly asked for residency.

Model/configuration: GPT-6 (ChatGPT); exact runtime reasoning variant not
exposed. Work conducted against the user-provided infer-stack 2026-10-12 source
snapshot. No remote GPUs/containers were accessed.

I kept the existing two-layer design, hardware-neutral YAML registry plus pure
inventory join, instead of creating benchmark-driven dominance inference without
measurements. Each pool entry now opts into `smoke` or `frontier` selection; all
others remain registered and selectable with `--all` or `--include`. The join
continues to use existing VRAM, count, architecture, and variant-profile gates
but does **not** drop smaller frontier models when a larger one fits. Distinct
Gemma audio and image variants preserve a modality frontier alongside text-only
Qwen. HyperQwen's experimental huge compressed-context / slower Turing fallback
remain opt-in; the fast/long or fast/full class-dependent profiles remain
suggested. FP8 122B is retained alongside the regular 122B as a different
precision/context choice.

Policy is explicit at both ends: `Catalog` parses missing reclaim as `stop`,
`Controller` handles missing deployment reclaim as `stop`, and the suggestion
compiler writes `stop` on generated endpoints/variants. Manual keep-warm
settings remain supported. A deployment can have IDLE historical ledger state
while the controller has already removed the actual engine; tests distinguish
ledger bookkeeping from runtime residency. The default `--apply` is additive,
so older entries are not deleted or silently overwritten. Removing old entries
is a separate operator action, especially when leases are active.

Tests: focused catalog/CLI/controller/lease suite passed (194 tests) with a
local kwconf source checkout and a temporary lightweight ubelt shim because
package download is unavailable in this execution environment. Additional
CLI metadata/access tests passed (20 tests) under `TERM=xterm`. Full suite
was attempted, but the run timed out during control-service tests; one
failure from an earlier `TERM=dumb` run was only ANSI-color expectations.
The remaining risk is real-hardware fitness (catalog estimates, especially
122B and multimodal serving), which requires runtime benchmarking; no new
capability/speed measurements were invented.

Reusable takeaways:
- Selection of a curated frontier and hardware-fit checks are independent
  filters: fitting a larger model never dominates a smaller one automatically.
- Lifecycle defaults belong in both the catalog parser and controller fallback,
  not just generated suggestions.
- Keep the registry complete and the automatic surface compact, and make
  a migration of existing user catalogs opt-in rather than destructive.

Final validation update (same session): GPT-2 was assigned a completions probe
protocol because its pretrained checkpoint has no chat template. The combined
catalog/CLI/controller/Compose/KubeAI/parity/route-registry suite passed with
431 tests, 6 skips and 2 exclusions. Both excluded Compose tests were run
separately against a pristine `git archive HEAD` of the user-provided source
and failed identically there, demonstrating that they predate this change:
`test_converge_to_empty_keeps_the_front_door` and
`test_converge_no_change_does_not_prompt`. The earlier full suite timeout is
not treated as a successful full-suite run. Work remains to fix the baseline
Compose diff/prompt behavior independently.

## 2026-11-05 19:17:00 -0500

User intent: expose Qwen3.8/HyperQwen's native reasoning effort selection from
pi's `/effort` UI through LiteLLM without using a reasoning-token cutoff or
needing to restart the inference engine for each change.

Model/configuration: GPT-6 (ChatGPT); exact runtime reasoning effort not
exposed. Worked against the user-uploaded infer-stack source snapshot from
2026-11-05; no remote inference service or GPU container accessed.

I chose an opt-in `reasoning_efforts` declaration on a vLLM chat endpoint,
carried as per-alias served metadata rather than process-wide runtime settings.
This preserves the endpoint's structural deployment identity and lets several
aliases sharing one vLLM process advertise their own request contract. The
suggestion pool now supplies `[low, medium, xhigh]` for the HyperQwen model and
all of its suggested variants. The catalog validates the generic OpenAI effort
vocabulary but intentionally leaves model-template compatibility to the
operator; HyperQwen's native supported set is explicitly narrow.

Both static and dynamic LiteLLM routes declare `supports_reasoning`,
`reasoning_effort_levels`, and `allowed_openai_params: [reasoning_effort]`.
The first two permit Pi's LiteLLM provider to expose the selector; the third
makes the nonstandard custom-model route accept and forward the request field.
Dynamic reconciliation checks all three fields for drift. Remembered routes
preserve the per-alias effort contract after a deployment disappears, and a
published catalog can provide missing metadata for older live deployments.
The new `docs/reasoning-effort.md` shows pi install/refresh and a curl smoke
request. No default reasoning budget is changed and no new vLLM launcher
argument is introduced.

Tests: 163 passed in the focused catalog, suggestion, context metadata,
route registry, dynamic routing, and new effort suites, using an in-memory
`ubelt.Path` shim because this runtime lacks that dependency. Tests include
rendered static LiteLLM YAML and simulated dynamic route add/repair/no-churn.
An earlier broader run exposed missing `kwconf` in this offline environment
and two previously documented baseline Compose failures, so it is not claimed
as a successful full suite. Real HyperQwen template rendering through the
pinned vLLM/LiteLLM containers and pi client remains a follow-up live smoke
check; `reasoning_effort` forwarding is supported by current vLLM and pi
provider documentation but untested against this deployed image.

Reusable takeaways:
- Model-specific reasoning controls belong to the endpoint's request contract,
  not process launch identity, when a gateway can forward the field per call.
- Explicit supported levels and an explicit forwarding allowlist are both
  needed for clients discovering a custom LiteLLM model.
- Reconciliation should consider advertised capability drift as well as
  routing destinations, but never churn unchanged legacy routes.

## 2026-11-05 22:17:35 -0500

Model: GPT-6 (thinking configuration not exposed).

The user wants Qwen3.8-Flash-Next integrated into infer-stack on their dual-3090
workstation, ideally using only one card so the second stays available. They
provided infer_stack and dual-3090 source archives; the upstream project also
has a separate single-3090 runtime. I inspected the upstream single-card image
entrypoint and Docker launcher rather than trying to run the two-card model
with tensor_parallel_size=1 (which would not work).

infer-stack already has generic vLLM image/command/env/mounts support and
single-card placement, so this integration should remain an ordinary managed
catalog endpoint instead of adding a special inference engine. The missing
container privileges are added through a narrowly validated, Compose-only
`runtime.compose` mapping (host IPC, SYS_PTRACE, memlock/stack). The existing
VRAM size floor assumes a whole checkpoint sits on GPUs; a new explicit
`placement.weight_floor: false` override is justified for RAM-offloaded
runtimes, retaining the per-GPU eligibility minimum. This opt-in avoids making
all models less safely placeable.

The remaining unknown is performance and startup on the user's i9-11900KF
platform; all verification here is static/offline. The example also avoids
claiming a hard RAM cap where no Compose memory limit was set. Future variants
might provide a generic explicit host memory cap, but that should be designed
with per-host contention and leasing in mind, not smuggled into this example.

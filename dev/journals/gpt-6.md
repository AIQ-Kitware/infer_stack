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

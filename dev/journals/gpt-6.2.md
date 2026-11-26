## 2026-11-24 19:17:38 -0500

Model: GPT-6.2 (ChatGPT; further configuration not available).

User intent: add a fully managed Maya Inkling 2 Mini endpoint to infer-stack,
with one RTX 3090, long-context agent work, and decode speed competitive with
HyperQwen. The existing llama.cpp profile is not sufficient for this purpose.

The generic Compose launcher already expresses every required lifecycle
and routing field, so the change is a local CUDA image, a small adapter for
Maya's upstream JSON profiles, and opt-in catalog entries. Avoided a new
engine enum or a model-specific branch in the compiler. The image accepts an
exact Maya commit at build time; the source ref is recorded in an OCI label.
The default 196608-token profile leaves generation room after a 163840-token
prompt; the upstream 262144-token DSpark and 344064-token non-draft profiles
are additional options. Text-only keeps vision-related dependencies out of
the initial deployment. We preserve the upstream FP8/tuning config rather
than guessing new CUDA flags. The exact Q4_K_XL file matters (not Q4_K_M), as
it contains the expert layout this engine expects. Reasoning metadata omits
`minimal`, which Maya's HTTP frontend currently aliases to off, and `max`,
which aliases `xhigh`. Model lifecycle belongs to infer-stack, not Maya's
load/unload API.

Validation: nine launcher-only pytest cases passed with no extra dependencies.
The 56 targeted Maya, leasing suggestion and effort tests also passed using
an import-only empty `ubelt` module because the environment lacks the actual
package; the existing test fixture supplies explicit path roots, so no
ubelt API was exercised. This is not a full installed-environment test.
Docker, CUDA, upstream Git checkout and model downloads could not be executed
in this environment. GPU acceptance, especially long occupied-context speed,
tool call round trips, and reasoning effort forwarding, is still required.
The implementation and host commands are in `docs/maya-inkling-mini.md`.

Design takeaway: maintain the serving protocol and placement/lifecycle
contract in the catalog; keep engine-specific setup in its image. Separate
upstream context capacity from measured decode rate at occupied context.

## 2026-11-24 19:23:14 -0500

Model: GPT-6.2 (ChatGPT; further configuration not available).

User intent: do not make infer-stack users reason about Docker compilation or
upstream Git revisions in order to acquire a catalog endpoint. Pin a default
Maya revision and make image preparation part of the existing acquisition path.

Upstream's Docker recipe does not include `maya-ik2`; I found no usable
published image for this fork. Use a bundled build context keyed by a
versioned image reference, not a catalog-supplied Dockerfile or mutable
`main`. The source pin is upstream 27daec5e608aa59dab45184e82a1f0611ea7c1aa,
whose Inkling guide and configuration I checked. It is not a verified GPU
build. The image is built only if absent and preparation runs before removal
of existing containers. Registry images keep the existing pull path.
`config publish --pull` also handles bundled recipes. Package resources allow
a wheel install to build without a repository checkout. Build logs use the
existing Docker runner; the stderr callback now drains concurrently with
stdout to avoid deadlocking on large compiler output. A cold build holds the
same publication lock as an image pull, so it can delay other leasing changes.
A prebuilt registry image would eliminate that wait, but this patch neither
requires registry credentials nor claims that such an image already exists.

Validation: 172 targeted tests passed, 4 skipped, 2 excluded after confirming
the same failures on the previous source: test_converge_to_empty_keeps_the_front_door
and test_converge_no_change_does_not_prompt. As before, an import-only ubelt
placeholder was required in this environment; no ubelt operations were needed
by these tests. Built a wheel and checked it contains the recipe and launcher.
Docker and GPU execution are still unavailable. Future upstream or build
recipe changes need a new image tag; apt/pip/base-image versions are not
fully content-pinned. The normal acquisition timeout still applies to model
startup/download after the build.

## 2026-11-24 19:42:00 -0500

Model: GPT-6.2 (ChatGPT; further configuration not available).

User intent: make building a local runtime image an explicit infer-stack
operation when no registry artifact is available. The Dockerfile and launcher
must travel with the package, not require a source checkout. Keep the existing
automatic cold-acquire path as a convenience, but allow preparation before an
endpoint even exists.

`infer-stack image list` exposes stable recipe names and versioned tags;
`infer-stack image build maya-inkling-mini` resolves the same recipe and
ensures the image exists, using the normal Docker environment. The shared
`runtime_images.ensure_managed_image` is called by Compose and the CLI so they
cannot drift on tags, contexts or timeouts. The explicit command does not
instantiate a backend, touch the lease ledger, or allocate GPUs. It does not
hold the publication lock, so prebuilding is a way to avoid delaying other
leasing operations. Concurrent callers can still request the same build;
no new lock or mutable image tag protocol is introduced. Build recipes remain
code-owned and packaged; a catalog does not get to execute arbitrary builds.

Validation: 22 targeted tests passed with an import-only ubelt placeholder;
no ubelt operations exercised. The new CLI dispatch test is included but
could not be run here because kwconf is unavailable. Wheel contents checked
against the actual recipe/launcher and CLI source. No actual Docker or CUDA
build was performed; real host acceptance remains required.

## 2026-11-24: Maya GGUF layout hotfix

The user's real container reached the native CUDA executable, then failed
with `maya-ik2: no key inkling2.block_count`. Unsloth changed its root GGUFs
on November 23 to mainline llama.cpp's `inkling2_loop` layout; its original
Maya-compatible GGUF remains in `Old_Support` with a `.gguf_file` suffix.
Pin the model download to 8078d149286eec7193795413b9e0643da6d11fa1 and the
official drafter download to 1fb4134b9a74f9681054f4ef82e4bf1f458a2337. Use a
separate cache path for the old layout and leave the existing mainline file
untouched. Check architecture and logical block count with Maya's own GGUF
reader. Preserve native stderr in container logs, with a checked patch of
the pinned upstream source after the CUDA build step. Bump the bundled
image recipe to r2; the Docker build cache can reuse the compiler layers.
Existing r1 catalog entries need their image reference updated. Existing
local files are not fully hash-verified. No change to context or DSpark.

Validation: 13 focused launcher tests passed (pytest addopts overridden to
avoid unavailable xdoctest). Docker/CUDA and real weight downloads remain
host-side validation; the upstream GGUF reader and HF file paths were
inspected but real model bytes were not downloaded here.

## 2026-11-25 14:43:00 -0500

Model: ChatGPT (GPT-6.2; serving configuration not independently identifiable).
The user wants a simple kwconf benchmark command taking endpoints, serial
lease/release, portable append-only observations and a readable report across
machines. Their two 3090s may have different PCIe links; GPU model alone is not
a comparison identity. Implemented a local Compose/LiteLLM runner using the
existing controller, releasing only its own lease and evicting newly idle
deployments to prevent a keep-warm policy from blocking the next run. Raw
observations distinguish target vs reported tokens and requested vs observed
prefix reuse. Host/GPU identity, upstream PCI path, actual image ID where
available and under-load telemetry preserve the differences otherwise lost in
an aggregate. Reports keep identities separate and avoid pretending client-side
stream timings are engine-internal counters. No changes to core lease semantics.
Risks: new agents can join a deployment during the benchmark, mutable weights
are not attested, and tokenizer calibration is approximate. Scope intentionally
limited to local Compose; remote scheduling needs server-side hardware facts.

## 2026-11-25 15:17:07 -0500 — Catalog discovery and shortlist

Model: ChatGPT (GPT-6.2; serving configuration not independently identifiable).
Promoted Maya's 192/256 Ki-token DSpark profiles after operator-reported
262144-token readiness; retained the 336 Ki-token no-drafter variant as opt-in.
Gemma E2B preserves a distinct small multimodal option; E4B, 26B and 31B move
out of the frontier. Older Qwen2.5/Llama/Mistral/GPT-OSS entries were already
optional. Kept the separate Qwen small/large classes and smoke tests. This is
curation, not proof of measured performance dominance.

The browser compiles the registry against the active backend's hardware and
placement restrictions, includes optional profiles, and imports the exact
selected endpoint plus its model through a non-overwriting merge. CLI import
shares this path; literal packaged presets remain unchanged. Import is a
catalog operation, not deployment. Import & acquire goes through the existing
controller, where Compose prepares bundled images automatically before startup.
No new Docker build logic in the TUI. KubeAI excludes unsupported custom runtime
fields from the browser. Catalog files are re-read on import to check conflicts.

Validation: 62 targeted tests passed (ubelt import-only shim and xdoctest
addopts disabled). Textual and kwconf unavailable; new headless browser and CLI
import tests are included but unexecuted here. No new GPU tests performed.

## 2026-11-25 — Explicit registry experimentation

Model: ChatGPT (GPT-6.2; serving configuration not independently identifiable).
The operator wants the registry browser to expose known configurations even
when detected hardware does not satisfy their suggested resource or architecture
constraints. The existing DeepSeek V4 Flash 0731 TP4 SM120 recipe was already
present in the supplied snapshot but hidden on smaller hosts. Explicit discovery
now renders every suggestion-pool profile, retaining the exact fitting recipe
where one exists and rendering declared defaults without fabricated GPU pins
otherwise. Fit and backend support are separate advisory metadata. Normal
suggestions retain their original hardware and tier gates; acquisition retains
normal controller placement checks. CLI import uses the same full-registry
path. The TUI presents warnings and a scrollable YAML preview, and can import
then open the endpoint editor to adjust settings, including the declared VRAM
floor and automatic cached-weight floor. Custom-launcher startup remains owned
by Compose. A second, deliberately opt-in DeepSeek TP2/32K profile for two
96-GiB SM120 cards is a hypothesis, not a validated deployment. The previous
TP4 readiness result is preserved; neither recipe is intended for one 3090.
Standalone packaged presets still use the existing separate CLI import path.
Validation: 60 focused registry/suggestion/Maya tests pass using an import-only
ubelt shim; the Textual suite is skipped because Textual is missing, kwconf is
also unavailable, and no live GPU execution was attempted. CLI and TUI
regressions are included for execution with the normal project dependencies.

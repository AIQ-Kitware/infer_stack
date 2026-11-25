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

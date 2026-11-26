## 2026-11-25 17:57:55 -0500

Model: ChatGPT (GPT-6.2); exact serving/thinking configuration not exposed.

The user asked to audit the documentation and turn the increasingly long README
into a friendly introduction with a useful quickstart. Reconstructed the latest
provided snapshot (0b637511df60) with the subsequent experimental-registry,
live-benchmark, benchmark-dashboard, benchmark-help and engine-logs overlays,
so the new guides preserve their changes. The README had become an 882-line
mixture of first-run instructions, model-specific settings, operation details,
and Kubernetes setup. The revised 159-line README prioritizes a concrete
Linux/Compose first run (including a GPU-free alternative), the shared API and
TUI, and scoped lease release. An explicit security warning precedes starting
any containers. Use task-oriented documentation navigation, not a growing
feature catalogue in the README.

Preserve the substantive reference material in catalog, operations, Compose,
and model-profile guides. Consolidate cluster guidance into the existing
KubeAI and cluster setup documents rather than create another duplicate. Keep
existing file paths stable. Add a repository docs index with separate links to
current guides, model recipes, limitations, architecture and historical work.
The existing Sphinx manual links to the repository index; expanding the Sphinx
source tree or migrating the documentation framework is deliberately deferred.
Remove stale development follow-ups and a stray manual-test note from the
leasing walkthrough; correct shell continuations/placeholders and the obsolete
require-generation instruction. Match the CLI's help epilog to the quickstart.
No serving, placement or lease behavior changes.

Important details: release the lease from the env file rather than recommending
release --all to first-time users; Compose UI has authentication disabled and
host-published ports, and enabling the reverse proxy does not unpublish the
other front-door ports. The shell environment file renderer now quotes values,
so the known-limitations text should not claim otherwise; credentials and
trusted input requirements still apply. Keep the original limitation anchor.
The model registry is editorial, and discovery warnings do not relax admission.
These distinctions prevent a simple quickstart from promising security,
compatibility, performance, or exclusive GPU ownership it cannot guarantee.

Validation: 96 local links/heading anchors and 75 Bash blocks checked; both
changed Python modules parse; whitespace diff check passed. The actual catalog
compiler generated and validated the smoke endpoint on four simulated GPU
classes, and the mock fragment and manual YAML example validated. 43 registry
and suggestion tests passed with an import-only ubelt placeholder. No full
CLI dispatch, Sphinx build or real Docker/GPU quickstart was exercised because
kwconf/ubelt, Sphinx and Docker are unavailable. This pass audits the README and
its connected user guides, not every historical planning document.

Reusable takeaways: make the quickstart teach the ordinary lifecycle, not the
exception path; preserve a single canonical home for advanced instructions;
validate example endpoint identities and shell syntax against the implementation.

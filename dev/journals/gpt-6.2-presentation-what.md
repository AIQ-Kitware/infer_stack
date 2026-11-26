## 2026-11-25 20:11:52 -0500

Model: ChatGPT (GPT-6.2; configuration not exposed).

The user requested a standalone what deck in the presentation suite. Added
an eight-slide conceptual introduction using the existing shared layout and
pinned dependency set. The deck defines infer-stack, separates the control
and inference-request paths, introduces model/endpoint/deployment/lease,
illustrates distinct profiles and compatible shared leases, and explains the
supported backends, scope, and limitations. Example endpoint names are
illustrative, not claims about built-in registry names or tested capacity.
I chose a short conceptual talk rather than another tutorial or adoption
pitch. This keeps it suitable as a first introduction, or as a preface to
one of the other talks. The existing decks and benchmark data are untouched.

Validation: all six decks (58 slides) build. All 100 slide and speaker-note
XML files of the existing five decks are byte-identical to their prior
builds. The new deck has eight source-noted slides, was exported to PDF and
rendered for visual review, and passed layout overlap and overflow checks.
No new dependencies or binaries in the repository overlay. No live serving
operations were performed; product claims follow the supplied repository
and subsequent documentation overlays.
